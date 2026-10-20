"""Wake-on-LAN before requests to a model host that sleeps when idle."""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, ProviderConfig, WakeOnLanConfig
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.factory import make_provider, provider_signature
from nanobot.providers.wake_on_lan import (
    WakeOnLanError,
    api_base_host_port,
    ensure_model_host_awake,
    send_magic_packet,
)
from nanobot.webui.settings_models import update_provider_settings


class _HostProvider(LLMProvider):
    def __init__(self, api_base: str) -> None:
        super().__init__(api_base=api_base, provider_name="ollama")
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "llama3.2"


def _wake(**overrides: Any) -> WakeOnLanConfig:
    values: dict[str, Any] = {"mac": "aa:bb:cc:dd:ee:ff", "network_delay_s": 0}
    values.update(overrides)
    return WakeOnLanConfig.model_validate(values)


def test_wake_on_lan_config_normalizes_mac_and_omits_unset_provider_field() -> None:
    dashed = WakeOnLanConfig.model_validate({"mac": "AA-BB-CC-DD-EE-FF"})
    cisco = WakeOnLanConfig.model_validate({"mac": "aabb.ccdd.eeff"})
    plain = WakeOnLanConfig.model_validate({"mac": "AABBCCDDEEFF"})

    assert dashed.mac == cisco.mac == plain.mac == "aa:bb:cc:dd:ee:ff"
    assert dashed.broadcast == "255.255.255.255"
    assert dashed.port == 9
    assert dashed.timeout_s == 90.0
    assert "wakeOnLan" not in ProviderConfig().model_dump(by_alias=True)

    dumped = ProviderConfig(wake_on_lan=dashed).model_dump(by_alias=True)
    assert dumped["wakeOnLan"] == {
        "mac": "aa:bb:cc:dd:ee:ff",
        "broadcast": "255.255.255.255",
        "port": 9,
        "timeoutS": 90.0,
        "networkDelayS": 15.0,
    }


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"mac": "not-a-mac"}, "MAC address"),
        ({"mac": "aa:bb:cc:dd:ee:ff", "broadcast": "example.com"}, "IP address"),
        ({"mac": "aa:bb:cc:dd:ee:ff", "port": 0}, "port"),
        ({"mac": "aa:bb:cc:dd:ee:ff", "timeoutS": 0}, "timeoutS"),
    ],
)
def test_wake_on_lan_config_rejects_invalid_targets(value: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        WakeOnLanConfig.model_validate(value)


def test_api_base_host_port_uses_the_provider_endpoint() -> None:
    assert api_base_host_port("http://192.168.1.20:11434/v1") == ("192.168.1.20", 11434)
    assert api_base_host_port("https://example.com/v1") == ("example.com", 443)
    assert api_base_host_port("192.168.1.20:11434") == ("192.168.1.20", 11434)
    assert api_base_host_port("http://[::1]:11434/v1") == ("::1", 11434)
    assert api_base_host_port("  ") is None
    assert api_base_host_port(None) is None


def test_send_magic_packet_reaches_the_configured_udp_port() -> None:
    mac = "00:11:22:33:44:55"
    expected = b"\xff" * 6 + bytes.fromhex("001122334455") * 16
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    port = sock.getsockname()[1]
    try:
        send_magic_packet(mac, "127.0.0.1", port)
        data, _addr = sock.recvfrom(256)
    finally:
        sock.close()

    assert data == expected
    assert len(data) == 102


@pytest.mark.asyncio
async def test_wake_waits_out_the_network_delay_before_using_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = time.monotonic()
    probes = {"n": 0}
    sends: list[tuple[Any, ...]] = []

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        probes["n"] += 1
        if probes["n"] == 1:
            return None
        return True if time.monotonic() - started >= 0.25 else None

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *args: sends.append(args),
    )

    await ensure_model_host_awake(
        _wake(network_delay_s=0.3, timeout_s=5),
        "http://192.0.2.30:11434/v1",
        provider_name="ollama",
    )

    assert time.monotonic() - started >= 0.3
    assert sends
    assert probes["n"] == 2


@pytest.mark.asyncio
async def test_awake_open_port_does_not_send_a_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    sends: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *args: sends.append(args),
    )

    async def _accept(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_accept, "127.0.0.1", 0)
    sockets = server.sockets or []
    port = sockets[0].getsockname()[1]
    try:
        await ensure_model_host_awake(_wake(), f"http://127.0.0.1:{port}/v1", provider_name="ollama")
    finally:
        server.close()
        await server.wait_closed()

    assert sends == []


@pytest.mark.asyncio
async def test_awake_refused_port_does_not_send_a_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    sends: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *args: sends.append(args),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    await ensure_model_host_awake(_wake(), f"http://127.0.0.1:{port}/v1")

    assert sends == []


@pytest.mark.asyncio
async def test_unreachable_host_is_woken_until_the_api_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([None, False, True])
    sends: list[tuple[Any, ...]] = []
    notes: list[str] = []

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        return next(states)

    async def note(message: str) -> None:
        notes.append(message)

    async def instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *args: sends.append(args),
    )
    monkeypatch.setattr("nanobot.providers.wake_on_lan.asyncio.sleep", instant)

    await ensure_model_host_awake(
        _wake(timeout_s=5),
        "http://192.0.2.21:11434/v1",
        provider_name="ollama",
        notify=note,
    )

    assert sends == [("aa:bb:cc:dd:ee:ff", "255.255.255.255", 9)]
    assert notes
    assert "ollama (192.0.2.21:11434)" in notes[0]
    assert "Wake-on-LAN" in notes[0]


@pytest.mark.asyncio
async def test_unreachable_host_reports_failure_after_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sends: list[int] = []

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        return None

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *_args: sends.append(1),
    )

    with pytest.raises(WakeOnLanError, match="did not accept connections") as raised:
        await ensure_model_host_awake(
            _wake(timeout_s=0.05, broadcast="192.0.2.255"),
            "http://192.0.2.22:11434/v1",
        )

    assert sends
    assert "192.0.2.255:9" in str(raised.value)


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_wake(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"probes": 0, "sends": 0}

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        state["probes"] += 1
        await asyncio.sleep(0.02)
        return None if state["probes"] == 1 else True

    def send(*_args: Any) -> None:
        state["sends"] += 1

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr("nanobot.providers.wake_on_lan.send_magic_packet", send)
    config = _wake()
    api_base = "http://192.0.2.23:11434/v1"

    await asyncio.gather(
        ensure_model_host_awake(config, api_base),
        ensure_model_host_awake(config, api_base),
    )

    assert state["sends"] == 1
    assert state["probes"] == 2


def test_provider_signature_and_factory_keep_wake_on_lan() -> None:
    base = {
        "providers": {"ollama": {"apiBase": "http://192.168.1.20:11434/v1"}},
        "modelPresets": {"primary": {"provider": "ollama", "model": "llama3.2"}},
        "agents": {"defaults": {"modelPreset": "primary"}},
    }
    changed = {
        **base,
        "providers": {
            "ollama": {
                "apiBase": "http://192.168.1.20:11434/v1",
                "wakeOnLan": {"mac": "aa:bb:cc:dd:ee:ff", "broadcast": "192.168.1.255"},
            }
        },
    }

    assert provider_signature(Config.model_validate(base)) != provider_signature(
        Config.model_validate(changed)
    )
    provider = make_provider(Config.model_validate(changed))
    assert provider.wake_on_lan is not None
    assert provider.wake_on_lan.mac == "aa:bb:cc:dd:ee:ff"
    assert provider.wake_on_lan.broadcast == "192.168.1.255"
    assert provider.api_base == "http://192.168.1.20:11434/v1"
    assert provider.wake_api_base == "http://192.168.1.20:11434/v1"


def test_provider_settings_update_preserves_wake_on_lan() -> None:
    config = Config.model_validate({
        "providers": {
            "ollama": {
                "apiBase": "http://192.168.1.20:11434/v1",
                "wakeOnLan": {"mac": "aa:bb:cc:dd:ee:ff"},
            }
        }
    })

    changed, _restart = update_provider_settings(
        config,
        {
            "provider": ["ollama"],
            "apiBase": ["http://192.168.1.21:11434/v1"],
        },
    )

    wake = config.providers.ollama.wake_on_lan
    assert changed is True
    assert wake is not None
    assert wake.mac == "aa:bb:cc:dd:ee:ff"
    assert config.providers.ollama.api_base == "http://192.168.1.21:11434/v1"


@pytest.mark.asyncio
async def test_chat_entries_skip_probe_without_wake_config(monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(*_args: Any, **_kwargs: Any) -> bool | None:
        raise AssertionError("probed a provider without wake-on-lan")

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", explode)
    provider = _HostProvider("http://192.168.1.20:11434/v1")

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])
    streamed = await provider.chat_stream_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == streamed.content == "ok"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_chat_entries_wake_before_the_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    sends: list[tuple[Any, ...]] = []
    notes: list[str] = []
    states = iter([None, True, None, True])

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        return next(states)

    async def note(message: str) -> None:
        notes.append(message)

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *args: sends.append(args),
    )
    provider = _HostProvider("http://192.168.1.20:11434/v1")
    provider.wake_on_lan = _wake(timeout_s=5)
    messages = [{"role": "user", "content": "hi"}]

    response = await provider.chat_with_retry(messages=messages, on_retry_wait=note)
    streamed = await provider.chat_stream_with_retry(messages=messages, on_retry_wait=note)

    assert response.content == streamed.content == "ok"
    assert provider.calls == 2
    assert len(sends) == 2
    assert len(notes) == 2
    assert "192.168.1.20:11434" in notes[0]


@pytest.mark.asyncio
async def test_connection_retry_wakes_a_host_that_sleeps_after_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sends: list[int] = []
    states = iter([True, None, True])

    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        return next(states)

    async def instant(_delay: float) -> None:
        return None

    class _Flaky(_HostProvider):
        def __init__(self) -> None:
            super().__init__("http://192.0.2.25:11434/v1")
            self.wake_on_lan = _wake(timeout_s=5)

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="connection error",
                    finish_reason="error",
                    error_kind="connection",
                    error_should_retry=True,
                )
            return LLMResponse(content="ok")

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr(
        "nanobot.providers.wake_on_lan.send_magic_packet",
        lambda *_args: sends.append(1),
    )
    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", instant)
    provider = _Flaky()

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert provider.calls == 2
    assert sends == [1]


@pytest.mark.asyncio
async def test_chat_entries_do_not_call_the_model_when_wake_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def probe(_host: str, _port: int, _timeout: float) -> bool | None:
        return None

    monkeypatch.setattr("nanobot.providers.wake_on_lan.probe_tcp_port", probe)
    monkeypatch.setattr("nanobot.providers.wake_on_lan.send_magic_packet", lambda *_args: None)
    provider = _HostProvider("http://192.0.2.24:11434/v1")
    provider.wake_on_lan = _wake(timeout_s=0.05)
    messages = [{"role": "user", "content": "hi"}]

    response = await provider.chat_with_retry(messages=messages)
    streamed = await provider.chat_stream_with_retry(messages=messages)

    assert provider.calls == 0
    assert response.finish_reason == streamed.finish_reason == "error"
    assert response.error_should_retry is streamed.error_should_retry is False
    assert "did not accept connections" in (response.content or "")
    assert "did not accept connections" in (streamed.content or "")
