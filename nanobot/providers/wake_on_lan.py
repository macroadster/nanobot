"""Wake a sleeping model host before a provider request.

The provider API host is probed with a short TCP connection. An accept or a
refusal means the computer is already awake, so the request proceeds. No
answer means the computer may be asleep: send a Wake-on-LAN magic packet and
poll until the API port accepts a connection. A refusal during that wait is
not success, because the operating system answers before the model server is
listening.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from loguru import logger

from nanobot.config.schema import WakeOnLanConfig

Notify = Callable[[str], Awaitable[None]]
_WakeKey = tuple[str, str, int, str, int]

PROBE_TIMEOUT_S = 1.0
RESEND_INTERVAL_S = 5.0
_FAST_PROBE_S = 0.2
_RETRY_PAUSE_S = 1.0

_inflight: dict[int, dict[_WakeKey, asyncio.Task[None]]] = {}
_warned_bases: set[str] = set()


class WakeOnLanError(RuntimeError):
    """The model host did not accept connections after Wake-on-LAN."""


def api_base_host_port(api_base: str | None) -> tuple[str, int] | None:
    """Return the TCP host and port of a provider API base URL."""
    raw = (api_base or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname
    if not host:
        return None
    try:
        parsed_port = parsed.port
    except ValueError:
        return None
    if parsed_port is not None:
        return host, parsed_port
    return host, 443 if parsed.scheme == "https" else 80


def magic_packet(mac: str) -> bytes:
    """Build a Wake-on-LAN magic packet for a canonical MAC address."""
    payload = bytes.fromhex(mac.replace(":", ""))
    if len(payload) != 6:
        raise ValueError("mac must be a canonical MAC address")
    return b"\xff" * 6 + payload * 16


def send_magic_packet(mac: str, broadcast: str, port: int) -> None:
    """Send one Wake-on-LAN magic packet to a configured IP and UDP port."""
    packet = magic_packet(mac)
    family = (
        socket.AF_INET6
        if ipaddress.ip_address(broadcast).version == 6
        else socket.AF_INET
    )
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        if family == socket.AF_INET:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (broadcast, port))
    finally:
        sock.close()


async def probe_tcp_port(host: str, port: int, timeout: float) -> bool | None:
    """Return True if the port accepts, False if it refuses, None if it does not answer."""
    if timeout <= 0:
        return None
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
    except OSError as exc:
        if _connection_refused(exc):
            return False
        return None
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


async def ensure_model_host_awake(
    config: WakeOnLanConfig,
    api_base: str | None,
    *,
    provider_name: str | None = None,
    notify: Notify | None = None,
) -> None:
    """Probe the provider API and wake its computer when the probe gets no answer.

    Concurrent callers for the same target share one wake attempt.
    """
    target = api_base_host_port(api_base)
    if target is None:
        warned = api_base or ""
        if warned not in _warned_bases:
            _warned_bases.add(warned)
            logger.warning(
                "Wake-on-LAN is configured but the provider API base {!r} has no host to probe",
                api_base,
            )
        return

    host, port = target
    key = (config.mac, config.broadcast, config.port, host, port)
    loop_tasks = _inflight.setdefault(id(asyncio.get_running_loop()), {})
    task = loop_tasks.get(key)
    if task is None or task.done():
        task = asyncio.create_task(
            _wake_if_unreachable(
                config,
                host,
                port,
                provider_name=provider_name,
                notify=notify,
            ),
            name=f"wake-on-lan-{host}",
        )
        loop_tasks[key] = task

        def _clear(done: asyncio.Task[None]) -> None:
            if loop_tasks.get(key) is done:
                loop_tasks.pop(key, None)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(_clear)
    await asyncio.shield(task)


async def _wake_if_unreachable(
    config: WakeOnLanConfig,
    host: str,
    port: int,
    *,
    provider_name: str | None,
    notify: Notify | None,
) -> None:
    deadline = time.monotonic() + config.timeout_s
    if await _computer_answered(host, port, deadline):
        return
    await _wait_until_api_accepts(
        config,
        host,
        port,
        deadline,
        provider_name=provider_name,
        notify=notify,
    )


async def _computer_answered(host: str, port: int, deadline: float) -> bool:
    timeout = min(PROBE_TIMEOUT_S, max(0.0, deadline - time.monotonic()))
    return await probe_tcp_port(host, port, timeout) is not None


async def _wait_until_api_accepts(
    config: WakeOnLanConfig,
    host: str,
    port: int,
    deadline: float,
    *,
    provider_name: str | None,
    notify: Notify | None,
) -> None:
    label = _host_label(provider_name, host, port)
    last_send = 0.0
    first_sent_at: float | None = None
    announced_delay = False
    send_error: str | None = None
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise WakeOnLanError(_timeout_message(label, config, send_error))
        if first_sent_at is not None and config.network_delay_s > 0:
            # The NIC can answer once before the rest of the network is usable.
            remaining_delay = first_sent_at + config.network_delay_s - now
            if remaining_delay > 0:
                if not announced_delay:
                    logger.info(
                        "Waiting {:.0f}s for {} network after Wake-on-LAN",
                        config.network_delay_s,
                        label,
                    )
                    announced_delay = True
                pause = min(remaining_delay, deadline - now)
                if pause <= 0:
                    raise WakeOnLanError(_timeout_message(label, config, send_error))
                await asyncio.sleep(pause)
                continue
        if now - last_send >= RESEND_INTERVAL_S:
            send_error = await _send_packet(config)
            sent_at = time.monotonic()
            last_send = sent_at
            if first_sent_at is None and send_error is None:
                first_sent_at = sent_at
            await _notify_waiting(notify, label, deadline - time.monotonic())
            if first_sent_at == sent_at and config.network_delay_s > 0:
                continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WakeOnLanError(_timeout_message(label, config, send_error))
        started = time.monotonic()
        state = await probe_tcp_port(host, port, min(PROBE_TIMEOUT_S, remaining))
        if state is True:
            logger.info("Model host {} accepted a connection after Wake-on-LAN", label)
            return
        # A fast refusal or DNS failure did not wait. Pause so the poll does not
        # spin while the operating system is up and the model server is starting.
        if state is False or time.monotonic() - started < _FAST_PROBE_S:
            pause = min(_RETRY_PAUSE_S, max(0.0, deadline - time.monotonic()))
            if pause > 0:
                await asyncio.sleep(pause)


def _connection_refused(exc: OSError) -> bool:
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, ConnectionRefusedError):
        return True
    winerror = getattr(exc, "winerror", None)
    return exc.errno in {61, 111} or winerror == 10061


async def _send_packet(config: WakeOnLanConfig) -> str | None:
    try:
        await asyncio.to_thread(
            send_magic_packet,
            config.mac,
            config.broadcast,
            config.port,
        )
    except OSError as exc:
        logger.warning(
            "Wake-on-LAN packet to {}:{} failed: {}",
            config.broadcast,
            config.port,
            exc,
        )
        return str(exc)
    logger.info(
        "Sent Wake-on-LAN packet to {}:{}",
        config.broadcast,
        config.port,
    )
    return None


async def _notify_waiting(notify: Notify | None, label: str, remaining_s: float) -> None:
    if notify is None:
        return
    remaining = max(0, int(remaining_s))
    message = (
        f"Model host {label} is not responding. "
        f"Sent Wake-on-LAN and waiting up to {remaining}s for it to come back."
    )
    try:
        await notify(message)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Wake-on-LAN progress notification failed")


def _host_label(provider_name: str | None, host: str, port: int) -> str:
    target = f"{host}:{port}"
    if provider_name:
        return f"{provider_name} ({target})"
    return target


def _timeout_message(label: str, config: WakeOnLanConfig, send_error: str | None) -> str:
    message = (
        f"Model host {label} did not accept connections within "
        f"{config.timeout_s:g}s after Wake-on-LAN to {config.broadcast}:{config.port}"
    )
    if send_error:
        return f"{message}: {send_error}"
    return message
