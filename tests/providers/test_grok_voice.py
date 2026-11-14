"""Grok Voice speech-to-text and spoken replies."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nanobot.audio.speech import (
    VOICE_REPLY_META,
    prepare_speech_text,
    resolve_voice_config,
    synthesize_voice_reply,
)
from nanobot.audio.transcription import transcription_provider_api_key
from nanobot.audio.transcription_registry import (
    get_transcription_provider,
    resolve_transcription_provider,
)
from nanobot.config.schema import Config
from nanobot.providers.speech import GrokTextToSpeech, grok_stt_url, grok_tts_url
from nanobot.providers.transcription import GrokTranscriptionProvider


def test_grok_voice_urls_use_the_public_api() -> None:
    assert grok_stt_url("https://api.x.ai/v1") == "https://api.x.ai/v1/stt"
    assert grok_tts_url("https://api.x.ai/v1") == "https://api.x.ai/v1/tts"
    assert grok_stt_url("https://cli-chat-proxy.grok.com/v1") == "https://api.x.ai/v1/stt"
    assert grok_stt_url("https://proxy.example/v1") == "https://proxy.example/v1/stt"


def test_grok_transcription_provider_is_registered() -> None:
    spec = get_transcription_provider("grok")
    assert spec is not None
    assert spec.default_model == "grok-voice-transcribe-2.0"
    assert spec.adapter == "nanobot.providers.transcription:GrokTranscriptionProvider"
    assert resolve_transcription_provider("grok") is spec


def test_prepare_speech_text_skips_code_and_truncates() -> None:
    spoken = prepare_speech_text("Hello.\n```python\nprint(1)\n```\nSee [docs](https://example.com).", 40)
    assert "print" not in spoken
    assert "https://" not in spoken
    assert spoken.startswith("Hello.")
    assert "docs" in spoken
    assert prepare_speech_text("abcdefghij. klmnopqrstuvwxyz", 12) == "abcdefghij."


def test_grok_transcription_key_prefers_config_then_oidc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "nanobot.providers.grok_provider.load_grok_oidc_token",
        lambda: {"key": " oidc-token "},
    )
    config = Config()
    assert transcription_provider_api_key(config, "grok") == "oidc-token"

    config.providers.grok.api_key = "explicit-key"
    assert transcription_provider_api_key(config, "grok") == "explicit-key"


def test_voice_config_uses_transcription_language_and_grok_key() -> None:
    config = Config()
    config.providers.grok.api_key = "xai-test"
    config.transcription.language = "ja"
    config.voice.voice_id = "ara"

    resolved = resolve_voice_config(config)

    assert resolved.configured is True
    assert resolved.voice_id == "ara"
    assert resolved.language == "ja"
    assert resolved.speak_replies is True


@pytest.mark.asyncio
async def test_grok_transcription_posts_multipart_with_file_last(tmp_path: Path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"OggSfake")
    provider = GrokTranscriptionProvider(
        api_key="xai-test",
        api_base="https://api.x.ai/v1",
        language="en",
        model="grok-voice-transcribe-2.0",
    )
    captured: dict[str, object] = {}

    async def post(self, url: str, **kwargs: object) -> httpx.Response:  # noqa: ARG001
        captured["url"] = url
        captured["files"] = kwargs["files"]
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"text": "hello from grok"}, request=request)

    with patch("httpx.AsyncClient.post", post):
        assert await provider.transcribe(audio) == "hello from grok"

    assert captured["url"] == "https://api.x.ai/v1/stt"
    files = captured["files"]
    assert isinstance(files, list)
    names = [name for name, _payload in files]
    assert names == ["model", "language", "format", "file"]


@pytest.mark.asyncio
async def test_grok_tts_returns_audio_bytes() -> None:
    synth = GrokTextToSpeech(
        api_key="xai-test",
        api_base="https://api.x.ai/v1",
        voice_id="eve",
        language="auto",
    )
    request = httpx.Request("POST", synth.api_url)
    response = httpx.Response(
        200,
        content=b"ID3fake-mp3",
        headers={"content-type": "audio/mpeg"},
        request=request,
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)) as post:
        audio, suffix = await synth.synthesize("Hello there")

    assert audio == b"ID3fake-mp3"
    assert suffix == ".mp3"
    sent = post.await_args.kwargs["json"]
    assert sent["text"] == "Hello there"
    assert sent["voice_id"] == "eve"
    assert sent["language"] == "auto"
    assert sent["text_normalization"] is True


@pytest.mark.asyncio
async def test_synthesize_voice_reply_writes_a_file_only_for_voice_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config()
    config.providers.grok.api_key = "xai-test"
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr("nanobot.audio.speech.get_media_dir", lambda _name: tmp_path)

    async def fake_synthesize(self, text: str) -> tuple[bytes, str]:  # noqa: ARG001
        assert "Hello" in text
        assert "print" not in text
        return b"ID3reply", ".mp3"

    monkeypatch.setattr(GrokTextToSpeech, "synthesize", fake_synthesize)

    skipped = await synthesize_voice_reply("Hello", {})
    assert skipped is None

    path = await synthesize_voice_reply(
        "Hello.\n```python\nprint(1)\n```",
        {VOICE_REPLY_META: True},
    )
    assert path is not None
    assert path.parent == tmp_path
    assert path.read_bytes() == b"ID3reply"
    assert path.suffix == ".mp3"


# --- Kokoro (local OpenAI-compatible /v1/audio/speech) ---------------------

from nanobot.providers.speech import KokoroTextToSpeech, kokoro_speech_url  # noqa: E402


def test_kokoro_speech_url_accepts_base_or_full_url() -> None:
    assert kokoro_speech_url("http://audio:8000/v1") == "http://audio:8000/v1/audio/speech"
    assert kokoro_speech_url("http://audio:8000/v1/audio/speech") == "http://audio:8000/v1/audio/speech"
    assert kokoro_speech_url(None) == "http://localhost:8000/v1/audio/speech"


def test_voice_config_resolves_kokoro_without_api_key() -> None:
    config = Config()
    config.voice.provider = "kokoro"
    config.voice.voice_id = "am_michael"
    config.providers.kokoro.api_base = "http://nanobot-audio.nanobot.svc.cluster.local:8000/v1"

    resolved = resolve_voice_config(config)

    assert resolved.provider == "kokoro"
    assert resolved.configured is True
    assert resolved.voice_id == "am_michael"
    assert resolved.api_base.endswith("/v1")
    assert resolved.model == "speaches-ai/Kokoro-82M-v1.0-ONNX"


def test_voice_config_kokoro_swaps_grok_default_voice() -> None:
    config = Config()
    config.voice.provider = "kokoro"  # voice_id left at the grok default "eve"
    resolved = resolve_voice_config(config)
    assert resolved.voice_id == "af_heart"


def test_voice_config_rejects_unknown_provider() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Config(voice={"provider": "elevenlabs"})


@pytest.mark.asyncio
async def test_kokoro_tts_posts_openai_compatible_body() -> None:
    synth = KokoroTextToSpeech(
        api_base="http://nanobot-audio:8000/v1",
        model="speaches-ai/Kokoro-82M-v1.0-ONNX",
        voice_id="af_heart",
    )
    assert synth.api_url == "http://nanobot-audio:8000/v1/audio/speech"
    request = httpx.Request("POST", synth.api_url)
    response = httpx.Response(
        200,
        content=b"ID3kokoro",
        headers={"content-type": "audio/mpeg"},
        request=request,
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)) as post:
        audio, suffix = await synth.synthesize("Hello there")

    assert audio == b"ID3kokoro"
    assert suffix == ".mp3"
    sent = post.await_args.kwargs["json"]
    assert sent == {
        "model": "speaches-ai/Kokoro-82M-v1.0-ONNX",
        "input": "Hello there",
        "voice": "af_heart",
        "response_format": "mp3",
    }


@pytest.mark.asyncio
async def test_synthesize_voice_reply_uses_kokoro_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config()
    config.voice.provider = "kokoro"
    config.voice.voice_id = "af_heart"
    config.providers.kokoro.api_base = "http://nanobot-audio:8000/v1"
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr("nanobot.audio.speech.get_media_dir", lambda _name: tmp_path)

    async def fake_synthesize(self, text: str) -> tuple[bytes, str]:  # noqa: ARG001
        return b"kokoro-audio", ".mp3"

    monkeypatch.setattr(KokoroTextToSpeech, "synthesize", fake_synthesize)

    path = await synthesize_voice_reply("Hello", {VOICE_REPLY_META: True})
    assert path is not None
    assert path.read_bytes() == b"kokoro-audio"
