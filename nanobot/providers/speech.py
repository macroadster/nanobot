"""Grok Voice and Kokoro text-to-speech.

Product policy (when to speak, which text, where the file goes) lives in
``nanobot.audio.speech``. This module only calls the voice endpoints.

Two backends are implemented:

- ``GrokTextToSpeech``: xAI Grok Voice (``POST /v1/tts``).
- ``KokoroTextToSpeech``: an OpenAI-compatible ``POST /v1/audio/speech``
  endpoint, such as a local speaches server serving Kokoro
  (``speaches-ai/Kokoro-82M-v1.0-ONNX``). No API key is required unless the
  server enforces one.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlsplit

import httpx
from loguru import logger

_DEFAULT_API_BASE = "https://api.x.ai/v1"
_CHAT_PROXY_HOST = "cli-chat-proxy.grok.com"
_KOKORO_DEFAULT_API_BASE = "http://localhost:8000/v1"
_KOKORO_DEFAULT_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
_KOKORO_DEFAULT_VOICE = "af_heart"
_MAX_RETRIES = 3
_BACKOFF_S = (1.0, 2.0, 4.0)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def grok_voice_api_base(api_base: str | None) -> str:
    """Return the xAI voice API base for a Grok chat-style base URL.

    Voice endpoints are ``/stt`` and ``/tts`` on ``api.x.ai``. The CLI chat
    proxy used by some Grok configs does not serve them, so that host is
    rewritten back to the public voice API. An explicit proxy base is kept.
    """
    base = (api_base or _DEFAULT_API_BASE).strip().rstrip("/")
    if not base:
        return _DEFAULT_API_BASE
    host = (urlsplit(base).hostname or "").lower()
    if host == _CHAT_PROXY_HOST:
        return _DEFAULT_API_BASE
    for suffix in ("/audio/transcriptions", "/stt", "/tts"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or _DEFAULT_API_BASE


def grok_stt_url(api_base: str | None) -> str:
    return f"{grok_voice_api_base(api_base)}/stt"


def grok_tts_url(api_base: str | None) -> str:
    return f"{grok_voice_api_base(api_base)}/tts"


def kokoro_speech_url(api_base: str | None) -> str:
    """Return the OpenAI-compatible speech URL for a Kokoro/speaches base URL."""
    base = (api_base or os.environ.get("KOKORO_API_BASE") or _KOKORO_DEFAULT_API_BASE).strip().rstrip("/")
    if not base:
        base = _KOKORO_DEFAULT_API_BASE
    if base.endswith("/audio/speech"):
        return base
    return f"{base}/audio/speech"


def audio_suffix(content: bytes, content_type: str) -> str:
    """Pick a file suffix from magic bytes, then the response content type."""
    if content.startswith(b"RIFF"):
        return ".wav"
    if content.startswith(b"OggS"):
        return ".ogg"
    lowered = content_type.lower()
    if "wav" in lowered:
        return ".wav"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    return ".mp3"


async def _post_speech_with_retry(
    *,
    provider_label: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    default_suffix: str = ".mp3",
) -> tuple[bytes, str]:
    """POST a speech request, retrying transient failures. Returns ``(audio, suffix)``."""
    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=60.0,
                )
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "{} voice transient error (attempt {}/{}): {}",
                        provider_label,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                    )
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.exception("{} voice synthesis failed after retries", provider_label)
                return b"", default_suffix
            except Exception:
                logger.exception("{} voice synthesis failed", provider_label)
                return b"", default_suffix

            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                logger.warning(
                    "{} voice transient HTTP {} (attempt {}/{})",
                    provider_label,
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue

            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400 or "json" in content_type.lower():
                detail = response.text.strip().replace("\n", " ")[:500]
                logger.error(
                    "{} voice HTTP {}{}",
                    provider_label,
                    response.status_code,
                    f": {detail}" if detail else "",
                )
                return b"", default_suffix
            content = response.content
            if not content:
                logger.error("{} voice synthesis returned an empty body", provider_label)
                return b"", default_suffix
            return content, audio_suffix(content, content_type)
    return b"", default_suffix


class GrokTextToSpeech:
    """Synthesize speech with ``POST /v1/tts``. Returns audio bytes or ``b""``."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str | None = None,
        voice_id: str = "eve",
        language: str = "auto",
    ) -> None:
        self.api_key = api_key
        self.api_url = grok_tts_url(api_base)
        self.voice_id = voice_id or "eve"
        self.language = language or "auto"

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if not self.api_key or not text:
            return b"", ".mp3"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "text_normalization": True,
        }
        return await _post_speech_with_retry(
            provider_label="Grok",
            url=self.api_url,
            headers=headers,
            body=body,
        )


class KokoroTextToSpeech:
    """Synthesize speech with an OpenAI-compatible ``/v1/audio/speech`` endpoint.

    Designed for a local speaches server serving Kokoro
    (``speaches-ai/Kokoro-82M-v1.0-ONNX``), but works with any compatible
    server. Returns audio bytes or ``b""``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        response_format: str = "mp3",
    ) -> None:
        self.api_key = api_key or os.environ.get("KOKORO_API_KEY") or ""
        self.api_url = kokoro_speech_url(api_base)
        self.model = (model or os.environ.get("KOKORO_MODEL") or _KOKORO_DEFAULT_MODEL).strip()
        self.voice_id = (voice_id or _KOKORO_DEFAULT_VOICE).strip() or _KOKORO_DEFAULT_VOICE
        self.response_format = (response_format or "mp3").strip().lower()

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if not text:
            return b"", ".mp3"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice_id,
            "response_format": self.response_format,
        }
        return await _post_speech_with_retry(
            provider_label="Kokoro",
            url=self.api_url,
            headers=headers,
            body=body,
        )
