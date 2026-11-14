"""Spoken replies for Discord voice messages and WebUI voice turns.

Inbound speech still uses the shared transcription service; this module
decides when a finished answer should be spoken and writes the audio file.

The TTS backend is selected with ``voice.provider``: ``grok`` (xAI Grok Voice,
the default) or ``kokoro`` (an OpenAI-compatible ``/v1/audio/speech`` endpoint,
such as a local speaches server, configured under ``providers.kokoro``).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.audio.transcription import (
    transcription_provider_api_base,
    transcription_provider_api_key,
)
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Config
from nanobot.providers.speech import GrokTextToSpeech, KokoroTextToSpeech

VOICE_REPLY_META = "grok_voice_reply"
_KOKORO_DEFAULT_API_BASE = "http://localhost:8000/v1"
_KOKORO_DEFAULT_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
_KOKORO_DEFAULT_VOICE = "af_heart"
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2})?$|^auto$")


@dataclass(frozen=True)
class EffectiveVoiceConfig:
    enabled: bool
    speak_replies: bool
    voice_id: str
    language: str
    api_key: str
    api_base: str
    max_chars: int
    provider: str = "grok"
    model: str | None = None

    @property
    def configured(self) -> bool:
        if self.provider == "kokoro":
            # Local Kokoro servers typically do not require a key.
            return bool(self.api_base)
        return bool(self.api_key)


def voice_reply_requested(metadata: Mapping[str, Any] | None) -> bool:
    """True when this turn started as speech and should be answered in speech."""
    return bool(metadata and metadata.get(VOICE_REPLY_META) is True)


def prepare_speech_text(text: str, max_chars: int) -> str:
    """Turn an assistant reply into text that is reasonable to read aloud."""
    cleaned = _CODE_FENCE.sub(" ", text or "")
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        clipped = cleaned[:max_chars]
        pivot = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        if pivot > max_chars // 2:
            clipped = clipped[: pivot + 1]
        cleaned = clipped.strip()
    return cleaned


def _tts_language(voice_language: str | None, transcription_language: str | None) -> str:
    for candidate in (voice_language, transcription_language):
        if isinstance(candidate, str) and _LANGUAGE.fullmatch(candidate.strip()):
            value = candidate.strip()
            return "auto" if value.lower() == "auto" else value
    return "auto"


def resolve_voice_config(config: Config) -> EffectiveVoiceConfig:
    """Resolve spoken-reply settings and credentials for the configured provider."""
    voice = config.voice
    provider = (getattr(voice, "provider", None) or "grok").strip().lower()
    transcription_language = getattr(config.transcription, "language", None)
    if provider == "kokoro":
        api_base = (
            transcription_provider_api_base(config, "kokoro")
            or _KOKORO_DEFAULT_API_BASE
        )
        api_key = transcription_provider_api_key(config, "kokoro")
        voice_id = (voice.voice_id or "").strip()
        # Grok's default voice id is not a Kokoro voice; swap in a Kokoro default.
        if not voice_id or voice_id == "eve":
            voice_id = _KOKORO_DEFAULT_VOICE
        return EffectiveVoiceConfig(
            enabled=bool(voice.enabled),
            speak_replies=bool(voice.speak_replies),
            voice_id=voice_id,
            language=_tts_language(voice.language, transcription_language),
            api_key=api_key,
            api_base=api_base,
            max_chars=int(voice.max_chars),
            provider="kokoro",
            model=(getattr(voice, "model", None) or None) or _KOKORO_DEFAULT_MODEL,
        )
    return EffectiveVoiceConfig(
        enabled=bool(voice.enabled),
        speak_replies=bool(voice.speak_replies),
        voice_id=(voice.voice_id or "eve").strip() or "eve",
        language=_tts_language(voice.language, transcription_language),
        api_key=transcription_provider_api_key(config, "grok"),
        api_base=transcription_provider_api_base(config, "grok"),
        max_chars=int(voice.max_chars),
        provider="grok",
        model=None,
    )


async def synthesize_voice_reply(
    text: str,
    metadata: Mapping[str, Any] | None,
) -> Path | None:
    """Write a spoken reply, or return None when this turn should stay text."""
    if not voice_reply_requested(metadata):
        return None
    try:
        from nanobot.config.loader import load_config

        resolved = resolve_voice_config(load_config())
    except Exception:
        logger.exception("voice config could not be loaded")
        return None
    if not resolved.enabled or not resolved.speak_replies or not resolved.configured:
        return None
    spoken = prepare_speech_text(text, resolved.max_chars)
    if not spoken:
        return None
    if resolved.provider == "kokoro":
        audio, suffix = await KokoroTextToSpeech(
            api_key=resolved.api_key or None,
            api_base=resolved.api_base or None,
            model=resolved.model,
            voice_id=resolved.voice_id,
        ).synthesize(spoken)
    else:
        audio, suffix = await GrokTextToSpeech(
            api_key=resolved.api_key,
            api_base=resolved.api_base or None,
            voice_id=resolved.voice_id,
            language=resolved.language,
        ).synthesize(spoken)
    if not audio:
        return None
    path = get_media_dir("voice") / f"voice-{uuid.uuid4().hex}{suffix}"
    try:
        path.write_bytes(audio)
    except OSError:
        logger.exception("Failed to store {} voice reply", resolved.provider)
        return None
    logger.info(
        "Synthesized {} voice reply ({} bytes, voice={})",
        resolved.provider,
        len(audio),
        resolved.voice_id,
    )
    return path
