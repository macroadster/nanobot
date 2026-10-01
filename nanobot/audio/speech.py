"""Spoken replies for Discord voice messages and WebUI voice turns.

Grok Voice text-to-speech is the only backend. Inbound speech still uses the
shared transcription service; this module decides when a finished answer
should be spoken and writes the audio file.
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
from nanobot.providers.speech import GrokTextToSpeech

VOICE_REPLY_META = "grok_voice_reply"
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

    @property
    def configured(self) -> bool:
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
    """Resolve Grok Voice reply settings and credentials."""
    voice = config.voice
    transcription_language = getattr(config.transcription, "language", None)
    return EffectiveVoiceConfig(
        enabled=bool(voice.enabled),
        speak_replies=bool(voice.speak_replies),
        voice_id=(voice.voice_id or "eve").strip() or "eve",
        language=_tts_language(voice.language, transcription_language),
        api_key=transcription_provider_api_key(config, "grok"),
        api_base=transcription_provider_api_base(config, "grok"),
        max_chars=int(voice.max_chars),
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
        logger.exception("Grok voice config could not be loaded")
        return None
    if not resolved.enabled or not resolved.speak_replies or not resolved.configured:
        return None
    spoken = prepare_speech_text(text, resolved.max_chars)
    if not spoken:
        return None
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
        logger.exception("Failed to store Grok voice reply")
        return None
    logger.info("Synthesized Grok voice reply ({} bytes, voice={})", len(audio), resolved.voice_id)
    return path
