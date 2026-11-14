"""Ollama thinking follows the active preset, not Grok's reasoning effort."""

from __future__ import annotations

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def _kwargs(spec_name: str, reasoning_effort: str | None) -> dict:
    provider = OpenAICompatProvider(
        api_key="test-key",
        default_model="qwen3.8-iq2s:32k",
        spec=find_by_name(spec_name),
    )
    return provider._build_kwargs(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        model="qwen3.8-iq2s:32k",
        max_tokens=128,
        temperature=0.7,
        reasoning_effort=reasoning_effort,
        tool_choice=None,
    )


def test_ollama_sends_reasoning_effort_none() -> None:
    kwargs = _kwargs("ollama", "none")
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["temperature"] == 0.7


def test_ollama_maps_grok_xhigh_to_high() -> None:
    kwargs = _kwargs("ollama", "xhigh")
    assert kwargs["reasoning_effort"] == "high"


def test_ollama_omits_reasoning_effort_when_unset() -> None:
    kwargs = _kwargs("ollama", None)
    assert "reasoning_effort" not in kwargs


def test_other_providers_still_omit_explicit_none() -> None:
    kwargs = _kwargs("grok", "none")
    assert "reasoning_effort" not in kwargs
