"""Tests for the Grok / xAI provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import Config, ProvidersConfig
from nanobot.providers.factory import make_provider
from nanobot.providers.grok_provider import (
    GROK_OIDC_PROXY_BASE,
    GROK_PUBLIC_BASE,
    GrokProvider,
    _normalize_api_base,
)
from nanobot.providers.registry import PROVIDERS, find_by_name


def test_grok_config_field_exists() -> None:
    assert hasattr(ProvidersConfig(), "grok")


def test_grok_provider_in_registry() -> None:
    spec = find_by_name("grok")
    assert spec is not None
    assert spec.name == "grok"
    assert spec.backend == "grok"
    assert spec.env_key == "XAI_API_KEY"
    assert spec.default_api_base == GROK_PUBLIC_BASE
    assert spec.is_oauth is True
    assert "grok" in {s.name for s in PROVIDERS}


def test_normalize_api_base_defaults_and_strips() -> None:
    assert _normalize_api_base(None) == GROK_PUBLIC_BASE
    assert _normalize_api_base("") == GROK_PUBLIC_BASE
    assert _normalize_api_base("   ") == GROK_PUBLIC_BASE
    assert _normalize_api_base(f"{GROK_OIDC_PROXY_BASE}/") == GROK_OIDC_PROXY_BASE
    assert _normalize_api_base(f"  {GROK_OIDC_PROXY_BASE}/  ") == GROK_OIDC_PROXY_BASE


def test_config_default_api_base_for_grok() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "grok", "model": "grok-4"}},
            "providers": {"grok": {}},
        }
    )

    assert config.get_provider_name("grok-4") == "grok"
    assert config.get_api_base("grok-4") == GROK_PUBLIC_BASE


def test_config_honors_grok_api_base_override() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "grok", "model": "grok-4"}},
            "providers": {
                "grok": {
                    "apiBase": f"{GROK_OIDC_PROXY_BASE}/",
                }
            },
        }
    )

    assert config.get_api_base("grok-4") == f"{GROK_OIDC_PROXY_BASE}/"


def test_grok_provider_defaults_to_public_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = GrokProvider(default_model="grok-4", api_key="xai-test-key")

    assert provider._configured_api_base == GROK_PUBLIC_BASE
    assert provider.api_base == GROK_PUBLIC_BASE
    assert provider._effective_base == GROK_PUBLIC_BASE


def test_grok_provider_uses_configured_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = GrokProvider(
            default_model="grok-4",
            api_key="xai-test-key",
            api_base=f"{GROK_OIDC_PROXY_BASE}/",
            extra_headers={"X-Test": "1"},
        )

    assert provider._configured_api_base == GROK_OIDC_PROXY_BASE
    assert provider.api_base == GROK_OIDC_PROXY_BASE
    assert provider._effective_base == GROK_OIDC_PROXY_BASE
    assert provider.extra_headers["X-Test"] == "1"
    assert provider._default_headers["X-Test"] == "1"
    assert "nanobot" in provider._default_headers["User-Agent"]


def test_make_provider_passes_grok_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "grok", "model": "grok-4"}},
            "providers": {
                "grok": {
                    "apiKey": "xai-from-config",
                    "apiBase": GROK_OIDC_PROXY_BASE,
                    "extraHeaders": {"X-Custom": "yes"},
                }
            },
        }
    )

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        provider = make_provider(config)
        asyncio.run(provider._ensure_client())

    assert isinstance(provider, GrokProvider)
    assert provider._configured_api_base == GROK_OIDC_PROXY_BASE
    kwargs = mock_client.call_args.kwargs
    assert kwargs["base_url"] == GROK_OIDC_PROXY_BASE
    assert kwargs["api_key"] == "xai-from-config"
    assert kwargs["default_headers"]["X-Custom"] == "yes"


@pytest.mark.asyncio
async def test_grok_token_refresh_keeps_configured_proxy_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "https://auth.x.ai::test": {
                    "key": "oidc-access-token",
                    "auth_mode": "oidc",
                    "expires_at": 9_999_999_999,
                    "email": "user@example.com",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "nanobot.providers.grok_provider.GROK_AUTH_FILE", auth_file
    )

    mock_client = MagicMock()
    mock_client.api_key = "no-key"
    mock_client.chat.completions.create = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    with patch(
        "nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=mock_client
    ):
        provider = GrokProvider(
            default_model="grok-4",
            api_base=GROK_OIDC_PROXY_BASE,
        )
        response = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
            max_tokens=16,
            temperature=0.1,
        )

    assert response.content == "ok"
    assert provider._configured_api_base == GROK_OIDC_PROXY_BASE
    assert provider.api_base == GROK_OIDC_PROXY_BASE
    assert provider._effective_base == GROK_OIDC_PROXY_BASE
    assert provider._client.api_key == "oidc-access-token"
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_grok_explicit_key_uses_configured_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "live-xai-key")

    mock_client = MagicMock()
    mock_client.api_key = "no-key"
    mock_client.chat.completions.create = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "keyed"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    with patch(
        "nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=mock_client
    ):
        provider = GrokProvider(
            default_model="grok-4",
            api_base=GROK_OIDC_PROXY_BASE,
        )
        response = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
            max_tokens=16,
            temperature=0.1,
        )

    assert response.content == "keyed"
    assert provider._using_oidc is False
    assert provider.api_base == GROK_OIDC_PROXY_BASE
    assert provider._client.api_key == "live-xai-key"
