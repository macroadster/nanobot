"""Grok (xAI) provider that consumes ~/.grok/auth.json OIDC tokens.

This provider prefers explicit XAI_API_KEY (or providers.grok.api_key).
When no API key is available, it falls back to the browser/OIDC login
credentials written by `grok login` (the Grok Build TUI / CLI) and uses the
stored JWT as a Bearer token.

The default API base is ``https://api.x.ai/v1`` (OIDC tokens include
``api:access`` and work there). Override with ``providers.grok.apiBase`` —
for example the legacy CLI proxy ``https://cli-chat-proxy.grok.com/v1``.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name

GROK_PUBLIC_BASE = "https://api.x.ai/v1"
# Legacy CLI proxy used by `grok` TUI/CLI chat; set providers.grok.apiBase to use it.
GROK_OIDC_PROXY_BASE = "https://cli-chat-proxy.grok.com/v1"


def _normalize_api_base(api_base: str | None) -> str:
    """Return a non-empty API base, defaulting to the public xAI endpoint."""
    if isinstance(api_base, str):
        trimmed = api_base.strip().rstrip("/")
        if trimmed:
            return trimmed
    return GROK_PUBLIC_BASE


def _resolve_grok_auth_path() -> Path:
    """Resolve location of auth.json, honoring the same env vars as the official Grok CLI.

    Precedence:
    1. GROK_AUTH_FILE (full path override)
    2. GROK_HOME (directory containing auth.json)
    3. Default: ~/.grok/auth.json
    """
    explicit = os.getenv("GROK_AUTH_FILE")
    if explicit:
        return Path(explicit).expanduser()
    grok_home = os.getenv("GROK_HOME")
    if grok_home:
        return Path(grok_home).expanduser() / "auth.json"
    return Path.home() / ".grok" / "auth.json"


GROK_AUTH_FILE: Path = _resolve_grok_auth_path()

GROK_OIDC_TOKEN_URL = "https://auth.x.ai/oauth/token"

_EXPIRY_SKEW_SECONDS = 300
_MIN_REFRESH_INTERVAL = 30


def _parse_expires_at(value: Any) -> float | None:
    """Best-effort parse of expires_at from auth.json into epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 10_000_000_000:
            v /= 1000.0
        return v
    if isinstance(value, str):
        with suppress(Exception):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc).timestamp()
        with suppress(Exception):
            dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return dt.replace(tzinfo=timezone.utc).timestamp()
    return None


def _load_raw_auth_file() -> dict[str, Any]:
    if not GROK_AUTH_FILE.exists():
        return {}
    try:
        text = GROK_AUTH_FILE.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Failed to read ~/.grok/auth.json: {}", exc)
        return {}


def _pick_best_oidc_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the most suitable OIDC/browser login entry from auth.json."""
    if not raw:
        return None

    candidates: list[dict[str, Any]] = []
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        token = entry.get("key")
        if not (isinstance(token, str) and token):
            continue

        if (
            entry.get("auth_mode") == "oidc"
            or entry.get("refresh_token")
            or "auth.x.ai" in str(key).lower()
            or "accounts.x.ai" in str(key).lower()
        ):
            candidates.append(dict(entry))
        elif token.count(".") >= 2:
            candidates.append(dict(entry))

    if not candidates:
        return None

    # Prefer an entry that still has a refresh_token when available.
    for c in candidates:
        if c.get("refresh_token"):
            return c

    def sort_key(c: dict[str, Any]) -> float:
        exp = _parse_expires_at(c.get("expires_at"))
        if exp is None:
            return 1_000_000_000_000
        return exp

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def load_grok_oidc_token() -> dict[str, Any] | None:
    """Return the best OIDC token dict from ~/.grok/auth.json, or None."""
    raw = _load_raw_auth_file()
    return _pick_best_oidc_entry(raw)


def get_grok_login_status() -> dict[str, Any]:
    """Return status info for CLI `status` and WebUI settings.

    This is intentionally cheap and does not perform network calls.
    """
    token = load_grok_oidc_token()
    if not token:
        return {
            "configured": False,
            "account": None,
            "expires_at": None,
            "login_supported": True,
        }

    email = token.get("email") or token.get("user_id")
    exp = _parse_expires_at(token.get("expires_at"))
    return {
        "configured": True,
        "account": email,
        "expires_at": int(exp * 1000) if exp else None,
        "login_supported": True,
    }


async def _attempt_oidc_refresh(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Try to refresh using the refresh_token. Returns updated entry dict or None."""
    refresh_token = entry.get("refresh_token")
    if not refresh_token:
        return None

    # Resolve OIDC client id from the auth entry (written by `grok login`).
    client_id = None
    if isinstance(entry.get("oidc_client_id"), str):
        client_id = entry["oidc_client_id"]

    if not client_id:
        logger.debug("Grok OIDC refresh: no client_id found; skipping refresh")
        return None

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }

    timeout = httpx.Timeout(15.0, connect=10.0)
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, trust_env=True
        ) as client:
            resp = await client.post(
                GROK_OIDC_TOKEN_URL,
                data=data,
                headers={"Accept": "application/json"},
            )
            if resp.status_code >= 400:
                logger.debug(
                    "Grok OIDC refresh failed: {} {}",
                    resp.status_code,
                    resp.text[:200],
                )
                return None
            payload = resp.json()
    except Exception as exc:
        logger.debug("Grok OIDC refresh network error: {}", exc)
        return None

    new_access = payload.get("access_token") or payload.get("access")
    if not new_access:
        return None

    updated = dict(entry)
    updated["key"] = new_access
    if "refresh_token" in payload:
        updated["refresh_token"] = payload["refresh_token"]
    if "expires_in" in payload:
        updated["expires_at"] = time.time() + int(payload["expires_in"])
    elif "expires_at" in payload:
        updated["expires_at"] = payload["expires_at"]

    _persist_refreshed_token(updated)
    return updated


def _persist_refreshed_token(updated_entry: dict[str, Any]) -> None:
    """Best-effort atomic update of the matching entry in ~/.grok/auth.json."""
    raw = _load_raw_auth_file()
    if not raw:
        return

    target_key = None
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        if updated_entry.get("key") and v.get("key") == updated_entry.get("key"):
            target_key = k
            break
        if updated_entry.get("user_id") and v.get("user_id") == updated_entry.get("user_id"):
            target_key = k
            break
        if updated_entry.get("email") and v.get("email") == updated_entry.get("email"):
            target_key = k
            break
        if v.get("auth_mode") == "oidc" or v.get("refresh_token"):
            target_key = k
            break

    if not target_key:
        return

    raw[target_key] = updated_entry
    try:
        tmp = GROK_AUTH_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        tmp.replace(GROK_AUTH_FILE)
        logger.debug("Persisted refreshed Grok OIDC token")
    except Exception as exc:
        logger.debug("Failed to persist refreshed Grok token: {}", exc)


class GrokProvider(OpenAICompatProvider):
    """Provider for Grok models.

    - If an explicit API key is available (config or XAI_API_KEY env), uses
      key auth against the configured API base (default ``api.x.ai``).
    - Otherwise reads the OIDC JWT from ~/.grok/auth.json and uses it as a
      Bearer token against the same configured API base.
    """

    def __init__(
        self,
        default_model: str = "grok-4",
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self._grok_token: str | None = None
        self._grok_token_expires: float = 0.0
        self._last_refresh_attempt: float = 0.0
        self._using_oidc: bool = False
        self._configured_api_base = _normalize_api_base(api_base)

        self._explicit_api_key = self._resolve_explicit_api_key(provided=api_key)

        if self._explicit_api_key:
            key_for_client = self._explicit_api_key
            self._using_oidc = False
        else:
            # Placeholder until the first request loads/refreshes the OIDC JWT.
            key_for_client = None
            self._using_oidc = True

        headers = {"User-Agent": "nanobot (grok-provider)"}
        if extra_headers:
            headers.update(extra_headers)

        super().__init__(
            api_key=key_for_client,
            api_base=self._configured_api_base,
            default_model=default_model,
            extra_headers=headers,
            spec=find_by_name("grok"),
        )

    def _resolve_explicit_api_key(self, provided: str | None = None) -> str | None:
        """Return an explicit XAI API key if available.

        Precedence: live XAI_API_KEY env var > providers.grok.api_key from config.
        Never returns OIDC JWTs from ``~/.grok/auth.json``.
        """
        env = os.getenv("XAI_API_KEY")
        if env and str(env).strip() and str(env).strip() not in ("no-key", ""):
            return str(env).strip()

        if provided and provided not in ("no-key", "", None):
            return provided

        explicit = getattr(self, "_explicit_api_key", None)
        if explicit and explicit not in ("no-key", "", None):
            return explicit

        return None

    def _needs_client_update(self, key: str) -> bool:
        """True when the live client base URL or API key diverges from config."""
        return (
            _normalize_api_base(self.api_base) != self._configured_api_base
            or _normalize_api_base(getattr(self, "_effective_base", None))
            != self._configured_api_base
            or self._api_key_for_client != key
        )

    async def _get_fresh_grok_token(self) -> str:
        """Return a valid JWT (either from key or from refreshed OIDC file)."""
        now = time.time()

        # Prefer a real API key when one is configured.
        explicit = self._resolve_explicit_api_key()
        if explicit:
            if self._using_oidc or self._needs_client_update(explicit):
                self._using_oidc = False
                await self._recreate_client_with_new_base(self._configured_api_base, explicit)
            self._grok_token = explicit
            self._grok_token_expires = now + 31_536_000
            return explicit

        # Cached OIDC token still valid?
        if (
            self._grok_token
            and now < self._grok_token_expires - _EXPIRY_SKEW_SECONDS
            and not self._needs_client_update(self._grok_token)
        ):
            return self._grok_token

        entry = load_grok_oidc_token()
        if not entry:
            raise RuntimeError(
                "Grok is not logged in. Run `grok login` (or `grok login --oauth`) to "
                "authenticate via the browser and store credentials in ~/.grok/auth.json, "
                "then retry. Alternatively set XAI_API_KEY for direct API access."
            )

        exp = _parse_expires_at(entry.get("expires_at"))

        needs_refresh = bool(entry.get("refresh_token")) and (
            exp is None or now >= exp - _EXPIRY_SKEW_SECONDS
        ) and (now - self._last_refresh_attempt > _MIN_REFRESH_INTERVAL)

        if needs_refresh:
            self._last_refresh_attempt = now
            refreshed = await _attempt_oidc_refresh(entry)
            if refreshed:
                entry = refreshed
                exp = _parse_expires_at(entry.get("expires_at"))

        token = entry.get("key")
        if not (isinstance(token, str) and token):
            raise RuntimeError("~/.grok/auth.json contained no usable access token.")

        if exp is None:
            exp = now + 3600

        self._grok_token = token
        self._grok_token_expires = exp
        self._using_oidc = True

        if self._needs_client_update(token):
            await self._recreate_client_with_new_base(self._configured_api_base, token)

        self.api_key = token
        return token

    async def _recreate_client_with_new_base(self, new_base: str, new_key: str) -> None:
        """Recreate the underlying OpenAI client when we switch auth mode or base URL."""
        normalized = _normalize_api_base(new_base)
        self._configured_api_base = normalized
        self.api_base = normalized
        self._effective_base = normalized
        self._api_key_for_client = new_key
        self._client = None
        try:
            client = await self._ensure_client()
            client.api_key = new_key
        except Exception:
            return

    async def _refresh_client_api_key(self) -> str:
        token = await self._get_fresh_grok_token()
        client = await self._ensure_client()
        self.api_key = token
        client.api_key = token
        return token

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ):
        await self._refresh_client_api_key()
        return await super().chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
        on_content_delta: Any = None,
        on_thinking_delta: Any = None,
        on_tool_call_delta: Any = None,
    ):
        await self._refresh_client_api_key()
        return await super().chat_stream(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_delta=on_tool_call_delta,
        )

    def get_default_model(self) -> str:
        return self.default_model
