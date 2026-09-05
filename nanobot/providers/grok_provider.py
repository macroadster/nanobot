"""Grok (xAI) provider that consumes ~/.grok/auth.json OIDC tokens.

This provider prefers explicit XAI_API_KEY (or providers.grok.api_key).
When no API key is available, it falls back to the browser/OIDC login
credentials written by `grok login` (the Grok Build TUI / CLI) and uses the
stored JWT as a Bearer token.

The default API base is ``https://api.x.ai/v1`` (OIDC tokens include
``api:access`` and work there). Override with ``providers.grok.apiBase`` —
for example the legacy CLI proxy ``https://cli-chat-proxy.grok.com/v1``.

Requests to the CLI chat proxy must advertise a client version via the
``x-grok-client-version`` header. Without it the proxy rejects the call with
HTTP 426 (``Grok CLI version (none) is outdated``). This provider injects
that header automatically when ``apiBase`` points at the CLI proxy.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import suppress
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name

GROK_PUBLIC_BASE = "https://api.x.ai/v1"
# Legacy CLI proxy used by `grok` TUI/CLI chat; set providers.grok.apiBase to use it.
GROK_OIDC_PROXY_BASE = "https://cli-chat-proxy.grok.com/v1"

# Minimum version accepted by cli-chat-proxy.grok.com as of 2026-07.
GROK_CLI_MIN_VERSION = "0.1.202"
_GROK_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:[.-][0-9A-Za-z]+)*)")

_HEADER_CLIENT_VERSION = "x-grok-client-version"
_HEADER_TOKEN_AUTH = "X-XAI-Token-Auth"
_HEADER_MODEL_OVERRIDE = "x-grok-model-override"
_TOKEN_AUTH_CLI = "xai-grok-cli"


def _normalize_api_base(api_base: str | None) -> str:
    """Return a non-empty API base, defaulting to the public xAI endpoint."""
    if isinstance(api_base, str):
        trimmed = api_base.strip().rstrip("/")
        if trimmed:
            return trimmed
    return GROK_PUBLIC_BASE


def _resolve_grok_home() -> Path:
    """Resolve GROK_HOME / default ~/.grok, mirroring the official CLI."""
    explicit = os.getenv("GROK_HOME")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".grok"


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
    return _resolve_grok_home() / "auth.json"


def is_cli_chat_proxy(api_base: str | None) -> bool:
    """True when *api_base* targets the legacy Grok CLI chat proxy."""
    normalized = _normalize_api_base(api_base).lower()
    if normalized == GROK_OIDC_PROXY_BASE.lower():
        return True
    host = urlparse(normalized).hostname or ""
    return host == "cli-chat-proxy.grok.com" or host.endswith(".cli-chat-proxy.grok.com")


def _parse_version_string(raw: str | None) -> str | None:
    if not raw:
        return None
    match = _GROK_VERSION_RE.search(str(raw))
    return match.group(1) if match else None


def _read_version_json(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("version", "stable_version", "client_version"):
        parsed = _parse_version_string(data.get(key) if isinstance(data.get(key), str) else None)
        if parsed:
            return parsed
    return None


def _read_grok_binary_version() -> str | None:
    binary = shutil.which("grok")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        logger.debug("Failed to run `grok --version`: {}", exc)
        return None
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return _parse_version_string(output)


@lru_cache(maxsize=1)
def resolve_grok_client_version() -> str:
    """Resolve a client version string acceptable to the CLI chat proxy.

    Precedence:
    1. ``GROK_CLIENT_VERSION`` env var
    2. ``~/.grok/version.json`` (written by the official installer/updater)
    3. ``grok --version`` on ``PATH``
    4. ``GROK_CLI_MIN_VERSION`` fallback
    """
    env = _parse_version_string(os.getenv("GROK_CLIENT_VERSION"))
    if env:
        return env

    version_json = _read_version_json(_resolve_grok_home() / "version.json")
    if version_json:
        return version_json

    binary_version = _read_grok_binary_version()
    if binary_version:
        return binary_version

    return GROK_CLI_MIN_VERSION


def build_cli_proxy_headers(
    *,
    model: str | None = None,
    client_version: str | None = None,
    include_token_auth: bool = True,
) -> dict[str, str]:
    """Headers required by ``cli-chat-proxy.grok.com`` for non-CLI clients."""
    version = client_version or resolve_grok_client_version()
    headers = {
        _HEADER_CLIENT_VERSION: version,
        "User-Agent": f"nanobot (grok-provider; grok-cli/{version})",
    }
    if include_token_auth:
        headers[_HEADER_TOKEN_AUTH] = _TOKEN_AUTH_CLI
    if model:
        headers[_HEADER_MODEL_OVERRIDE] = model
    return headers


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


async def _attempt_oidc_refresh(
    entry: dict[str, Any], *, proxy: str | None = None
) -> dict[str, Any] | None:
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

    # Use headers that resemble the official Grok CLI client to improve
    # compatibility with any upstream protection on the token endpoint.
    version = resolve_grok_client_version()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": f"nanobot (grok-oidc; grok-cli/{version})",
    }

    timeout = httpx.Timeout(15.0, connect=10.0)
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": True,
        "trust_env": True,
    }
    if proxy:
        client_kwargs["proxy"] = proxy

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(
                GROK_OIDC_TOKEN_URL,
                data=data,
                headers=headers,
            )
            if resp.status_code >= 400:
                logger.debug(
                    "Grok OIDC refresh failed: {} {}",
                    resp.status_code,
                    resp.text[:300],
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
    logger.info("Refreshed Grok OIDC access token (expires_at={})", updated.get("expires_at"))
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
    - When ``apiBase`` points at the CLI chat proxy, injects the version and
      auth headers that proxy requires (``x-grok-client-version``, etc.).
    """

    def __init__(
        self,
        default_model: str = "grok-4",
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        proxy: str | None = None,
    ):
        self._grok_token: str | None = None
        self._grok_token_expires: float = 0.0
        self._last_refresh_attempt: float = 0.0
        self._using_oidc: bool = False
        self._configured_api_base = _normalize_api_base(api_base)
        self._uses_cli_proxy = is_cli_chat_proxy(self._configured_api_base)
        self._client_version = resolve_grok_client_version()
        self._user_extra_headers = dict(extra_headers or {})
        self._proxy = proxy or None

        self._explicit_api_key = self._resolve_explicit_api_key(provided=api_key)

        if self._explicit_api_key:
            key_for_client = self._explicit_api_key
            self._using_oidc = False
        else:
            # Placeholder until the first request loads/refreshes the OIDC JWT.
            key_for_client = None
            self._using_oidc = True

        headers = self._compose_headers(model=default_model)

        super().__init__(
            api_key=key_for_client,
            api_base=self._configured_api_base,
            default_model=default_model,
            extra_headers=headers,
            spec=find_by_name("grok"),
            proxy=self._proxy,
        )

    def _compose_headers(self, *, model: str | None = None) -> dict[str, str]:
        """Merge nanobot defaults, CLI-proxy requirements, and user overrides."""
        headers: dict[str, str] = {"User-Agent": "nanobot (grok-provider)"}
        target_model = model
        if not target_model and hasattr(self, "default_model"):
            target_model = self.default_model
        if self._uses_cli_proxy:
            headers.update(
                build_cli_proxy_headers(
                    model=target_model,
                    client_version=self._client_version,
                    include_token_auth=True,
                )
            )
        headers.update(self._user_extra_headers)
        return headers

    def _apply_headers(self, headers: dict[str, str]) -> None:
        """Write composed headers onto the live provider/client state."""
        changed = any(self._default_headers.get(key) != value for key, value in headers.items())
        self._default_headers.update(headers)
        self.extra_headers = dict(headers)
        if changed and self._client is not None:
            # Recreate so OpenAI client's frozen default_headers pick up changes.
            self._client = None

    def _sync_request_headers(self, model: str | None) -> None:
        """Keep live default headers aligned with the model for CLI proxy calls."""
        if not self._uses_cli_proxy:
            return
        target_model = model or self.default_model
        self._apply_headers(self._compose_headers(model=target_model))

    def _retarget_api_base(self, new_base: str) -> None:
        """Update configured base URL and refresh CLI-proxy header mode."""
        normalized = _normalize_api_base(new_base)
        self._configured_api_base = normalized
        self.api_base = normalized
        self._effective_base = normalized
        was_cli_proxy = self._uses_cli_proxy
        self._uses_cli_proxy = is_cli_chat_proxy(normalized)
        if was_cli_proxy != self._uses_cli_proxy or self._uses_cli_proxy:
            current_model = None
            if hasattr(self, "_default_headers"):
                current_model = self._default_headers.get(_HEADER_MODEL_OVERRIDE)
            self._apply_headers(
                self._compose_headers(model=current_model or self.default_model)
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
        """Return a valid JWT (either from key or from refreshed OIDC file).

        For OIDC (no explicit key) we always re-read ~/.grok/auth.json so that:
        - External updates by the official `grok` CLI (hot reload) are picked up.
        - Our own refresh can take effect for this and future calls immediately.
        - We decide refresh using the on-disk expires_at / refresh_token state.
        """
        now = time.time()

        # Prefer a real API key when one is configured (long-lived, no frequent re-read needed).
        explicit = self._resolve_explicit_api_key()
        if explicit:
            if self._using_oidc or self._needs_client_update(explicit):
                self._using_oidc = False
                await self._recreate_client_with_new_base(self._configured_api_base, explicit)
            self._grok_token = explicit
            self._grok_token_expires = now + 31_536_000
            return explicit

        # OIDC path: always consult the latest on-disk state (cheap) to support
        # background refreshes performed by `grok` itself or by us.
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
            refreshed = await _attempt_oidc_refresh(entry, proxy=self._proxy)
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
        self._retarget_api_base(new_base)
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

    def _is_likely_auth_error(self, exc: Exception) -> bool:
        """Heuristic to detect expired/invalid OIDC token errors from the xAI API."""
        text = str(exc).lower()
        if "401" in text or "unauthorized" in text or "invalid" in text and "token" in text:
            return True
        if "authentication" in text or "forbidden" in text and "auth" in text:
            return True
        # openai SDK errors
        name = type(exc).__name__.lower()
        if "auth" in name or "authenticationerror" in name:
            return True
        # Some responses surface status
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return True
        return False

    async def _force_oidc_refresh(self) -> bool:
        """Force an OIDC refresh attempt right now (ignoring normal time gates for this call)."""
        if not self._using_oidc:
            return False
        entry = load_grok_oidc_token()
        if not entry or not entry.get("refresh_token"):
            return False
        # Allow refresh even if we recently tried (user is explicitly recovering).
        self._last_refresh_attempt = 0.0
        refreshed = await _attempt_oidc_refresh(entry, proxy=self._proxy)
        if not refreshed:
            return False
        # Adopt the fresh token immediately
        token = refreshed.get("key")
        if not (isinstance(token, str) and token):
            return False
        exp = _parse_expires_at(refreshed.get("expires_at")) or (time.time() + 3600)
        self._grok_token = token
        self._grok_token_expires = exp
        if self._needs_client_update(token):
            await self._recreate_client_with_new_base(self._configured_api_base, token)
        self.api_key = token
        try:
            client = await self._ensure_client()
            client.api_key = token
        except Exception:
            pass
        logger.info("Grok OIDC token force-refreshed after auth error")
        return True

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
        provider_context: Any = None,
        **kwargs: Any,
    ):
        await self._refresh_client_api_key()
        # Apply after refresh: token refresh may recreate the client and reset
        # CLI proxy headers back to default_model.
        self._sync_request_headers(model)
        try:
            return await super().chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                provider_context=provider_context,
                **kwargs,
            )
        except Exception as exc:
            if self._using_oidc and self._is_likely_auth_error(exc):
                if await self._force_oidc_refresh():
                    self._sync_request_headers(model)
                    return await super().chat(
                        messages=messages,
                        tools=tools,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        tool_choice=tool_choice,
                        provider_context=provider_context,
                        **kwargs,
                    )
            raise

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
        provider_context: Any = None,
        **kwargs: Any,
    ):
        await self._refresh_client_api_key()
        self._sync_request_headers(model)
        try:
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
                provider_context=provider_context,
                **kwargs,
            )
        except Exception as exc:
            if self._using_oidc and self._is_likely_auth_error(exc):
                if await self._force_oidc_refresh():
                    self._sync_request_headers(model)
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
                        provider_context=provider_context,
                        **kwargs,
                    )
            raise

    def get_default_model(self) -> str:
        return self.default_model
