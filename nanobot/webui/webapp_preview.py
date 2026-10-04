"""Sandboxed HTML preview for a webapp directory the agent just wrote.

The returned document is meant for an iframe ``srcdoc`` without
``allow-same-origin``. Local files stay inside the entry file's directory and
are inlined, so the page cannot ask the gateway for sibling paths.
"""

from __future__ import annotations

import base64
import html
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from nanobot.security.workspace_access import WorkspaceScope
from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    is_path_within,
    resolve_allowed_path,
)
from nanobot.webui.file_preview import WebUIFilePreviewError, _display_path, _resolve_preview_path

MAX_WEBAPP_ASSET_BYTES = 512 * 1024
MAX_WEBAPP_TOTAL_BYTES = 2 * 1024 * 1024
MAX_WEBAPP_FILES = 40
_PAGE_SUFFIXES = {".html", ".htm"}
_TEXT_SUFFIXES = {".css", ".js", ".mjs", ".json", ".svg"}
_MIME_BY_SUFFIX = {
    ".css": "text/css",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "text/javascript",
    ".json": "application/json",
    ".mjs": "text/javascript",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}
_CSS_URL = re.compile(r"""url\(\s*(?P<q>['"]?)(?P<url>[^)'"]+)(?P=q)\s*\)""", re.IGNORECASE)
_CSS_IMPORT = re.compile(
    r"""@import\s+(?:url\(\s*)?['"](?P<url>[^'"]+)['"]\s*\)?[^;]*;""",
    re.IGNORECASE,
)
_JS_SPEC = re.compile(
    r"""(?P<prefix>(?:\bfrom\s*|\bimport\s*(?:\(\s*)?))(?P<q>['"])(?P<spec>\.[^'"]*)(?P=q)""",
    re.IGNORECASE,
)
_PREVIEW_CSP = (
    "<meta http-equiv=\"Content-Security-Policy\" content=\""
    "default-src 'none'; "
    "script-src 'unsafe-inline' data: blob: https: http:; "
    "style-src 'unsafe-inline' data: https: http:; "
    "img-src data: blob: https: http:; "
    "font-src data: https: http:; "
    "media-src data: blob: https: http:; "
    "connect-src data: blob: https: http:; "
    "worker-src blob:; "
    "object-src 'none'; "
    "base-uri about:; "
    "form-action 'none'\">"
    "<base href=\"about:blank\">"
)


class _Budget:
    def __init__(self) -> None:
        self.files = 0
        self.total = 0

    def charge(self, size: int) -> bool:
        if size < 0 or size > MAX_WEBAPP_ASSET_BYTES:
            return False
        if self.files >= MAX_WEBAPP_FILES or self.total + size > MAX_WEBAPP_TOTAL_BYTES:
            return False
        self.files += 1
        self.total += size
        return True


def webapp_page_preview_payload(
    raw_path: str | None,
    *,
    scope: WorkspaceScope,
) -> dict[str, str | int]:
    """Return one self-contained HTML document for a workspace page."""

    resolved = _resolve_preview_path(raw_path, scope=scope)
    if resolved.suffix.lower() not in _PAGE_SUFFIXES:
        raise WebUIFilePreviewError(415, "only html pages can be previewed")
    try:
        source = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise WebUIFilePreviewError(500, "failed to read file") from exc
    document = _render_document(source, resolved, scope)
    return {
        "kind": "page",
        "path": str(resolved),
        "display_path": _display_path(resolved, scope.project_path),
        "project_path": str(scope.project_path),
        "size": resolved.stat().st_size,
        "html": document,
    }


def _render_document(source: str, entry: Path, scope: WorkspaceScope) -> str:
    rewriter = _PreviewRewriter(entry, scope, _Budget())
    rewriter.feed(source)
    rewriter.close()
    return _PREVIEW_CSP + "".join(rewriter.parts)


class _PreviewRewriter(HTMLParser):
    def __init__(self, entry: Path, scope: WorkspaceScope, budget: _Budget) -> None:
        super().__init__(convert_charrefs=True)
        self.entry = entry
        self.scope = scope
        self.budget = budget
        self.root = entry.parent
        self.parts: list[str] = []
        self._skip_data = False
        self._in_style = False
        self._in_script = False
        self._bytes: dict[Path, bytes | None] = {}
        self._text: dict[Path, str] = {}
        self._modules: dict[Path, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._emit(tag, attrs, close=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._emit(tag, attrs, close=True)

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "script" and self._skip_data:
            self._skip_data = False
            self._in_script = False
            return
        if name == "script":
            self._in_script = False
        if name == "style":
            self._in_style = False
        if name != "base":
            self.parts.append(f"</{html.escape(name, quote=False)}>")

    def handle_data(self, data: str) -> None:
        if self._skip_data:
            return
        if self._in_style:
            self.parts.append(_escape_raw(self._rewrite_css(data, self.entry.parent, ())))
            return
        if self._in_script:
            self.parts.append(_escape_raw(data))
            return
        self.parts.append(html.escape(data))

    def handle_comment(self, data: str) -> None:
        if not self._skip_data:
            self.parts.append(f"<!--{data.replace('--', '')}-->")

    def handle_decl(self, decl: str) -> None:
        self.parts.append(f"<!{decl}>")

    def _emit(self, tag: str, attrs: list[tuple[str, str | None]], *, close: bool) -> None:
        name = tag.lower()
        if name == "base":
            return
        if name == "link" and _attr(attrs, "rel") and "stylesheet" in _attr(attrs, "rel").lower():
            href = _attr(attrs, "href")
            css = self._text_asset(href, self.entry.parent, ".css") if href else None
            if css is not None:
                self.parts.append(f"<style>{_escape_raw(self._rewrite_css(css, self._asset_dir(href), ()))}</style>")
                return
            if href and _url_kind(href) == "local":
                return
        if name == "script":
            src = _attr(attrs, "src")
            if src and _url_kind(src) == "local":
                module = (_attr(attrs, "type") or "").lower() == "module"
                script = self._javascript(src, ())
                if script is not None:
                    kind = " type=\"module\"" if module else ""
                    self.parts.append(f"<script{kind}>{_escape_raw(script)}</script>")
                self._skip_data = True
                return
        rewritten = [(key, self._rewrite_attr(name, key, value)) for key, value in attrs]
        self.parts.append(_format_tag(name, rewritten, close))
        if close:
            return
        if name == "style":
            self._in_style = True
        elif name == "script" and not self._skip_data:
            self._in_script = True

    def _rewrite_attr(self, tag: str, name: str, value: str | None) -> str | None:
        if value is None:
            return None
        key = name.lower()
        if key == "srcset" and tag in {"img", "source"}:
            return self._rewrite_srcset(value)
        if key in {"src", "poster"} and tag in {"img", "source", "video", "audio"}:
            return self._resource_url(value, self.entry.parent)
        if key == "src" and tag == "iframe":
            return value if _url_kind(value) == "keep" else "about:blank"
        return value

    def _rewrite_srcset(self, value: str) -> str:
        parts: list[str] = []
        for item in value.split(","):
            bits = item.strip().split()
            if not bits:
                continue
            url = self._resource_url(bits[0], self.entry.parent)
            if url == "about:blank":
                continue
            parts.append(" ".join([url, *bits[1:]]))
        return ", ".join(parts)

    def _resource_url(self, url: str, base: Path) -> str:
        kind = _url_kind(url)
        if kind == "keep":
            return url
        if kind != "local":
            return "about:blank"
        target = self._resolve(url, base)
        data = self._data_url(target) if target is not None else None
        return data or "about:blank"

    def _javascript(self, url: str, stack: tuple[Path, ...]) -> str | None:
        target = self._resolve(url, self.entry.parent)
        if target is None or target.suffix.lower() not in {".js", ".mjs"}:
            return None
        return self._rewrite_js(target, stack)

    def _rewrite_js(self, path: Path, stack: tuple[Path, ...]) -> str | None:
        if path in stack:
            return ""
        cached = self._modules.get(path)
        if cached is not None:
            return cached
        source = self._read_text(path)
        if source is None:
            return None

        def replace(match: re.Match[str]) -> str:
            spec = match.group("spec")
            target = self._resolve(spec, path.parent)
            if target is None:
                return match.group(0)
            if target.suffix.lower() == ".json":
                body = self._read_text(target)
                module = f"export default {body}" if body is not None else None
            elif target.suffix.lower() in {".js", ".mjs"}:
                module = self._rewrite_js(target, (*stack, path))
            else:
                module = None
            if module is None:
                return match.group(0)
            return f"{match.group('prefix')}{match.group('q')}{_js_data_url(module)}{match.group('q')}"

        rewritten = _JS_SPEC.sub(replace, source)
        self._modules[path] = rewritten
        return rewritten

    def _rewrite_css(self, source: str, base: Path, stack: tuple[Path, ...]) -> str:
        def replace_import(match: re.Match[str]) -> str:
            target = self._resolve(match.group("url"), base)
            if target is None or target.suffix.lower() != ".css" or target in stack:
                return "" if _url_kind(match.group("url")) == "local" else match.group(0)
            imported = self._read_text(target)
            if imported is None:
                return ""
            return self._rewrite_css(imported, target.parent, (*stack, target))

        def replace_url(match: re.Match[str]) -> str:
            url = match.group("url").strip()
            if _url_kind(url) != "local":
                return match.group(0)
            data = self._data_url(self._resolve(url, base))
            return f"url('{data}')" if data else "url('')"

        return _CSS_URL.sub(replace_url, _CSS_IMPORT.sub(replace_import, source))

    def _text_asset(self, url: str, base: Path, suffix: str) -> str | None:
        target = self._resolve(url, base)
        if target is None or target.suffix.lower() != suffix:
            return None
        return self._read_text(target)

    def _asset_dir(self, url: str | None) -> Path:
        if not url:
            return self.entry.parent
        target = self._resolve(url, self.entry.parent)
        return target.parent if target is not None else self.entry.parent

    def _data_url(self, path: Path | None) -> str | None:
        if path is None:
            return None
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
        raw = self._read_bytes(path)
        if mime is None or raw is None:
            return None
        if path.suffix.lower() in _TEXT_SUFFIXES and path.suffix.lower() != ".svg":
            return None
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _resolve(self, url: str, base: Path) -> Path | None:
        if _url_kind(url) != "local":
            return None
        relative = unquote(urlsplit(url.strip()).path)
        if not relative or relative in {".", "./"}:
            return None
        try:
            resolved = resolve_allowed_path(
                str(base / relative),
                workspace=self.scope.project_path,
                allowed_root=(
                    self.scope.project_path if self.scope.restrict_to_workspace else self.root
                ),
                strict=True,
            )
        except (WorkspaceBoundaryError, OSError):
            return None
        if not resolved.is_file() or not is_path_within(resolved, self.root):
            return None
        return resolved

    def _read_text(self, path: Path) -> str | None:
        cached = self._text.get(path)
        if cached is not None:
            return cached
        raw = self._read_bytes(path)
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace")
        self._text[path] = text
        return text

    def _read_bytes(self, path: Path) -> bytes | None:
        if path in self._bytes:
            return self._bytes[path]
        try:
            with open(path, "rb") as handle:
                raw = handle.read(MAX_WEBAPP_ASSET_BYTES + 1)
        except OSError:
            self._bytes[path] = None
            return None
        if not self.budget.charge(len(raw)):
            self._bytes[path] = None
            return None
        self._bytes[path] = raw
        return raw


def _url_kind(url: str) -> str:
    value = url.strip()
    if not value or value.startswith(("#", "data:", "blob:", "mailto:", "javascript:")):
        return "keep"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return "keep"
    path = parsed.path
    if not path or path.startswith(("/", "\\")) or "\\" in path or "\0" in path:
        return "drop"
    return "local"


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for key, value in attrs:
        if key.lower() == name and value is not None:
            return value
    return None


def _format_tag(tag: str, attrs: list[tuple[str, str | None]], close: bool) -> str:
    rendered = "".join(
        f" {html.escape(key, quote=False)}"
        if value is None
        else f" {html.escape(key, quote=False)}=\"{html.escape(value, quote=True)}\""
        for key, value in attrs
    )
    ending = " />" if close else ">"
    return f"<{tag}{rendered}{ending}"


def _escape_raw(value: str) -> str:
    return value.replace("</", "<\\/")


def _js_data_url(source: str) -> str:
    return "data:text/javascript;charset=utf-8," + quote(source, safe="")
