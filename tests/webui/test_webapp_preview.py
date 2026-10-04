import base64
from pathlib import Path

import pytest

from nanobot.security.workspace_access import default_workspace_scope
from nanobot.webui.file_preview import WebUIFilePreviewError, file_preview_payload
from nanobot.webui.webapp_preview import webapp_page_preview_payload

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
)


def _page(tmp_path: Path, name: str = "index.html") -> str:
    scope = default_workspace_scope(tmp_path, restrict_to_workspace=True)
    return str(webapp_page_preview_payload(name, scope=scope)["html"])


def test_page_preview_inlines_local_assets_and_keeps_external_urls(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "app.css").write_text("body{background:url('pixel.png')}", encoding="utf-8")
    (app / "pixel.png").write_bytes(PNG)
    (app / "app.js").write_text("document.body.dataset.ready = 'yes'", encoding="utf-8")
    (app / "index.html").write_text(
        """<!DOCTYPE html><html><head>
        <base href="https://evil.example/">
        <link rel="stylesheet" href="app.css">
        <script src="https://cdn.example/lib.js"></script>
        </head><body>
        <img src="pixel.png" alt="pixel">
        <script src="app.js"></script>
        </body></html>""",
        encoding="utf-8",
    )
    document = _page(app)
    assert "document.body.dataset.ready" in document
    assert "background:url('data:image/png;base64," in document
    assert "https://cdn.example/lib.js" in document
    assert "https://evil.example" not in document
    assert '<base href="about:blank">' in document
    assert "Content-Security-Policy" in document
    assert 'src="app.css"' not in document
    assert 'src="app.js"' not in document
    assert 'src="pixel.png"' not in document


def test_page_preview_rewrites_local_module_imports(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "lib.js").write_text("export const label = 'from-lib'", encoding="utf-8")
    (app / "main.js").write_text("import { label } from './lib.js'", encoding="utf-8")
    (app / "index.html").write_text(
        '<script type="module" src="main.js"></script>',
        encoding="utf-8",
    )
    document = _page(app)
    assert "from-lib" in document
    assert 'type="module"' in document
    assert "data:text/javascript" in document
    assert "./lib.js" not in document


def test_page_preview_does_not_inline_files_outside_the_app_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app = workspace / "app"
    app.mkdir(parents=True)
    (workspace / "secret.js").write_text("secret-token", encoding="utf-8")
    outside = tmp_path / "outside.js"
    outside.write_text("outside-token", encoding="utf-8")
    (app / "linked.js").symlink_to(outside)
    (app / "index.html").write_text(
        '<script src="../secret.js"></script><script src="linked.js"></script>',
        encoding="utf-8",
    )
    document = _page(workspace / "app", "index.html")
    assert "secret-token" not in document
    assert "outside-token" not in document
    assert 'src="../secret.js"' not in document
    assert 'src="linked.js"' not in document


def test_inlined_script_cannot_close_the_preview_tag(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text(
        "const marker = '</script><script>window.escaped = true'",
        encoding="utf-8",
    )
    (tmp_path / "index.html").write_text('<script src="app.js"></script>', encoding="utf-8")
    document = _page(tmp_path)
    assert "</script><script>window.escaped" not in document
    assert "<\\/script>" in document


def test_source_preview_of_html_stays_text(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<script>window.example = true</script>", encoding="utf-8")
    result = file_preview_payload(
        "index.html",
        scope=default_workspace_scope(tmp_path, restrict_to_workspace=True),
    )
    assert result["kind"] == "text"
    assert result["content"] == "<script>window.example = true</script>"


def test_page_preview_rejects_non_html(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(WebUIFilePreviewError) as error:
        webapp_page_preview_payload(
            "notes.txt",
            scope=default_workspace_scope(tmp_path, restrict_to_workspace=True),
        )
    assert error.value.status == 415
