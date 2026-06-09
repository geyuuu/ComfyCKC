"""HTTP routes backing the CustomKnight Creator web extension.

These power the dynamic dropdowns (collections / animations) and the animation
preview on ``CKAnimationSelector``. They are registered against ComfyUI's
aiohttp server (``PromptServer.instance.routes``) when this package is imported.

All routes are read-only and validate that any served file lives underneath the
selected project's base path, so they cannot be used to read arbitrary disk
locations.
"""

from __future__ import annotations

import os

from aiohttp import web

from .sprite_handler import SpriteProject, parse_index_range, parse_root_folders

try:
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI.
    PromptServer = None


def _err(message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _load_project(root_folders: str) -> SpriteProject:
    return SpriteProject.from_root_folders(root_folders)


def register_routes() -> None:
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return

    routes = PromptServer.instance.routes

    @routes.get("/customknight/collections")
    async def collections(request: web.Request):
        root_folders = request.query.get("root_folders", "")
        try:
            project = _load_project(root_folders)
        except Exception as exc:  # noqa: BLE001 - surface message to the UI.
            return _err(str(exc))
        return web.json_response({"collections": project.collections()})

    @routes.get("/customknight/animations")
    async def animations(request: web.Request):
        root_folders = request.query.get("root_folders", "")
        collection = request.query.get("collection", "") or None
        path_filter = request.query.get("filter", "")
        try:
            project = _load_project(root_folders)
        except Exception as exc:  # noqa: BLE001
            return _err(str(exc))
        return web.json_response(
            {"animations": project.animations(collection, path_filter)}
        )

    @routes.get("/customknight/frames")
    async def frames(request: web.Request):
        root_folders = request.query.get("root_folders", "")
        collection = request.query.get("collection", "") or None
        animation = request.query.get("animation", "")
        try:
            project = _load_project(root_folders)
            sprites = project.sprites_in_animation(animation, collection)
        except Exception as exc:  # noqa: BLE001
            return _err(str(exc))
        return web.json_response(
            {
                "frames": [
                    {"path": s.path, "name": s.filename, "w": s.w, "h": s.h}
                    for s in sprites
                ]
            }
        )

    @routes.get("/customknight/range_frames")
    async def range_frames(request: web.Request):
        root_folders = request.query.get("root_folders", "")
        range_text = request.query.get("range", "")
        try:
            project = _load_project(root_folders)
            numbers = parse_index_range(range_text)
            sprites = project.sprites_in_animation_range(numbers)
        except Exception as exc:  # noqa: BLE001
            return _err(str(exc))
        return web.json_response(
            {
                "frames": [
                    {
                        "path": s.path,
                        "name": s.filename,
                        "w": s.w,
                        "h": s.h,
                        "collection": s.collection,
                    }
                    for s in sprites
                ]
            }
        )

    @routes.get("/customknight/image")
    async def image(request: web.Request):
        root_folders = request.query.get("root_folders", "")
        rel_path = request.query.get("path", "")
        if not parse_root_folders(root_folders):
            return _err("missing root_folders")
        try:
            project = _load_project(root_folders)
        except Exception as exc:  # noqa: BLE001
            return _err(str(exc))

        base = os.path.abspath(project.basepath)
        target = os.path.abspath(os.path.join(base, rel_path))
        # Confine reads to the project base path.
        if os.path.commonpath([base, target]) != base:
            return _err("forbidden path", status=403)
        if not os.path.isfile(target):
            return _err("not found", status=404)
        return web.FileResponse(target)


register_routes()
