"""ComfyUI entry point for the CustomKnight Creator custom nodes.

ComfyUI imports this top-level package and reads ``NODE_CLASS_MAPPINGS``,
``NODE_DISPLAY_NAME_MAPPINGS`` and ``WEB_DIRECTORY`` from it.
"""

from .src.customknight_creator import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Folder (relative to this file) that directly contains the JS extension
# files. ComfyUI serves each file at /extensions/<module>/<file>, so the JS
# imports resolve "../../scripts/app.js" -> "/scripts/app.js".
WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
