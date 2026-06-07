"""ComfyUI entry point for the CustomKnight Creator custom nodes.

ComfyUI imports this top-level package and reads ``NODE_CLASS_MAPPINGS``,
``NODE_DISPLAY_NAME_MAPPINGS`` and ``WEB_DIRECTORY`` from it.
"""

from .src.customknight_creator import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Folder (relative to this file) that ComfyUI serves JS extensions from.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
