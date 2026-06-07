"""CustomKnight Creator - ComfyUI custom nodes.

Exposes the node mappings and registers the HTTP routes used by the web
extension. The top-level package ``__init__`` re-exports the mappings and the
``WEB_DIRECTORY`` that ComfyUI looks for.

The node classes depend on ``torch`` and the routes on ``aiohttp`` - both are
provided by a ComfyUI install. The imports are guarded so that the pure-Python
``sprite_handler`` core can still be imported (and unit-tested) in a bare
environment without those packages.
"""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:  # pragma: no cover - missing torch outside ComfyUI.
    import warnings

    warnings.warn(
        f"CustomKnight Creator nodes not loaded ({exc}). "
        "This is expected outside a ComfyUI environment.",
        stacklevel=2,
    )

try:
    # Importing the module registers the aiohttp routes as a side effect.
    from . import server_routes  # noqa: F401
except ImportError:  # pragma: no cover - missing aiohttp outside ComfyUI.
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
