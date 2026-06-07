"""Core sprite logic for the CustomKnight Creator ComfyUI nodes.

This module is a faithful port of the original CustomKnight-Creator
(https://github.com/cmot17/CustomKnight-Creator) ``spritehandler.py`` /
``spritepacker.py`` logic, rewritten in clean, dependency-light Python.

It depends only on the standard library and Pillow (no torch / ComfyUI), so it
can be reused both by the node implementations and by the lightweight HTTP
routes that power the dynamic dropdowns and the animation preview.

Data model (matching the CustomKnight sprite dump format)
---------------------------------------------------------
* A *root folder* is a top level dump directory (e.g. ``Knight``) that contains
  ``0.Atlases/SpriteInfo.json``.
* All selected root folders must live in the same parent directory. That parent
  is the project ``basepath``; every sprite ``path`` is relative to it.
* ``SpriteInfo.json`` stores parallel arrays. The keys are::

      sid             sprite id
      sx, sy          sprite position inside the packed atlas (bottom-left origin)
      sxr, syr        sprite region inside its own source PNG (bottom-left origin)
      swidth, sheight sprite width / height
      sfilpped        whether the sprite is rotated 90 degrees in the atlas
      spath           sprite PNG path, relative to basepath
      scollectionname the collection (atlas) the sprite belongs to

* A *collection* maps to a single output atlas ``<collection>.png``.
* An *animation* is ``basename(dirname(spath))`` - a folder of frame PNGs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image

SPRITE_INFO_RELPATH = os.path.join("0.Atlases", "SpriteInfo.json")

# The original JSON ships with the misspelled key "sfilpped"; we accept a few
# spellings so hand-edited / future dumps keep working.
_FLIPPED_KEYS = ("sfilpped", "sflipped", "sflipped", "flipped")


@dataclass
class Sprite:
    """A single sprite entry parsed from ``SpriteInfo.json``."""

    id: str
    x: int
    y: int
    xr: int
    yr: int
    w: int
    h: int
    flipped: bool
    path: str          # relative to the project basepath
    collection: str

    @property
    def animation(self) -> str:
        """The animation folder name this frame belongs to."""
        return os.path.basename(os.path.dirname(self.path))

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "xr": self.xr,
            "yr": self.yr,
            "w": self.w,
            "h": self.h,
            "flipped": self.flipped,
            "path": self.path,
            "collection": self.collection,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sprite":
        return cls(
            id=d["id"],
            x=int(d["x"]),
            y=int(d["y"]),
            xr=int(d["xr"]),
            yr=int(d["yr"]),
            w=int(d["w"]),
            h=int(d["h"]),
            flipped=bool(d["flipped"]),
            path=d["path"],
            collection=d["collection"],
        )


def parse_root_folders(text: str | Iterable[str]) -> list[str]:
    """Split the multiline ``root_folders`` widget value into folder paths.

    Accepts newline separated paths. Blank lines and surrounding quotes /
    whitespace are stripped. The order is preserved and duplicates removed.
    """
    if isinstance(text, str):
        lines = text.splitlines()
    else:
        lines = list(text)

    folders: list[str] = []
    for line in lines:
        folder = line.strip().strip('"').strip("'")
        if folder and folder not in folders:
            folders.append(os.path.normpath(folder))
    return folders


def _next_power_of_two(n: int) -> int:
    """Smallest power of two that is >= ``n`` (>= 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class SpriteProject:
    """Parsed view over one or more CustomKnight root folders."""

    def __init__(self, basepath: str, sprites: list[Sprite]):
        self.basepath = basepath
        self.sprites = sprites

    # ------------------------------------------------------------------ load
    @classmethod
    def from_root_folders(cls, root_folders: str | Iterable[str]) -> "SpriteProject":
        folders = parse_root_folders(root_folders)
        if not folders:
            raise ValueError("No root folders provided.")

        basepath: str | None = None
        sprites: list[Sprite] = []

        for folder in folders:
            info_path = os.path.join(folder, SPRITE_INFO_RELPATH)
            if not os.path.isfile(info_path):
                raise FileNotFoundError(
                    f'"{info_path}" not found. A root folder must contain '
                    f'"{SPRITE_INFO_RELPATH}".'
                )

            parent = os.path.dirname(os.path.abspath(folder))
            if basepath is None:
                basepath = parent
            elif os.path.normcase(parent) != os.path.normcase(basepath):
                raise ValueError(
                    "Inconsistent base path: all root folders must live in the "
                    "same parent directory (e.g. the 'Knight' and 'Spells Anim' "
                    "folders should sit side by side)."
                )

            sprites.extend(cls._parse_sprite_info(info_path))

        assert basepath is not None
        return cls(basepath=basepath, sprites=sprites)

    @staticmethod
    def _parse_sprite_info(info_path: str) -> list[Sprite]:
        with open(info_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        flipped = None
        for key in _FLIPPED_KEYS:
            if key in data:
                flipped = data[key]
                break
        if flipped is None:
            flipped = [False] * len(data["sid"])

        sprites: list[Sprite] = []
        for i in range(len(data["sid"])):
            sprites.append(
                Sprite(
                    id=data["sid"][i],
                    x=int(data["sx"][i]),
                    y=int(data["sy"][i]),
                    xr=int(data["sxr"][i]),
                    yr=int(data["syr"][i]),
                    w=int(data["swidth"][i]),
                    h=int(data["sheight"][i]),
                    flipped=bool(flipped[i]),
                    path=data["spath"][i].replace("\\", "/"),
                    collection=data["scollectionname"][i],
                )
            )
        return sprites

    # -------------------------------------------------------------- queries
    def collections(self) -> list[str]:
        """Unique collection (atlas) names, in first-seen order."""
        result: list[str] = []
        for sprite in self.sprites:
            if sprite.collection not in result:
                result.append(sprite.collection)
        return result

    def animations(self, collection: str | None = None, path_filter: str = "") -> list[str]:
        """Unique animation names, optionally limited to one collection.

        ``path_filter`` is a case-insensitive substring match on the sprite
        path, mirroring the search box of the original tool.
        """
        needle = path_filter.casefold()
        result: list[str] = []
        for sprite in self._filtered(collection):
            if needle and needle not in sprite.path.casefold():
                continue
            anim = sprite.animation
            if anim not in result:
                result.append(anim)
        return result

    def sprites_in_animation(
        self, animation: str, collection: str | None = None
    ) -> list[Sprite]:
        """Frames of one animation, preserving SpriteInfo (playback) order."""
        return [
            sprite
            for sprite in self._filtered(collection)
            if sprite.animation == animation
        ]

    def sprites_in_collection(self, collection: str) -> list[Sprite]:
        return [s for s in self.sprites if s.collection == collection]

    def _filtered(self, collection: str | None) -> list[Sprite]:
        if collection in (None, "", "All", "(All)"):
            return self.sprites
        return [s for s in self.sprites if s.collection == collection]

    # ----------------------------------------------------------------- io
    def source_path(self, sprite: Sprite) -> str:
        """Absolute path of a sprite's source PNG on disk."""
        return os.path.join(self.basepath, sprite.path)

    def crop_content(self, sprite: Sprite) -> Image.Image:
        """Open a sprite's source PNG and crop it to its ``w x h`` content.

        Uses the same bottom-left-origin crop box as the original packer.
        Returns an upright (not yet flipped) RGBA image of size ``(w, h)``.
        """
        im = Image.open(self.source_path(sprite)).convert("RGBA")
        return self._crop_content_from(im, sprite)

    @staticmethod
    def _crop_content_from(im: Image.Image, sprite: Sprite) -> Image.Image:
        height = im.size[1]
        box = (
            sprite.xr,
            height - sprite.yr - sprite.h,
            sprite.xr + sprite.w,
            height - sprite.yr,
        )
        return im.crop(box)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
@dataclass
class AtlasResult:
    image: Image.Image
    collection: str
    width: int
    height: int
    sprite_count: int
    replaced_count: int


def atlas_size_for(sprites: list[Sprite]) -> tuple[int, int]:
    """Power-of-two atlas dimensions large enough to hold every sprite.

    A sprite placed at ``(x, y)`` (bottom-left origin) occupies, in atlas
    pixels, the right extent ``x + (h if flipped else w)`` and the top extent
    ``y + (w if flipped else h)``. The atlas is rounded up to the next power of
    two on each axis, matching the original tool's output.
    """
    max_w = 0
    max_h = 0
    for s in sprites:
        if s.flipped:
            max_w = max(max_w, s.x + s.h)
            max_h = max(max_h, s.y + s.w)
        else:
            max_w = max(max_w, s.x + s.w)
            max_h = max(max_h, s.y + s.h)
    return _next_power_of_two(max_w), _next_power_of_two(max_h)


def place_sprite(atlas: Image.Image, sprite: Sprite, content: Image.Image) -> None:
    """Paste an upright ``(w, h)`` sprite content image into ``atlas``.

    Applies the 90-degree rotation + horizontal flip used for "flipped"
    sprites, exactly as the original packer did, and positions the result with
    a bottom-left origin.
    """
    height = atlas.size[1]
    if sprite.flipped:
        x = sprite.x
        y = height - sprite.y - sprite.w
        content = content.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    else:
        x = sprite.x
        y = height - sprite.y - sprite.h
    atlas.paste(content, (x, y))


def pack_collection(
    project: SpriteProject,
    collection: str,
    replacements: dict[str, Image.Image] | None = None,
    size: tuple[int, int] | None = None,
) -> AtlasResult:
    """Build an atlas image for ``collection``.

    ``replacements`` maps a sprite's relative path to an upright ``(w, h)`` RGBA
    image to use instead of the on-disk source (this is how edited frames are
    injected). Any sprite without a replacement is read from disk unchanged.
    """
    replacements = replacements or {}
    sprites = project.sprites_in_collection(collection)
    if not sprites:
        raise ValueError(f'Collection "{collection}" has no sprites.')

    width, height = size or atlas_size_for(sprites)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    replaced = 0
    for sprite in sprites:
        replacement = replacements.get(sprite.path)
        if replacement is not None:
            content = replacement
            if content.size != (sprite.w, sprite.h):
                content = content.resize((sprite.w, sprite.h), Image.NEAREST)
            replaced += 1
        else:
            content = project.crop_content(sprite)
        place_sprite(atlas, sprite, content)

    return AtlasResult(
        image=atlas,
        collection=collection,
        width=width,
        height=height,
        sprite_count=len(sprites),
        replaced_count=replaced,
    )
