"""ComfyUI nodes for CustomKnight Creator.

Two core nodes mirror the original desktop tool:

``CKAnimationSelector``
    Input is one or more *Root Folders*. Pick a collection (atlas) and an
    animation - exactly like the original - and the node outputs that
    animation's RGBA frames as a PNG image sequence (an ``IMAGE`` batch) plus a
    ``CK_FRAMES`` descriptor used for repacking. A live animated preview is
    rendered on the node by the bundled web extension.

``CKPackAtlas``
    Takes the (possibly edited) image sequence back, drops the modified frames
    on top of all the *unchanged* sprites of the collection, and packs the lot
    into a single atlas PNG - the file CustomKnight loads.

Helper nodes (``CKMergeEdits``, ``CKLoadProjectInfo``) make multi-animation
workflows and inspection convenient.
"""

from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from PIL import Image

from .sprite_handler import (
    Sprite,
    SpriteProject,
    atlas_size_for,
    pack_collection,
    parse_index_range,
)

try:  # ComfyUI runtime - always present in a real install.
    import folder_paths
except Exception:  # pragma: no cover - lets the module import for tests.
    folder_paths = None


CATEGORY = "CustomKnight"
_COMBO_PLACEHOLDER = "<refresh>"


# ---------------------------------------------------------------------------
# Tensor <-> PIL helpers
# ---------------------------------------------------------------------------
def pil_rgba_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGBA ``[1, H, W, 4]`` float tensor in ``[0, 1]``.

    Keeps the alpha channel so the output IMAGE renders with transparency in a
    downstream Preview/Save node, matching the saved atlas PNG. Dropping alpha
    here (``convert("RGB")``) would expose the arbitrary RGB that sprite atlases
    leave in fully-transparent texels (white boxes / colour noise).
    """
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def tensor_to_pil_rgba(image: torch.Tensor) -> Image.Image:
    """Convert one frame (``[H, W, C]`` or ``[1, H, W, C]``) to an RGBA PIL image.

    The image's fourth channel supplies alpha, falling back to fully opaque for
    an RGB input.
    """
    if image.dim() == 4:
        image = image[0]
    rgb = (image[..., :3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    h, w = rgb.shape[:2]

    if image.shape[-1] >= 4:
        alpha = (image[..., 3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        alpha = np.full((h, w), 255, dtype=np.uint8)

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, "RGBA")


def stack_frames(frames: list[Image.Image]) -> tuple[torch.Tensor, int, int]:
    """Pad ``(w, h)`` RGBA frames to a common size and stack into batches.

    Frames are anchored top-left on a transparent canvas sized to the largest
    frame, so a single RGBA ``IMAGE`` batch can carry differently sized
    sprites. Returns ``(images, max_w, max_h)``.
    """
    max_w = max((f.size[0] for f in frames), default=1)
    max_h = max((f.size[1] for f in frames), default=1)

    imgs: list[torch.Tensor] = []
    for frame in frames:
        canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        canvas.paste(frame, (0, 0))
        imgs.append(pil_rgba_to_tensor(canvas))

    return torch.cat(imgs, 0), max_w, max_h


def _project_from(root_folders: str) -> SpriteProject:
    return SpriteProject.from_root_folders(root_folders)


# ---------------------------------------------------------------------------
# CKAnimationSelector
# ---------------------------------------------------------------------------
class CKAnimationSelector:
    """Select an animation from a CustomKnight dump and emit its frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_folders": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": (
                            "One root folder per line, e.g.\n"
                            "D:/MySkin/Knight\nD:/MySkin/Spells Anim"
                        ),
                        "tooltip": "Top-level dump folders, each containing "
                        "0.Atlases/SpriteInfo.json.",
                    },
                ),
                # Combos are filled by the web extension from the server. The
                # placeholder keeps the value list non-empty before a refresh.
                "collection": (
                    [_COMBO_PLACEHOLDER],
                    {"tooltip": "Atlas / collection to edit."},
                ),
                "animation": (
                    [_COMBO_PLACEHOLDER],
                    {"tooltip": "Animation folder whose frames are output "
                    "(single-animation mode)."},
                ),
                "mode": (
                    ["single animation", "animation range"],
                    {
                        "default": "single animation",
                        "tooltip": "'single animation' outputs the collection + "
                        "animation chosen above. 'animation range' ignores them "
                        "and concatenates every animation whose folder-name "
                        "number falls in 'animation_range'.",
                    },
                ),
                "animation_range": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Animation-range mode only. Numbers taken from "
                        "the animation folder names (e.g. '034.Idle' -> 34). "
                        "Examples: '1-10', '34', '1-3,5,7-9'. Ranges are "
                        "inclusive; frames are concatenated in animation order.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CK_FRAMES")
    RETURN_NAMES = ("frames", "ck_frames")
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Pick a collection and animation from CustomKnight Root Folders and "
        "output the frames as an RGBA PNG image sequence (plus a CK_FRAMES "
        "descriptor for repacking). Switch 'mode' to 'animation range' to "
        "concatenate every animation in a number range (e.g. '1-10') into one "
        "sequence - the frames may span several collections and still repack."
    )

    # Combos are populated dynamically on the client, so skip the built-in
    # "value must be in the declared list" validation for them.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def select(
        self,
        root_folders: str,
        collection: str,
        animation: str,
        mode: str = "single animation",
        animation_range: str = "",
    ):
        project = _project_from(root_folders)

        if mode == "animation range":
            numbers = parse_index_range(animation_range)
            if not numbers:
                raise ValueError(
                    "Animation-range mode: enter a range like “1-10” "
                    "(numbers come from the animation folder names)."
                )
            sprites = project.sprites_in_animation_range(numbers)
            if not sprites:
                raise ValueError(
                    "No animations matched folder-name numbers in "
                    f"“{animation_range.strip()}”."
                )
            animation_label = f"range:{animation_range.strip()}"
            collection_value = None
            collections_present: list[str] = []
            for s in sprites:
                if s.collection not in collections_present:
                    collections_present.append(s.collection)
        else:
            if collection in ("", _COMBO_PLACEHOLDER):
                raise ValueError(
                    "Select a collection (click 'Refresh', then choose one)."
                )
            if animation in ("", _COMBO_PLACEHOLDER):
                raise ValueError("Select an animation.")
            sprites = project.sprites_in_animation(animation, collection)
            if not sprites:
                raise ValueError(
                    f'No frames found for animation "{animation}" in collection '
                    f'"{collection}".'
                )
            animation_label = animation
            collection_value = collection
            collections_present = [collection]

        frames = [project.crop_content(s) for s in sprites]
        images, max_w, max_h = stack_frames(frames)

        ck_frames = {
            "root_folders": root_folders,
            "basepath": project.basepath,
            "mode": mode,
            "collection": collection_value,
            "collections": collections_present,
            "animation": animation_label,
            "frame_size": [max_w, max_h],
            "sprites": [s.to_dict() for s in sprites],
        }
        return (images, ck_frames)


# ---------------------------------------------------------------------------
# Reversible frame-sheet helpers
# ---------------------------------------------------------------------------
_SHEET_ALIGNMENT = 16
_SHEET_MIN_PIXELS = 655_360
_SHEET_MAX_PIXELS = 8_294_400


def _frame_sheet_size(content_width: int, content_height: int) -> tuple[int, int]:
    """Return a lossless, 16-aligned sheet size within custom-resolution limits."""
    alignment = _SHEET_ALIGNMENT
    sheet_width = ((content_width + alignment - 1) // alignment) * alignment
    sheet_height = ((content_height + alignment - 1) // alignment) * alignment
    pixel_count = sheet_width * sheet_height

    if pixel_count > _SHEET_MAX_PIXELS:
        raise ValueError(
            "CK Frames to Sheet cannot make a lossless custom-resolution sheet: "
            f"the {sheet_width}x{sheet_height} aligned frame grid contains "
            f"{pixel_count:,} pixels, exceeding the {_SHEET_MAX_PIXELS:,} maximum. "
            "Use a different column count or fewer/smaller frames."
        )
    if pixel_count >= _SHEET_MIN_PIXELS:
        return sheet_width, sheet_height

    # Grow toward the minimum while preserving the content grid's aspect ratio.
    # Rounding both dimensions upward keeps the final area at or above the
    # minimum; an already-constraining content dimension is never reduced.
    aspect_ratio = content_width / content_height
    target_width = math.sqrt(_SHEET_MIN_PIXELS * aspect_ratio)
    target_height = math.sqrt(_SHEET_MIN_PIXELS / aspect_ratio)
    if target_width < sheet_width:
        target_width = sheet_width
        target_height = max(target_height, _SHEET_MIN_PIXELS / target_width)
    elif target_height < sheet_height:
        target_height = sheet_height
        target_width = max(target_width, _SHEET_MIN_PIXELS / target_height)

    sheet_width = max(
        sheet_width,
        math.ceil(target_width / alignment) * alignment,
    )
    sheet_height = max(
        sheet_height,
        math.ceil(target_height / alignment) * alignment,
    )
    pixel_count = sheet_width * sheet_height
    if not _SHEET_MIN_PIXELS <= pixel_count <= _SHEET_MAX_PIXELS:
        raise ValueError(
            "CK Frames to Sheet could not satisfy the custom-resolution pixel limits "
            "without resizing frame pixels."
        )
    return sheet_width, sheet_height


class CKFramesToSheet:
    """Arrange an IMAGE batch on a 16-aligned sheet without changing pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "columns": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": (
                            "Grid columns. 0 chooses a compact, near-square grid; "
                            "1 makes a vertical strip."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CK_FRAME_SHEET")
    RETURN_NAMES = ("sheet", "sheet_layout")
    FUNCTION = "combine"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Arrange every frame in an IMAGE batch on one grid image. The layout "
        "is transparently padded to a multiple of 16 in both dimensions and "
        "between 655,360 and 8,294,400 total pixels. CK Sheet to Frames then "
        "restores the original batch exactly."
    )

    def combine(self, frames: torch.Tensor, columns: int = 0):
        if frames.dim() != 4:
            raise ValueError(
                "CK Frames to Sheet expects an IMAGE batch shaped [frames, height, width, channels]."
            )

        frame_count, frame_height, frame_width, channels = frames.shape
        if frame_count < 1:
            raise ValueError("CK Frames to Sheet received an empty frame batch.")
        if frame_height < 1 or frame_width < 1 or channels < 1:
            raise ValueError("CK Frames to Sheet received frames with an invalid shape.")

        columns = int(columns)
        if columns <= 0:
            columns = math.ceil(math.sqrt(frame_count))
        columns = min(columns, frame_count)
        rows = math.ceil(frame_count / columns)
        content_height = rows * frame_height
        content_width = columns * frame_width
        sheet_width, sheet_height = _frame_sheet_size(content_width, content_height)

        # new_zeros retains the input tensor's dtype and device. Direct slice
        # assignment performs no resize, colour conversion, or quantisation.
        # Alignment padding is added only to the right and bottom, outside all
        # frame cells, so it can never become part of a restored frame.
        sheet = frames.new_zeros((1, sheet_height, sheet_width, channels))
        for index in range(frame_count):
            row, column = divmod(index, columns)
            y = row * frame_height
            x = column * frame_width
            sheet[0, y : y + frame_height, x : x + frame_width, :] = frames[index]

        layout = {
            "version": 3,
            "frame_count": frame_count,
            "frame_height": frame_height,
            "frame_width": frame_width,
            "channels": channels,
            "columns": columns,
            "rows": rows,
            "sheet_height": sheet_height,
            "sheet_width": sheet_width,
        }
        return (sheet, layout)


class CKSheetToFrames:
    """Restore the IMAGE batch described by a CK_FRAME_SHEET layout."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet": ("IMAGE",),
                "sheet_layout": ("CK_FRAME_SHEET",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Split an image made by CK Frames to Sheet back into its original "
        "ordered IMAGE batch, with identical frame dimensions and values."
    )

    def split(self, sheet: torch.Tensor, sheet_layout):
        if sheet.dim() != 4 or sheet.shape[0] != 1:
            raise ValueError("CK Sheet to Frames expects exactly one sheet image.")
        if (
            not isinstance(sheet_layout, dict)
            or sheet_layout.get("version") not in (1, 2, 3)
        ):
            raise ValueError("CK Sheet to Frames received an unsupported sheet layout.")

        keys = (
            "frame_count",
            "frame_height",
            "frame_width",
            "channels",
            "columns",
            "rows",
        )
        try:
            frame_count, frame_height, frame_width, channels, columns, rows = (
                int(sheet_layout[key]) for key in keys
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("CK Sheet to Frames received an invalid sheet layout.") from exc

        if min(frame_count, frame_height, frame_width, channels, columns, rows) < 1:
            raise ValueError("CK Sheet to Frames received an invalid sheet layout.")
        if sheet_layout["version"] in (2, 3):
            try:
                sheet_height = int(sheet_layout["sheet_height"])
                sheet_width = int(sheet_layout["sheet_width"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "CK Sheet to Frames received an invalid sheet layout."
                ) from exc
            content_height = rows * frame_height
            content_width = columns * frame_width
            if sheet_layout["version"] == 2:
                expected_height = ((content_height + 15) // 16) * 16
                expected_width = ((content_width + 15) // 16) * 16
                valid_size = (sheet_height, sheet_width) == (
                    expected_height,
                    expected_width,
                )
            else:
                pixel_count = sheet_height * sheet_width
                valid_size = (
                    sheet_height % _SHEET_ALIGNMENT == 0
                    and sheet_width % _SHEET_ALIGNMENT == 0
                    and sheet_height >= content_height
                    and sheet_width >= content_width
                    and _SHEET_MIN_PIXELS <= pixel_count <= _SHEET_MAX_PIXELS
                )
            if not valid_size:
                raise ValueError("CK Sheet to Frames received an invalid sheet layout.")
        else:
            # Backward compatibility for layouts created before sheet padding
            # was introduced.
            sheet_height = rows * frame_height
            sheet_width = columns * frame_width

        content_height = rows * frame_height
        content_width = columns * frame_width
        if sheet_height < content_height or sheet_width < content_width:
            raise ValueError("CK Sheet to Frames received an invalid sheet layout.")

        expected_shape = (sheet_height, sheet_width, channels)
        if tuple(sheet.shape[1:]) != expected_shape:
            raise ValueError(
                "Sheet shape mismatch: layout expects "
                f"[1, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}] "
                f"but received {list(sheet.shape)}. Was the sheet resized or cropped?"
            )
        if frame_count > rows * columns:
            raise ValueError("CK Sheet to Frames layout has more frames than grid cells.")

        frames: list[torch.Tensor] = []
        for index in range(frame_count):
            row, column = divmod(index, columns)
            y = row * frame_height
            x = column * frame_width
            frames.append(sheet[:, y : y + frame_height, x : x + frame_width, :])
        return (torch.cat(frames, dim=0),)


# ---------------------------------------------------------------------------
# CKMergeEdits
# ---------------------------------------------------------------------------
class CKMergeEdits:
    """Combine any number of edited animation batches for a single pack."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_a": ("IMAGE",),
                "ck_frames_a": ("CK_FRAMES",),
                "frames_b": ("IMAGE",),
                "ck_frames_b": ("CK_FRAMES",),
            },
            "optional": _CKMergeOptionalInputs(),
        }

    RETURN_TYPES = ("CK_FRAMES", "IMAGE")
    RETURN_NAMES = ("ck_frames", "frames")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Merge any number of edited animation frame/CK_FRAMES pairs so they "
        "can be packed together."
    )

    def merge(
        self,
        frames_a=None,
        ck_frames_a=None,
        frames_b=None,
        ck_frames_b=None,
        **kwargs,
    ):
        if isinstance(frames_a, dict) and hasattr(ck_frames_a, "dim"):
            frames_a, ck_frames_a = ck_frames_a, frames_a
        if isinstance(frames_b, dict) and hasattr(ck_frames_b, "dim"):
            frames_b, ck_frames_b = ck_frames_b, frames_b

        pairs = _collect_merge_pairs(
            frames_a,
            ck_frames_a,
            frames_b,
            ck_frames_b,
            kwargs,
        )
        _validate_merge_pairs(pairs)

        max_w = max(
            max(int(desc["frame_size"][0]), int(frames.shape[2]))
            for _, frames, desc in pairs
        )
        max_h = max(
            max(int(desc["frame_size"][1]), int(frames.shape[1]))
            for _, frames, desc in pairs
        )

        frames = torch.cat(
            [_pad_batch(batch, max_w, max_h, 0.0) for _, batch, _ in pairs],
            0,
        )

        merged = dict(pairs[0][2])
        merged["frame_size"] = [max_w, max_h]
        merged["animation"] = "+".join(
            str(desc.get("animation", "")) for _, _, desc in pairs
            if desc.get("animation", "")
        )
        merged["sprites"] = [
            sprite
            for _, _, desc in pairs
            for sprite in list(desc["sprites"])
        ]
        collections = _merge_collections([desc for _, _, desc in pairs])
        merged["collections"] = collections
        merged["collection"] = collections[0] if len(collections) == 1 else None
        return (merged, frames)


class _CKMergeOptionalInputs(dict):
    """ComfyUI validator hook for dynamic frames_N / ck_frames_N inputs."""

    def __contains__(self, key):
        return self._input_type_for(key) is not None

    def __getitem__(self, key):
        input_type = self._input_type_for(key)
        if input_type is None:
            raise KeyError(key)
        return input_type

    def get(self, key, default=None):
        return self._input_type_for(key) or default

    @staticmethod
    def _input_type_for(key):
        if _has_numeric_suffix(key, "ck_frames_"):
            return ("CK_FRAMES",)
        if _has_numeric_suffix(key, "frames_") or _has_numeric_suffix(key, "images_"):
            return ("IMAGE",)
        return None


def _has_numeric_suffix(value: str, prefix: str) -> bool:
    return value.startswith(prefix) and value[len(prefix):].isdigit()


def _collect_merge_pairs(
    frames_a,
    ck_frames_a,
    frames_b,
    ck_frames_b,
    kwargs,
):
    pairs = []

    def add_pair(label, frames, ck_frames):
        if frames is None and ck_frames is None:
            return
        if frames is None or ck_frames is None:
            raise ValueError(
                f"CK Merge Edits input pair {label} must include both "
                "frames and ck_frames."
            )
        pairs.append((label, frames, ck_frames))

    add_pair("a", frames_a if frames_a is not None else kwargs.get("images_a"), ck_frames_a)
    add_pair("b", frames_b if frames_b is not None else kwargs.get("images_b"), ck_frames_b)

    indices = sorted(
        {
            int(name.rsplit("_", 1)[1])
            for name in kwargs
            if (
                _has_numeric_suffix(name, "ck_frames_")
                or _has_numeric_suffix(name, "frames_")
                or _has_numeric_suffix(name, "images_")
            )
        }
    )
    for index in indices:
        frames_key = f"frames_{index}"
        images_key = f"images_{index}"
        add_pair(
            str(index),
            kwargs.get(frames_key) if frames_key in kwargs else kwargs.get(images_key),
            kwargs.get(f"ck_frames_{index}"),
        )

    if len(pairs) < 2:
        raise ValueError("CK Merge Edits needs at least two frame/ck_frames pairs.")
    return pairs


def _validate_merge_pairs(pairs):
    channel_counts = {int(frames.shape[-1]) for _, frames, _ in pairs}
    if len(channel_counts) != 1:
        raise ValueError("CK Merge Edits cannot merge IMAGE batches with different channel counts.")

    for label, frames, desc in pairs:
        if frames.dim() != 4:
            raise ValueError(
                f"CK Merge Edits input pair {label} has frames shaped "
                f"{list(frames.shape)}; expected [frames, height, width, channels]."
            )
        if frames.shape[0] != len(desc["sprites"]):
            raise ValueError(
                f"CK Merge Edits input pair {label} describes {len(desc['sprites'])} "
                f"sprite(s) but received {frames.shape[0]} frame(s)."
            )

    for field in ("root_folders", "basepath"):
        values = [desc.get(field) for _, _, desc in pairs if desc.get(field) is not None]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"CK Merge Edits cannot merge descriptors from different {field}.")


def _merge_collections(descriptors):
    collections = []
    for desc in descriptors:
        names = desc.get("collections")
        if not names and desc.get("collection") is not None:
            names = [desc["collection"]]
        for name in names or []:
            if name not in collections:
                collections.append(name)
    return collections


def _pad_batch(batch: torch.Tensor, w: int, h: int, fill: float) -> torch.Tensor:
    b, ih, iw, c = batch.shape
    if ih == h and iw == w:
        return batch
    out = batch.new_full((b, h, w, c), fill)
    out[:, :ih, :iw, :] = batch
    return out


# ---------------------------------------------------------------------------
# CKPackAtlas
# ---------------------------------------------------------------------------
class CKPackAtlas:
    """Pack edited frames + unchanged sprites into a CustomKnight atlas PNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edited_frames": ("IMAGE",),
                "ck_frames": ("CK_FRAMES",),
            },
            "optional": {
                "save_to_output": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "CustomKnight/atlas"}),
                "external_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: also write <collection>.png here "
                        "(your CustomKnight skin folder).",
                    },
                ),
                "override_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "override_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("atlas", "saved_path")
    FUNCTION = "pack"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Drop the edited frames onto every unchanged sprite of the collection "
        "and pack them into one atlas PNG, ready for CustomKnight. When the "
        "frames span several collections (CK Animation Selector's 'animation "
        "range' mode), one atlas PNG is written per collection."
    )

    def pack(
        self,
        edited_frames,
        ck_frames,
        save_to_output=True,
        filename_prefix="CustomKnight/atlas",
        external_directory="",
        override_width=0,
        override_height=0,
    ):
        if isinstance(edited_frames, dict) and hasattr(ck_frames, "shape"):
            edited_frames, ck_frames = ck_frames, edited_frames

        project = SpriteProject.from_root_folders(ck_frames["root_folders"])
        sprites = [Sprite.from_dict(d) for d in ck_frames["sprites"]]

        if edited_frames.shape[0] != len(sprites):
            raise ValueError(
                f"Frame count mismatch: ck_frames describes {len(sprites)} "
                f"sprite(s) but {edited_frames.shape[0]} image(s) were supplied. "
                "Did an upstream node add/drop frames?"
            )

        # Group the edited frames by their collection, preserving first-seen
        # order. Single-animation edits have exactly one collection; an
        # "animation range" selection may carry several, each packed into its
        # own atlas PNG below.
        groups: dict[str, list[int]] = {}
        for i, sprite in enumerate(sprites):
            groups.setdefault(sprite.collection, []).append(i)

        # An explicit override only makes sense for a single atlas; with several
        # collections each keeps its own natural (power-of-two) size.
        size = None
        if override_width > 0 and override_height > 0 and len(groups) == 1:
            size = (override_width, override_height)

        ext_dir = external_directory.strip().strip('"')
        if ext_dir:
            os.makedirs(ext_dir, exist_ok=True)

        fallback = {s.path: s for s in project.sprites}

        atlas_tensors: list[torch.Tensor] = []
        saved_paths: list[str] = []
        ui_images: list[dict] = []

        for collection, indices in groups.items():
            # Build replacements: each edited frame -> upright (w, h) RGBA content.
            replacements: dict[str, Image.Image] = {}
            for i in indices:
                sprite = sprites[i]
                content = tensor_to_pil_rgba(edited_frames[i])
                content = content.crop((0, 0, sprite.w, sprite.h))
                # If the edit dropped alpha, keep the original sprite's transparency.
                if (
                    edited_frames.shape[-1] < 4
                    and sprite.path in fallback
                ):
                    orig = project.crop_content(fallback[sprite.path])
                    content.putalpha(orig.getchannel("A"))
                replacements[sprite.path] = content

            result = pack_collection(project, collection, replacements, size)
            # Keep alpha: the atlas is transparent, and the output IMAGE must
            # match the saved PNG shown in the node's thumbnail.
            atlas_tensors.append(pil_rgba_to_tensor(result.image))

            # Always produce a thumbnail for the node. When saving to output we
            # persist it there; otherwise we write a throwaway copy to ComfyUI's
            # temp folder purely so the in-node preview shows up regardless of
            # the save_to_output toggle.
            if save_to_output:
                out_path, ui = self._save_image(
                    result.image, f"{filename_prefix}_{collection}", "output"
                )
            else:
                out_path, ui = self._save_image(
                    result.image, self._temp_prefix(collection), "temp"
                )
            ui_images.extend(ui.get("images", []))

            ext_path = ""
            if ext_dir:
                ext_path = os.path.join(ext_dir, f"{collection}.png")
                result.image.save(ext_path)

            chosen = out_path or ext_path
            if chosen:
                saved_paths.append(chosen)

        # Stack the per-collection atlases into one IMAGE batch (transparent-
        # padded to a common size) so a single output wire previews them all.
        batch_w = max(t.shape[2] for t in atlas_tensors)
        batch_h = max(t.shape[1] for t in atlas_tensors)
        atlas_batch = torch.cat(
            [_pad_batch(t, batch_w, batch_h, 0.0) for t in atlas_tensors], 0
        )
        saved_path = "\n".join(saved_paths)

        return {"ui": {"images": ui_images}, "result": (atlas_batch, saved_path)}

    @staticmethod
    def _temp_prefix(collection: str) -> str:
        """A per-run prefix so temp previews don't clash between executions."""
        rand = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        return f"ckatlas_preview_{collection}_{rand}"

    @staticmethod
    def _save_image(image: Image.Image, prefix: str, type_: str):
        """Save ``image`` under the output (``type_="output"``) or temp
        (``type_="temp"``) directory and return ``(path, ui_dict)``.

        The ``ui_dict`` is what makes ComfyUI render the thumbnail on the node.
        Returns ``("", {})`` when ComfyUI's ``folder_paths`` isn't importable
        (e.g. unit tests outside a ComfyUI host).
        """
        if folder_paths is None:
            return "", {}
        out_dir = (
            folder_paths.get_output_directory()
            if type_ == "output"
            else folder_paths.get_temp_directory()
        )
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, out_dir, image.width, image.height
        )
        file = f"{filename}_{counter:05}_.png"
        path = os.path.join(full_folder, file)
        image.save(path)
        return path, {
            "images": [{"filename": file, "subfolder": subfolder, "type": type_}]
        }


# ---------------------------------------------------------------------------
# CK_FRAMES JSON persistence
# ---------------------------------------------------------------------------
class CKSaveFramesDescriptor:
    """Save a CK_FRAMES descriptor as JSON and pass it through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ck_frames": ("CK_FRAMES",),
                "path": (
                    "STRING",
                    {
                        "default": "ck_frames.json",
                        "tooltip": "JSON file path to write, e.g. C:/skin/walk.ck_frames.json.",
                    },
                ),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CK_FRAMES", "STRING")
    RETURN_NAMES = ("ck_frames", "saved_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Save a CK_FRAMES layout descriptor to a JSON file."

    def save(self, ck_frames, path: str, overwrite=True):
        _validate_ck_frames_descriptor(ck_frames)
        resolved = _resolve_descriptor_path(path)
        if os.path.exists(resolved) and not overwrite:
            raise FileExistsError(f"CK_FRAMES descriptor already exists: {resolved}")
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as fh:
            json.dump(ck_frames, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return (ck_frames, resolved)


class CKLoadFramesDescriptor:
    """Load a CK_FRAMES descriptor from JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "ck_frames.json",
                        "tooltip": "JSON file path previously written by CK Save Frames Descriptor.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CK_FRAMES", "STRING")
    RETURN_NAMES = ("ck_frames", "loaded_path")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a CK_FRAMES layout descriptor from a JSON file."

    def load(self, path: str):
        resolved = _resolve_descriptor_path(path)
        with open(resolved, "r", encoding="utf-8") as fh:
            ck_frames = json.load(fh)
        _validate_ck_frames_descriptor(ck_frames)
        return (ck_frames, resolved)


def _resolve_descriptor_path(path: str) -> str:
    path = str(path or "").strip().strip('"')
    if not path:
        raise ValueError("CK_FRAMES descriptor path cannot be empty.")
    if os.path.isdir(path):
        path = os.path.join(path, "ck_frames.json")
    if not os.path.splitext(path)[1]:
        path = f"{path}.json"
    return os.path.abspath(os.path.expanduser(path))


def _validate_ck_frames_descriptor(value):
    if not isinstance(value, dict):
        raise ValueError("CK_FRAMES descriptor must be a JSON object.")
    missing = [
        key
        for key in ("root_folders", "frame_size", "sprites")
        if key not in value
    ]
    if missing:
        raise ValueError(
            "CK_FRAMES descriptor is missing required field(s): "
            + ", ".join(missing)
        )
    if not isinstance(value["sprites"], list):
        raise ValueError("CK_FRAMES descriptor field 'sprites' must be a list.")


# ---------------------------------------------------------------------------
# CKLoadProjectInfo (inspection helper)
# ---------------------------------------------------------------------------
class CKLoadProjectInfo:
    """List the collections and animations found in the given Root Folders."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_folders": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("basepath", "collections", "summary")
    FUNCTION = "info"
    CATEGORY = CATEGORY
    DESCRIPTION = "Inspect a CustomKnight dump: base path, collections, animations."

    def info(self, root_folders: str):
        project = _project_from(root_folders)
        collections = project.collections()
        lines = [f"basepath: {project.basepath}", f"sprites: {len(project.sprites)}", ""]
        for col in collections:
            w, h = atlas_size_for(project.sprites_in_collection(col))
            anims = project.animations(col)
            lines.append(f"[{col}]  atlas {w}x{h}  ({len(anims)} animations)")
            for anim in anims:
                count = len(project.sprites_in_animation(anim, col))
                lines.append(f"    {anim}  ({count} frames)")
            lines.append("")
        summary = "\n".join(lines)
        return (project.basepath, json.dumps(collections), summary)


NODE_CLASS_MAPPINGS = {
    "CKAnimationSelector": CKAnimationSelector,
    "CKFramesToSheet": CKFramesToSheet,
    "CKSheetToFrames": CKSheetToFrames,
    "CKPackAtlas": CKPackAtlas,
    "CKMergeEdits": CKMergeEdits,
    "CKSaveFramesDescriptor": CKSaveFramesDescriptor,
    "CKLoadFramesDescriptor": CKLoadFramesDescriptor,
    "CKLoadProjectInfo": CKLoadProjectInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKAnimationSelector": "CK Animation Selector",
    "CKFramesToSheet": "CK Frames to Sheet",
    "CKSheetToFrames": "CK Sheet to Frames",
    "CKPackAtlas": "CK Pack Atlas",
    "CKMergeEdits": "CK Merge Edits",
    "CKSaveFramesDescriptor": "CK Save Frames Descriptor",
    "CKLoadFramesDescriptor": "CK Load Frames Descriptor",
    "CKLoadProjectInfo": "CK Project Info",
}

# CK Video Combine depends on ComfyUI's native VIDEO type (comfy_api). Register it
# only when that's available so the rest of the nodes still load on older hosts.
try:
    import comfy_api.input_impl  # noqa: F401

    from .video_combine import CKVideoCombine

    NODE_CLASS_MAPPINGS["CKVideoCombine"] = CKVideoCombine
    NODE_DISPLAY_NAME_MAPPINGS["CKVideoCombine"] = "CK Video Combine"
except Exception as exc:  # pragma: no cover - host without the VIDEO type
    import warnings

    warnings.warn(f"CK Video Combine not loaded ({exc}).", stacklevel=2)
