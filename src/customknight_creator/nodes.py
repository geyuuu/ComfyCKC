"""ComfyUI nodes for CustomKnight Creator.

Two core nodes mirror the original desktop tool:

``CKAnimationSelector``
    Input is one or more *Root Folders*. Pick a collection (atlas) and an
    animation - exactly like the original - and the node outputs that
    animation's frames as a PNG image sequence (an ``IMAGE`` batch) plus a
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
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB ``[1, H, W, 3]`` float tensor in ``[0, 1]``."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def pil_rgba_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGBA ``[1, H, W, 4]`` float tensor in ``[0, 1]``.

    Keeps the alpha channel so the output IMAGE renders with transparency in a
    downstream Preview/Save node, matching the saved atlas PNG. Dropping alpha
    here (``convert("RGB")``) would expose the arbitrary RGB that sprite atlases
    leave in fully-transparent texels (white boxes / colour noise).
    """
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def pil_alpha_to_mask(img: Image.Image) -> torch.Tensor:
    """Alpha channel as a ``[1, H, W]`` mask (1 = opaque)."""
    if "A" in img.getbands():
        alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
    else:
        alpha = np.ones((img.size[1], img.size[0]), dtype=np.float32)
    return torch.from_numpy(alpha)[None, ...]


def tensor_to_pil_rgba(image: torch.Tensor, mask: torch.Tensor | None = None) -> Image.Image:
    """Convert one frame (``[H, W, C]`` or ``[1, H, W, C]``) to an RGBA PIL image.

    The optional ``mask`` supplies the alpha channel; otherwise the image's own
    4th channel is used, falling back to fully opaque.
    """
    if image.dim() == 4:
        image = image[0]
    rgb = (image[..., :3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    h, w = rgb.shape[:2]

    if mask is not None:
        if mask.dim() == 3:
            mask = mask[0]
        alpha = (mask.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if alpha.shape != (h, w):
            alpha = np.asarray(
                Image.fromarray(alpha).resize((w, h), Image.NEAREST), dtype=np.uint8
            )
    elif image.shape[-1] >= 4:
        alpha = (image[..., 3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        alpha = np.full((h, w), 255, dtype=np.uint8)

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, "RGBA")


def stack_frames(frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Pad ``(w, h)`` RGBA frames to a common size and stack into batches.

    Frames are anchored top-left on a transparent canvas sized to the largest
    frame, so a single ``IMAGE`` / ``MASK`` batch can carry differently sized
    sprites. Returns ``(images, masks, max_w, max_h)``.
    """
    max_w = max((f.size[0] for f in frames), default=1)
    max_h = max((f.size[1] for f in frames), default=1)

    imgs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for frame in frames:
        canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        canvas.paste(frame, (0, 0))
        imgs.append(pil_to_tensor(canvas))
        masks.append(pil_alpha_to_mask(canvas))

    return torch.cat(imgs, 0), torch.cat(masks, 0), max_w, max_h


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

    RETURN_TYPES = ("IMAGE", "MASK", "CK_FRAMES")
    RETURN_NAMES = ("frames", "alpha", "ck_frames")
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Pick a collection and animation from CustomKnight Root Folders and "
        "output the frames as a PNG image sequence (plus alpha + a CK_FRAMES "
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
        images, masks, max_w, max_h = stack_frames(frames)

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
        return (images, masks, ck_frames)


# ---------------------------------------------------------------------------
# CKMergeEdits
# ---------------------------------------------------------------------------
class CKMergeEdits:
    """Combine two animations' edits (same collection) for a single pack."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ck_frames_a": ("CK_FRAMES",),
                "images_a": ("IMAGE",),
                "ck_frames_b": ("CK_FRAMES",),
                "images_b": ("IMAGE",),
            },
            "optional": {
                "alpha_a": ("MASK",),
                "alpha_b": ("MASK",),
            },
        }

    RETURN_TYPES = ("CK_FRAMES", "IMAGE", "MASK")
    RETURN_NAMES = ("ck_frames", "frames", "alpha")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Merge two edited animations from the same collection so they can be "
        "packed together. Chain multiple of these for more animations."
    )

    def merge(self, ck_frames_a, images_a, ck_frames_b, images_b, alpha_a=None, alpha_b=None):
        if ck_frames_a["collection"] != ck_frames_b["collection"]:
            raise ValueError(
                "Cannot merge edits from different collections: "
                f'"{ck_frames_a["collection"]}" vs "{ck_frames_b["collection"]}".'
            )

        size_a = ck_frames_a["frame_size"]
        size_b = ck_frames_b["frame_size"]
        max_w = max(size_a[0], size_b[0], images_a.shape[2], images_b.shape[2])
        max_h = max(size_a[1], size_b[1], images_a.shape[1], images_b.shape[1])

        images = _pad_batch(images_a, max_w, max_h, 0.0)
        images = torch.cat([images, _pad_batch(images_b, max_w, max_h, 0.0)], 0)

        masks = torch.cat(
            [
                _pad_mask(_mask_for(alpha_a, images_a), max_w, max_h),
                _pad_mask(_mask_for(alpha_b, images_b), max_w, max_h),
            ],
            0,
        )

        merged = dict(ck_frames_a)
        merged["frame_size"] = [max_w, max_h]
        merged["animation"] = f'{ck_frames_a.get("animation", "")}+{ck_frames_b.get("animation", "")}'
        merged["sprites"] = list(ck_frames_a["sprites"]) + list(ck_frames_b["sprites"])
        return (merged, images, masks)


def _mask_for(mask, images):
    if mask is not None:
        return mask
    return torch.ones((images.shape[0], images.shape[1], images.shape[2]), dtype=torch.float32)


def _pad_batch(batch: torch.Tensor, w: int, h: int, fill: float) -> torch.Tensor:
    b, ih, iw, c = batch.shape
    if ih == h and iw == w:
        return batch
    out = torch.full((b, h, w, c), fill, dtype=batch.dtype)
    out[:, :ih, :iw, :] = batch
    return out


def _pad_mask(mask: torch.Tensor, w: int, h: int) -> torch.Tensor:
    b, ih, iw = mask.shape
    if ih == h and iw == w:
        return mask
    out = torch.zeros((b, h, w), dtype=mask.dtype)
    out[:, :ih, :iw] = mask
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
                "ck_frames": ("CK_FRAMES",),
                "edited_frames": ("IMAGE",),
            },
            "optional": {
                "edited_alpha": ("MASK",),
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
        ck_frames,
        edited_frames,
        edited_alpha=None,
        save_to_output=True,
        filename_prefix="CustomKnight/atlas",
        external_directory="",
        override_width=0,
        override_height=0,
    ):
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
                mask = edited_alpha[i] if edited_alpha is not None else None
                content = tensor_to_pil_rgba(edited_frames[i], mask)
                content = content.crop((0, 0, sprite.w, sprite.h))
                # If the edit dropped alpha, keep the original sprite's transparency.
                if (
                    edited_alpha is None
                    and edited_frames.shape[-1] < 4
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
    "CKPackAtlas": CKPackAtlas,
    "CKMergeEdits": CKMergeEdits,
    "CKLoadProjectInfo": CKLoadProjectInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKAnimationSelector": "CK Animation Selector",
    "CKPackAtlas": "CK Pack Atlas",
    "CKMergeEdits": "CK Merge Edits",
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
