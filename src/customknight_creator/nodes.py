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
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
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


def _temp_preview_directory() -> str:
    if folder_paths is not None:
        return folder_paths.get_temp_directory()
    return tempfile.gettempdir()


def _write_temp_preview_frames(frames: torch.Tensor, prefix: str) -> list[dict]:
    _validate_alignment_image_batch("frames", frames)
    temp_dir = _temp_preview_directory()
    os.makedirs(temp_dir, exist_ok=True)

    rand = random.randint(0, 0xFFFFFFFF)
    results = []
    for index, frame in enumerate(frames):
        filename = f"{prefix}_{rand:08x}_{index:05d}.png"
        tensor_to_pil_rgba(frame).save(os.path.join(temp_dir, filename))
        results.append(
            {
                "filename": filename,
                "subfolder": "",
                "type": "temp",
                "name": f"frame {index + 1}",
            }
        )
    return results


def _validate_alignment_image_batch(name: str, image: torch.Tensor) -> None:
    if not hasattr(image, "dim") or image.dim() != 4:
        raise ValueError(
            f"{name} must be an IMAGE batch shaped [batch, height, width, channels]."
        )
    if image.shape[0] < 1 or min(image.shape[1:]) < 1:
        raise ValueError(f"{name} has an invalid empty shape.")


def _paired_image_batches(
    modified: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_alignment_image_batch("modified", modified)
    _validate_alignment_image_batch("reference", reference)

    modified_count = modified.shape[0]
    reference_count = reference.shape[0]
    if modified_count == reference_count:
        return modified, reference
    if modified_count == 1:
        return modified.expand(reference_count, -1, -1, -1), reference
    if reference_count == 1:
        return modified, reference.expand(modified_count, -1, -1, -1)
    raise ValueError(
        "modified and reference batches must have the same batch size, or one "
        "of them must contain exactly one image."
    )


def _normalised_axis(
    length: int, source_length: int, stretch: float, device, dtype
) -> torch.Tensor:
    if length == 1 or source_length == 1:
        return torch.zeros((length,), device=device, dtype=dtype)
    base_scale = (source_length - 1) / (length - 1)
    source_position = torch.arange(length, device=device, dtype=dtype)
    source_position = source_position * base_scale * stretch
    return source_position / (source_length - 1) * 2.0 - 1.0


def _resample_top_left(
    image: torch.Tensor,
    height: int,
    width: int,
    stretch_x: float,
    stretch_y: float,
    resampling: str = "bilinear",
    clamp_output: bool = True,
) -> torch.Tensor:
    if stretch_x <= 0 or stretch_y <= 0:
        raise ValueError("stretch_x and stretch_y must be greater than 0.")
    if resampling not in ("nearest", "bilinear", "bicubic"):
        raise ValueError("resampling must be one of: nearest, bilinear, bicubic.")

    _validate_alignment_image_batch("image", image)
    source = image.permute(0, 3, 1, 2).contiguous().to(torch.float32)
    batch, _channels, source_height, source_width = source.shape
    grid_x = _normalised_axis(
        width, source_width, float(stretch_x), source.device, source.dtype
    )
    grid_y = _normalised_axis(
        height, source_height, float(stretch_y), source.device, source.dtype
    )
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    sampled = F.grid_sample(
        source,
        grid,
        mode=resampling,
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.permute(0, 2, 3, 1)
    if clamp_output:
        sampled = sampled.clamp(0, 1)
    return sampled


def _single_channel_intensity(frame: torch.Tensor, prefer_alpha: bool) -> torch.Tensor:
    frame = frame.to(torch.float32).clamp(0, 1)
    if prefer_alpha and frame.shape[-1] >= 4:
        alpha = frame[..., 3]
        if alpha.max() > 0.001 and alpha.max() - alpha.min() > 0.01:
            return alpha

    if frame.shape[-1] >= 3:
        return (
            frame[..., 0] * 0.299
            + frame[..., 1] * 0.587
            + frame[..., 2] * 0.114
        )
    return frame[..., 0]


def _edge_feature(frame: torch.Tensor) -> torch.Tensor:
    base = _single_channel_intensity(frame, prefer_alpha=True)
    dx = torch.zeros_like(base)
    dy = torch.zeros_like(base)
    dx[:, 1:] = (base[:, 1:] - base[:, :-1]).abs()
    dy[1:, :] = (base[1:, :] - base[:-1, :]).abs()
    feature = base * 0.25 + dx + dy
    minimum = feature.min()
    maximum = feature.max()
    if maximum > minimum:
        feature = (feature - minimum) / (maximum - minimum)
    return feature


def _match_feature(frame: torch.Tensor, max_size: int = 256) -> torch.Tensor:
    feature = _edge_feature(frame)
    height, width = feature.shape
    scale = min(1.0, max_size / max(height, width))
    target_height = max(2, int(round(height * scale)))
    target_width = max(2, int(round(width * scale)))
    if (target_height, target_width) != (height, width):
        feature = F.interpolate(
            feature[None, None, :, :],
            size=(target_height, target_width),
            mode="area",
        )[0, 0]
    feature = feature - feature.mean()
    std = feature.std()
    if std > 1e-6:
        feature = feature / std
    return feature


def _score_stretch(
    modified_feature: torch.Tensor,
    reference_feature: torch.Tensor,
    stretch_x: float,
    stretch_y: float,
) -> float:
    sampled = _resample_top_left(
        modified_feature[None, :, :, None],
        reference_feature.shape[0],
        reference_feature.shape[1],
        stretch_x,
        stretch_y,
        "bilinear",
        clamp_output=False,
    )[0, :, :, 0]
    return float(torch.mean((sampled - reference_feature) ** 2).item())


def _odd_step_count(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _estimate_top_left_stretch(
    modified: torch.Tensor,
    reference: torch.Tensor,
    search_percent: float,
    coarse_steps: int,
    fine_steps: int,
) -> tuple[float, float, float]:
    search = max(0.0, float(search_percent)) / 100.0
    min_scale = max(0.01, 1.0 - search)
    max_scale = 1.0 + search
    if search == 0:
        score = _score_stretch(
            _match_feature(modified),
            _match_feature(reference),
            1.0,
            1.0,
        )
        return 1.0, 1.0, score

    modified_feature = _match_feature(modified)
    reference_feature = _match_feature(reference)

    best_x = 1.0
    best_y = 1.0
    best_score = math.inf
    coarse_steps = _odd_step_count(coarse_steps)
    for x in torch.linspace(min_scale, max_scale, coarse_steps).tolist():
        for y in torch.linspace(min_scale, max_scale, coarse_steps).tolist():
            score = _score_stretch(modified_feature, reference_feature, x, y)
            if score < best_score:
                best_x = float(x)
                best_y = float(y)
                best_score = score

    coarse_step = (max_scale - min_scale) / max(1, coarse_steps - 1)
    fine_half_span = coarse_step * 2.0
    fine_steps = _odd_step_count(fine_steps)
    fine_min_x = max(min_scale, best_x - fine_half_span)
    fine_max_x = min(max_scale, best_x + fine_half_span)
    fine_min_y = max(min_scale, best_y - fine_half_span)
    fine_max_y = min(max_scale, best_y + fine_half_span)
    for x in torch.linspace(fine_min_x, fine_max_x, fine_steps).tolist():
        for y in torch.linspace(fine_min_y, fine_max_y, fine_steps).tolist():
            score = _score_stretch(modified_feature, reference_feature, x, y)
            if score < best_score:
                best_x = float(x)
                best_y = float(y)
                best_score = score

    return best_x, best_y, best_score


def _rgb_for_preview(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.to(torch.float32).clamp(0, 1)
    if frame.shape[-1] >= 3:
        rgb = frame[..., :3]
    else:
        rgb = frame[..., :1].expand(-1, -1, 3)
    if frame.shape[-1] >= 4:
        rgb = rgb * frame[..., 3:4]
    return rgb


def _overlay_intensity(frame: torch.Tensor) -> torch.Tensor:
    intensity = _single_channel_intensity(frame, prefer_alpha=True)
    maximum = intensity.max()
    if maximum > 1e-6:
        intensity = intensity / maximum
    return intensity.clamp(0, 1)


def _overlay_preview(
    adjusted: torch.Tensor,
    reference: torch.Tensor,
    preview_mode: str,
    preview_opacity: float,
) -> torch.Tensor:
    opacity = max(0.0, min(1.0, float(preview_opacity)))
    previews: list[torch.Tensor] = []
    for index in range(adjusted.shape[0]):
        adj = adjusted[index]
        ref = reference[index]
        if preview_mode == "blend":
            rgb = _rgb_for_preview(ref) * (1.0 - opacity) + _rgb_for_preview(adj) * opacity
        elif preview_mode == "difference":
            rgb = (_rgb_for_preview(adj) - _rgb_for_preview(ref)).abs()
        else:
            ref_i = _overlay_intensity(ref)
            adj_i = _overlay_intensity(adj)
            rgb = torch.stack(
                (
                    ref_i,
                    adj_i,
                    (ref_i - adj_i).abs() * 0.35,
                ),
                dim=-1,
            )
        alpha = torch.ones((*rgb.shape[:2], 1), dtype=rgb.dtype, device=rgb.device)
        previews.append(torch.cat((rgb.clamp(0, 1), alpha), dim=-1)[None, ...])
    return torch.cat(previews, dim=0)


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


class CKFramesPreview:
    """Preview an IMAGE batch with the same animated canvas as the selector."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "preview"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Preview a frames IMAGE batch using the same animated in-node preview "
        "style as CK Animation Selector. The frames are passed through "
        "unchanged."
    )

    def preview(self, frames):
        images = _write_temp_preview_frames(frames, "ckframes_preview")
        return {"ui": {"images": images, "animated": (True,)}, "result": (frames,)}


# ---------------------------------------------------------------------------
# Reversible frame-sheet helpers
# ---------------------------------------------------------------------------
_SHEET_ALIGNMENT = 16
_SHEET_CONTENT_PADDING = 16
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
        padding_left = _SHEET_CONTENT_PADDING
        padding_top = _SHEET_CONTENT_PADDING
        content_height = padding_top + rows * frame_height
        content_width = padding_left + columns * frame_width
        sheet_width, sheet_height = _frame_sheet_size(content_width, content_height)

        # new_zeros retains the input tensor's dtype and device. Direct slice
        # assignment performs no resize, colour conversion, or quantisation.
        # The frame grid starts after a fixed left/top transparent gutter, and
        # any remaining alignment padding stays outside all frame cells.
        sheet = frames.new_zeros((1, sheet_height, sheet_width, channels))
        for index in range(frame_count):
            row, column = divmod(index, columns)
            y = padding_top + row * frame_height
            x = padding_left + column * frame_width
            sheet[0, y : y + frame_height, x : x + frame_width, :] = frames[index]

        layout = {
            "version": 4,
            "frame_count": frame_count,
            "frame_height": frame_height,
            "frame_width": frame_width,
            "channels": channels,
            "columns": columns,
            "rows": rows,
            "padding_left": padding_left,
            "padding_top": padding_top,
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
                "resize_method": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": (
                            "How to resize the input sheet if its width/height "
                            "does not match sheet_layout."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Split an image made by CK Frames to Sheet back into its original "
        "ordered IMAGE batch. If the sheet was resized, it is resized back to "
        "the dimensions recorded in sheet_layout before slicing."
    )

    def split(
        self,
        sheet: torch.Tensor,
        sheet_layout,
        resize_method: str = "bilinear",
    ):
        if sheet.dim() != 4 or sheet.shape[0] != 1:
            raise ValueError("CK Sheet to Frames expects exactly one sheet image.")
        if resize_method not in ("bilinear", "bicubic", "nearest"):
            raise ValueError("resize_method must be one of: bilinear, bicubic, nearest.")
        if (
            not isinstance(sheet_layout, dict)
            or sheet_layout.get("version") not in (1, 2, 3, 4)
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
        if sheet_layout["version"] >= 4:
            try:
                padding_left = int(sheet_layout["padding_left"])
                padding_top = int(sheet_layout["padding_top"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "CK Sheet to Frames received an invalid sheet layout."
                ) from exc
            if padding_left < 0 or padding_top < 0:
                raise ValueError("CK Sheet to Frames received an invalid sheet layout.")
        else:
            padding_left = 0
            padding_top = 0

        if sheet_layout["version"] in (2, 3, 4):
            try:
                sheet_height = int(sheet_layout["sheet_height"])
                sheet_width = int(sheet_layout["sheet_width"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "CK Sheet to Frames received an invalid sheet layout."
                ) from exc
            content_height = padding_top + rows * frame_height
            content_width = padding_left + columns * frame_width
            if sheet_layout["version"] == 2:
                expected_height = ((rows * frame_height + 15) // 16) * 16
                expected_width = ((columns * frame_width + 15) // 16) * 16
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

        content_height = padding_top + rows * frame_height
        content_width = padding_left + columns * frame_width
        if sheet_height < content_height or sheet_width < content_width:
            raise ValueError("CK Sheet to Frames received an invalid sheet layout.")

        expected_shape = (sheet_height, sheet_width, channels)
        if int(sheet.shape[3]) != channels:
            raise ValueError(
                "Sheet channel mismatch: layout expects "
                f"{channels} channels but received {int(sheet.shape[3])}."
            )
        if tuple(sheet.shape[1:3]) != (sheet_height, sheet_width):
            source = sheet.permute(0, 3, 1, 2).contiguous().to(torch.float32)
            kwargs = {}
            if resize_method != "nearest":
                kwargs["align_corners"] = False
            sheet = F.interpolate(
                source,
                size=(sheet_height, sheet_width),
                mode=resize_method,
                **kwargs,
            ).permute(0, 2, 3, 1).clamp(0, 1)
        if frame_count > rows * columns:
            raise ValueError("CK Sheet to Frames layout has more frames than grid cells.")

        frames: list[torch.Tensor] = []
        for index in range(frame_count):
            row, column = divmod(index, columns)
            y = padding_top + row * frame_height
            x = padding_left + column * frame_width
            frames.append(sheet[:, y : y + frame_height, x : x + frame_width, :])
        return (torch.cat(frames, dim=0),)


# ---------------------------------------------------------------------------
# CK sheet stretch alignment helpers
# ---------------------------------------------------------------------------
class CKAutoAlignSheetStretch:
    """Automatically undo a small top-left anchored sheet stretch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modified": ("IMAGE",),
                "reference": ("IMAGE",),
                "search_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": (
                            "Maximum stretch correction to search around 1.0, "
                            "as a percent. 8 searches 0.92-1.08."
                        ),
                    },
                ),
                "coarse_steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 3,
                        "max": 101,
                        "tooltip": (
                            "Number of coarse samples per axis; even values "
                            "are rounded up."
                        ),
                    },
                ),
                "fine_steps": (
                    "INT",
                    {
                        "default": 11,
                        "min": 3,
                        "max": 101,
                        "tooltip": "Number of fine samples per axis around the best coarse match.",
                    },
                ),
                "resampling": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation used for the corrected output image.",
                    },
                ),
                "preview_mode": (
                    ["red/green overlap", "blend", "difference"],
                    {
                        "default": "red/green overlap",
                        "tooltip": (
                            "Preview style. Red/green shows reference in red, "
                            "corrected image in green, and yellow where they overlap."
                        ),
                    },
                ),
                "preview_opacity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Opacity of the corrected image in blend preview mode.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "aligned",
        "overlay_preview",
        "stretch_x",
        "stretch_y",
        "match_error",
    )
    FUNCTION = "align"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Correct a modified spritesheet that has a small stretch relative to a "
        "reference sheet, using the top-left corner as the fixed origin. The "
        "aligned output always uses the reference resolution."
    )

    def align(
        self,
        modified: torch.Tensor,
        reference: torch.Tensor,
        search_percent: float = 8.0,
        coarse_steps: int = 25,
        fine_steps: int = 11,
        resampling: str = "bilinear",
        preview_mode: str = "red/green overlap",
        preview_opacity: float = 0.5,
    ):
        modified, reference = _paired_image_batches(modified, reference)
        output_height = int(reference.shape[1])
        output_width = int(reference.shape[2])

        aligned_frames: list[torch.Tensor] = []
        scale_xs: list[float] = []
        scale_ys: list[float] = []
        scores: list[float] = []
        for index in range(modified.shape[0]):
            stretch_x, stretch_y, score = _estimate_top_left_stretch(
                modified[index],
                reference[index],
                search_percent,
                coarse_steps,
                fine_steps,
            )
            aligned_frames.append(
                _resample_top_left(
                    modified[index : index + 1],
                    output_height,
                    output_width,
                    stretch_x,
                    stretch_y,
                    resampling,
                    clamp_output=True,
                )
            )
            scale_xs.append(stretch_x)
            scale_ys.append(stretch_y)
            scores.append(score)

        aligned = torch.cat(aligned_frames, dim=0)
        preview = _overlay_preview(aligned, reference, preview_mode, preview_opacity)
        return (
            aligned,
            preview,
            float(sum(scale_xs) / len(scale_xs)),
            float(sum(scale_ys) / len(scale_ys)),
            float(sum(scores) / len(scores)),
        )


class CKManualAlignSheetStretch:
    """Manually adjust top-left anchored sheet stretch and preview overlap."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modified": ("IMAGE",),
                "reference": ("IMAGE",),
                "stretch_x": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 1.5,
                        "step": 0.0005,
                        "tooltip": (
                            "Horizontal correction around the top-left origin. "
                            "Values above 1 pull pixels from farther right, "
                            "shrinking stretched content."
                        ),
                    },
                ),
                "stretch_y": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 1.5,
                        "step": 0.0005,
                        "tooltip": (
                            "Vertical correction around the top-left origin. "
                            "Values above 1 pull pixels from farther down, "
                            "shrinking stretched content."
                        ),
                    },
                ),
                "resampling": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation used for the corrected output image.",
                    },
                ),
                "preview_mode": (
                    ["red/green overlap", "blend", "difference"],
                    {
                        "default": "red/green overlap",
                        "tooltip": (
                            "Preview style. Red/green shows reference in red, "
                            "corrected image in green, and yellow where they overlap."
                        ),
                    },
                ),
                "preview_opacity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Opacity of the corrected image in blend preview mode.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned", "overlay_preview")
    FUNCTION = "adjust"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Manually correct a top-left anchored sheet stretch. Use the preview "
        "output to compare the corrected image against the reference."
    )

    def adjust(
        self,
        modified: torch.Tensor,
        reference: torch.Tensor,
        stretch_x: float = 1.0,
        stretch_y: float = 1.0,
        resampling: str = "bilinear",
        preview_mode: str = "red/green overlap",
        preview_opacity: float = 0.5,
    ):
        modified, reference = _paired_image_batches(modified, reference)
        aligned = _resample_top_left(
            modified,
            int(reference.shape[1]),
            int(reference.shape[2]),
            stretch_x,
            stretch_y,
            resampling,
            clamp_output=True,
        )
        preview = _overlay_preview(aligned, reference, preview_mode, preview_opacity)
        return (aligned, preview)


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


class CKSaveFrames:
    """Save IMAGE frames and, optionally, their CK_FRAMES descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Parent folder. With ck_frames, an animation-named subfolder is created.",
                    },
                ),
            },
            "optional": {
                "ck_frames": ("CK_FRAMES",),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "CK_FRAMES", "STRING")
    RETURN_NAMES = ("frames", "ck_frames", "saved_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Save frames and optional ck_frames together under one folder."

    def save(self, frames, path: str, ck_frames=None, overwrite=True):
        _validate_image_batch(frames)
        if ck_frames is not None:
            _validate_frames_for_descriptor(frames, ck_frames)

        base = _resolve_directory_path(path, "CK Save Frames path cannot be empty.")
        folder = os.path.join(base, _animation_folder_name(ck_frames))
        if os.path.exists(folder) and not overwrite:
            raise FileExistsError(f"CK frames folder already exists: {folder}")
        os.makedirs(folder, exist_ok=True)

        frame_names = _saved_frame_file_names(frames.shape[0], ck_frames)
        for index, frame_name in enumerate(frame_names):
            tensor_to_pil_rgba(frames[index]).save(
                os.path.join(folder, frame_name)
            )

        if ck_frames is not None:
            with open(os.path.join(folder, "ck_frames.json"), "w", encoding="utf-8") as fh:
                json.dump(ck_frames, fh, ensure_ascii=False, indent=2)
                fh.write("\n")

        return (frames, ck_frames, folder)


class CKLoadFrames:
    """Load IMAGE frames and an optional CK_FRAMES descriptor from a folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder written by CK Save Frames, or its ck_frames.json file.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CK_FRAMES", "STRING")
    RETURN_NAMES = ("frames", "ck_frames", "loaded_path")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load frames and optional ck_frames saved by CK Save Frames."

    def load(self, path: str):
        folder = _resolve_saved_frames_directory(path)
        descriptor_path = os.path.join(folder, "ck_frames.json")
        ck_frames = None
        if os.path.isfile(descriptor_path):
            with open(descriptor_path, "r", encoding="utf-8") as fh:
                ck_frames = json.load(fh)
            _validate_ck_frames_descriptor(ck_frames)

        frame_paths = _saved_frame_paths(folder, ck_frames)
        frames = [Image.open(frame_path).convert("RGBA") for frame_path in frame_paths]
        images, _, _ = stack_frames(frames)
        if ck_frames is not None:
            _validate_frames_for_descriptor(images, ck_frames)
        return (images, ck_frames, folder)


def _validate_image_batch(frames):
    if not hasattr(frames, "dim") or frames.dim() != 4:
        raise ValueError("CK Save Frames expects an IMAGE batch shaped [frames, height, width, channels].")
    if frames.shape[0] <= 0:
        raise ValueError("CK Save Frames received no frames.")


def _validate_frames_for_descriptor(frames, ck_frames):
    _validate_image_batch(frames)
    _validate_ck_frames_descriptor(ck_frames)
    if frames.shape[0] != len(ck_frames["sprites"]):
        raise ValueError(
            f"Frame count mismatch: ck_frames describes {len(ck_frames['sprites'])} "
            f"sprite(s) but {frames.shape[0]} image(s) were supplied."
        )


def _resolve_directory_path(path: str, empty_message: str) -> str:
    path = str(path or "").strip().strip('"')
    if not path:
        raise ValueError(empty_message)
    return os.path.abspath(os.path.expanduser(path))


def _resolve_saved_frames_directory(path: str) -> str:
    resolved = _resolve_directory_path(path, "CK Load Frames path cannot be empty.")
    if os.path.isfile(resolved):
        if os.path.basename(resolved) != "ck_frames.json":
            raise ValueError("CK Load Frames file path must point to ck_frames.json.")
        return os.path.dirname(resolved)
    if os.path.isdir(resolved):
        return resolved
    raise FileNotFoundError(f"CK Load Frames folder does not exist: {resolved}")


def _animation_folder_name(ck_frames) -> str:
    animation = "frames"
    if ck_frames is not None:
        animation = str(ck_frames.get("animation") or "frames")
    cleaned = "".join(ch if ch.isalnum() or ch in " ._+-" else "_" for ch in animation)
    return (cleaned.strip(" .") or "frames")[:120]


def _frame_file_name(index: int) -> str:
    return f"frame_{index:05}.png"


def _saved_frame_file_names(frame_count: int, ck_frames):
    if ck_frames is None:
        return [_frame_file_name(index) for index in range(frame_count)]
    names = [
        os.path.basename(str(sprite.get("path", "")).replace("\\", "/"))
        for sprite in ck_frames["sprites"]
    ]
    if any(not name for name in names):
        raise ValueError("CK_FRAMES sprite path is missing a file name.")
    if len(set(names)) != len(names):
        raise ValueError(
            "CK Save Frames cannot preserve original file names because "
            "the descriptor contains duplicate file names."
        )
    return names


def _saved_frame_paths(folder: str, ck_frames):
    if ck_frames is not None:
        paths = [
            os.path.join(folder, name)
            for name in _saved_frame_file_names(len(ck_frames["sprites"]), ck_frames)
        ]
    else:
        paths = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith(".png") and name.startswith("frame_")
        ]
    if not paths:
        raise FileNotFoundError(f"No saved frame PNGs found in: {folder}")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing saved frame: {missing[0]}")
    return paths


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
    "CKFramesPreview": CKFramesPreview,
    "CKFramesToSheet": CKFramesToSheet,
    "CKSheetToFrames": CKSheetToFrames,
    "CKAutoAlignSheetStretch": CKAutoAlignSheetStretch,
    "CKManualAlignSheetStretch": CKManualAlignSheetStretch,
    "CKPackAtlas": CKPackAtlas,
    "CKMergeEdits": CKMergeEdits,
    "CKSaveFramesDescriptor": CKSaveFramesDescriptor,
    "CKLoadFramesDescriptor": CKLoadFramesDescriptor,
    "CKSaveFrames": CKSaveFrames,
    "CKLoadFrames": CKLoadFrames,
    "CKLoadProjectInfo": CKLoadProjectInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKAnimationSelector": "CK Animation Selector",
    "CKFramesPreview": "CK Frames Preview",
    "CKFramesToSheet": "CK Frames to Sheet",
    "CKSheetToFrames": "CK Sheet to Frames",
    "CKAutoAlignSheetStretch": "CK Auto Align Sheet Stretch",
    "CKManualAlignSheetStretch": "CK Manual Align Sheet Stretch",
    "CKPackAtlas": "CK Pack Atlas",
    "CKMergeEdits": "CK Merge Edits",
    "CKSaveFramesDescriptor": "CK Save Frames Descriptor",
    "CKLoadFramesDescriptor": "CK Load Frames Descriptor",
    "CKSaveFrames": "CK Save Frames",
    "CKLoadFrames": "CK Load Frames",
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
