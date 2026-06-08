"""CK Video Combine - turn an IMAGE batch into an in-memory ComfyUI ``VIDEO``.

This is a port of the *Video Combine* node from
`ComfyUI-VideoHelperSuite <https://github.com/kosinkadink/ComfyUI-VideoHelperSuite>`_
(GPLv3). The original writes the encoded video into ComfyUI's ``output/`` folder
and returns the file path(s). Here the encoder is kept (ffmpeg for real video
containers, Pillow for animated gif/webp, using VHS's bundled ``video_formats``
JSON definitions) but the node's product is a native ``VIDEO`` object instead:

* frames are encoded into ComfyUI's *temp* directory (full container + audio
  support; ffmpeg cannot mux audio purely in a pipe on Windows),
* the finished bytes are read back into an ``io.BytesIO`` and wrapped in
  ``comfy_api``'s ``VideoFromFile`` -> a self-contained ``VIDEO`` output,
* the same temp file feeds the on-node preview.

Nothing is persisted to ``output/``.

The ComfyUI-only imports (``folder_paths``, ``comfy_api``) are deferred into
:meth:`CKVideoCombine.combine_video` so this module - and the pure
:func:`encode_to_file` encoder - can be imported and unit-tested without a
ComfyUI host.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
from string import Template

import numpy as np
import torch
from PIL import ExifTags, Image

ENCODE_ARGS = ("utf-8", "backslashreplace")

CATEGORY = "CustomKnight"


# ---------------------------------------------------------------------------
# ffmpeg discovery (ported from videohelpersuite/utils.py)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    if "VHS_FORCE_FFMPEG_PATH" in os.environ:
        return os.environ["VHS_FORCE_FFMPEG_PATH"]
    try:  # bundled binary - the recommended way to get ffmpeg without a system install
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    for cand in ("ffmpeg", "ffmpeg.exe"):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


ffmpeg_path = _find_ffmpeg()


# ---------------------------------------------------------------------------
# Format loading (ported from videohelpersuite/nodes.py)
# ---------------------------------------------------------------------------
base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats")

# Formats whose container can carry an alpha channel, so "preserve transparency"
# can keep it instead of compositing onto the background colour. Everything else
# composites: h264/h265 mp4 and av1 have no alpha support, and ffmpeg can *decode*
# but not *encode* VP8/VP9 alpha, so webm can't carry it either.
ALPHA_CAPABLE_FORMATS = {"gif", "webp", "ProRes", "ffv1-mkv"}

# Widget values that actually switch a video format's pixel format to an alpha
# one. (gif/webp keep alpha just by being fed RGBA; ffv1's default rgba64le
# already carries alpha, so neither needs an override here.)
_ALPHA_VIDEO_OVERRIDES = {
    "ProRes": {"profile": "4444"},
}


def format_keeps_alpha(fmt):
    """Whether the chosen ``image/`` or ``video/`` format can store transparency."""
    return fmt.split("/", 1)[-1] in ALPHA_CAPABLE_FORMATS


def flatten_list(lst):
    ret = []
    for e in lst:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret


def iterate_format(video_format, for_widgets=True):
    """Iterate over a format definition's widgets (or, with ``for_widgets=False``,
    its concrete ffmpeg arguments), allowing values to be substituted in place."""

    def indirector(cont, index):
        if isinstance(cont[index], list) and (
            not for_widgets
            or len(cont[index]) > 1
            and not isinstance(cont[index][1], dict)
        ):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield

    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)


def get_video_formats() -> list[str]:
    """Return the ``video/<name>`` ids of every bundled format JSON."""
    formats = []
    if not os.path.isdir(base_formats_dir):
        return formats
    for item in sorted(os.scandir(base_formats_dir), key=lambda e: e.name):
        if not item.is_file() or not item.name.endswith(".json"):
            continue
        with open(item.path) as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format:
            # gifski needs an external binary we don't bundle.
            continue
        formats.append("video/" + item.name[:-5])
    return formats


def apply_format_widgets(format_name, kwargs):
    """Load a format JSON and resolve its widgets, filling any value not supplied
    in ``kwargs`` with the JSON-declared default (we don't expose the per-format
    widgets, so every value falls back to its default)."""
    video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    with open(video_format_path) as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and "default" in w[2]:
                default = w[2]["default"]
            elif type(w[1]) is list:
                default = w[1][0]
            else:
                default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format


# ---------------------------------------------------------------------------
# Tensor / frame helpers (ported from videohelpersuite/nodes.py & utils.py)
# ---------------------------------------------------------------------------
def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits - 1) + 0.5
    return np.clip(tensor, 0, (2**bits - 1))


def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)


def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)


def parse_color(color, default=(0.0, 0.0, 0.0)):
    """Parse a ``#RRGGBB`` / ``#RGB`` (``#`` optional) hex string into an RGB
    tuple of floats in ``[0, 1]``. Falls back to ``default`` on anything invalid."""
    if not isinstance(color, str):
        return default
    s = color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return default
    try:
        r, g, b = (int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return default
    return (r, g, b)


def composite_over(images, first_image, background_color):
    """If frames carry alpha (4 channels), alpha-blend them onto a solid
    ``background_color`` and drop the alpha. Most video codecs can't store
    transparency, and sprite atlases leave arbitrary RGB in fully-transparent
    texels - without this they would leak through as a garbage background.

    Returns ``(images, first_image)`` where both are now 3-channel RGB. When the
    input has no alpha they are returned unchanged.
    """
    if first_image.shape[-1] != 4:
        return images, first_image
    bg = torch.tensor(parse_color(background_color), dtype=torch.float32)

    def _flatten(img):
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        return rgb * alpha + bg * (1.0 - alpha)

    return map(_flatten, images), _flatten(first_image)


def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]


def merge_filter_args(args, ftype="-vf"):
    """Collapse repeated ``-vf`` (or ``-af``) options into a single comma-joined
    filter chain, as ffmpeg only honours the last one otherwise."""
    try:
        start_index = args.index(ftype) + 1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += "," + args[index + 1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass


def ffmpeg_process(args, file_path, env):
    """Generator that pumps raw frame bytes into ffmpeg's stdin. ``send`` each
    frame, then ``send(None)`` to flush; it yields the number of frames written."""
    res = None
    frame_data = yield
    total_frames_output = 0
    with subprocess.Popen(
        args + [file_path], stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env
    ) as proc:
        try:
            while frame_data is not None:
                proc.stdin.write(frame_data)
                frame_data = yield
                total_frames_output += 1
            proc.stdin.flush()
            proc.stdin.close()
            res = proc.stderr.read()
        except BrokenPipeError:
            res = proc.stderr.read()
            raise Exception(
                "An error occurred in the ffmpeg subprocess:\n" + res.decode(*ENCODE_ARGS)
            )
    yield total_frames_output
    if res is not None and len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def _pump_ffmpeg(args, file_path, env, frame_iter, pbar=None):
    """Drive :func:`ffmpeg_process` to completion: prime it, pump every frame's
    bytes, then flush. Returns the number of frames written."""
    proc = ffmpeg_process(args, file_path, env)
    proc.send(None)  # advance to the first yield
    for frame in frame_iter:
        if pbar is not None:
            pbar.update(1)
        proc.send(frame)
    total = 0
    try:
        total = proc.send(None)
        proc.send(None)
    except StopIteration:
        pass
    return total


# ---------------------------------------------------------------------------
# The encoder - ComfyUI-free so it can be unit-tested directly.
# ---------------------------------------------------------------------------
def encode_to_file(
    images,
    frame_rate,
    out_folder,
    filename,
    counter,
    loop_count=0,
    fmt="video/h264-mp4",
    pingpong=False,
    audio=None,
    background_color="#000000",
    preserve_alpha=False,
    pbar=None,
    **kwargs,
):
    """Encode an IMAGE batch into a single container in ``out_folder``.

    Returns ``(final_path, mimetype)``. ``final_path`` is the audio-muxed file
    when ``audio`` is supplied, otherwise the plain video. No ComfyUI imports.
    """
    if images is None or (isinstance(images, torch.Tensor) and images.size(0) == 0):
        raise ValueError("CK Video Combine received no frames.")

    num_frames = len(images)
    first_image = images[0]
    images = iter(images)

    # Keep real transparency only when asked AND the format supports it; otherwise
    # flatten alpha onto the background colour so transparent sprite pixels don't
    # leak their arbitrary RGB into formats that can't store transparency.
    keep_alpha = preserve_alpha and first_image.shape[-1] == 4 and format_keeps_alpha(fmt)
    if not keep_alpha:
        images, first_image = composite_over(images, first_image, background_color)

    format_type, format_ext = fmt.split("/")

    if format_type == "image":
        file = f"{filename}_{counter:05}.{format_ext}"
        file_path = os.path.join(out_folder, file)
        if pingpong:
            images = to_pingpong(images)

        if ffmpeg_path is not None:
            # Encode gif/webp via ffmpeg so frame_rate is honoured exactly. Pillow
            # does not persist animated-webp frame durations (the timing is lost),
            # and ffmpeg's palettegen produces better gifs. Pillow is only used as
            # a fallback below when no ffmpeg is available.
            has_alpha = first_image.shape[-1] == 4
            i_pix_fmt = "rgba" if has_alpha else "rgb24"
            in_args = [
                ffmpeg_path,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                i_pix_fmt,
                "-s",
                f"{first_image.shape[1]}x{first_image.shape[0]}",
                "-r",
                str(frame_rate),
                "-i",
                "-",
            ]
            if format_ext == "gif":
                out_args = [
                    "-filter_complex",
                    "split[a][b];[a]palettegen=reserve_transparent=on"
                    ":transparency_color=ffffff[p];[b][p]paletteuse=dither=sierra2_4a",
                    "-loop",
                    str(loop_count),
                ]
            else:  # webp
                lossless = 1 if kwargs.get("lossless", True) else 0
                out_args = [
                    "-c:v",
                    "libwebp_anim",
                    "-lossless",
                    str(lossless),
                    "-q:v",
                    "90",
                    "-loop",
                    str(loop_count),
                ]
            frame_bytes = (tensor_to_bytes(i).tobytes() for i in images)
            _pump_ffmpeg(in_args + out_args, file_path, os.environ.copy(), frame_bytes, pbar)
            return file_path, f"image/{format_ext}"

        # ---- Pillow fallback (no ffmpeg available) ------------------------
        # NOTE: gif is centisecond-limited and animated-webp timing may be
        # approximate on some Pillow versions.
        image_kwargs = {}
        if format_ext == "gif":
            image_kwargs["disposal"] = 2
        if format_ext == "webp":
            exif = Image.Exif()
            exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
            image_kwargs["exif"] = exif
            image_kwargs["lossless"] = kwargs.get("lossless", True)

        def frames_gen(imgs):
            for i in imgs:
                if pbar is not None:
                    pbar.update(1)
                yield Image.fromarray(tensor_to_bytes(i))

        frames = frames_gen(images)
        next(frames).save(
            file_path,
            format=format_ext.upper(),
            save_all=True,
            append_images=frames,
            duration=round(1000 / frame_rate),
            loop=loop_count,
            compress_level=4,
            **image_kwargs,
        )
        return file_path, f"image/{format_ext}"

    # ---- ffmpeg-encoded video container -----------------------------------
    if ffmpeg_path is None:
        raise ProcessLookupError(
            "ffmpeg is required for video formats and could not be found. Install "
            "imageio-ffmpeg (`pip install imageio-ffmpeg`) or put ffmpeg on PATH."
        )

    has_alpha = first_image.shape[-1] == 4
    kwargs["has_alpha"] = has_alpha
    if has_alpha:
        # Switch the format to an alpha-carrying pixel format / profile.
        kwargs.update(_ALPHA_VIDEO_OVERRIDES.get(format_ext, {}))
    video_format = apply_format_widgets(format_ext, kwargs)
    dim_alignment = video_format.get("dim_alignment", 2)
    if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
        # Dimensions must be a multiple of dim_alignment; pad (replicate edge).
        to_pad = (-first_image.shape[1] % dim_alignment, -first_image.shape[0] % dim_alignment)
        padding = (
            to_pad[0] // 2,
            to_pad[0] - to_pad[0] // 2,
            to_pad[1] // 2,
            to_pad[1] - to_pad[1] // 2,
        )
        padfunc = torch.nn.ReplicationPad2d(padding)

        def pad(image):
            image = image.permute((2, 0, 1))  # HWC -> CHW
            padded = padfunc(image.to(dtype=torch.float32))
            return padded.permute((1, 2, 0))

        images = map(pad, images)
        dimensions = (
            -first_image.shape[1] % dim_alignment + first_image.shape[1],
            -first_image.shape[0] % dim_alignment + first_image.shape[0],
        )
    else:
        dimensions = (first_image.shape[1], first_image.shape[0])

    if pingpong:
        images = to_pingpong(images)
        if num_frames > 2:
            num_frames += num_frames - 2
            if pbar is not None:
                pbar.total = num_frames

    if loop_count > 0:
        loop_args = ["-vf", "loop=loop=" + str(loop_count) + ":size=" + str(num_frames)]
    else:
        loop_args = []

    if video_format.get("input_color_depth", "8bit") == "16bit":
        images = map(tensor_to_shorts, images)
        i_pix_fmt = "rgba64" if has_alpha else "rgb48"
    else:
        images = map(tensor_to_bytes, images)
        i_pix_fmt = "rgba" if has_alpha else "rgb24"

    file = f"{filename}_{counter:05}.{video_format['extension']}"
    file_path = os.path.join(out_folder, file)

    bitrate_arg = []
    bitrate = video_format.get("bitrate")
    if bitrate is not None:
        bitrate_arg = [
            "-b:v",
            str(bitrate) + "M" if video_format.get("megabit") == "True" else str(bitrate) + "K",
        ]

    args = [
        ffmpeg_path,
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        i_pix_fmt,
        # Treat the incoming RGB as full-range sRGB and tell ffmpeg it is already
        # BT.709 primaries so per-format YUV conversion stays consistent (VHS note).
        "-color_range",
        "pc",
        "-colorspace",
        "rgb",
        "-color_primaries",
        "bt709",
        "-color_trc",
        video_format.get("fake_trc", "iec61966-2-1"),
        "-s",
        f"{dimensions[0]}x{dimensions[1]}",
        "-r",
        str(frame_rate),
        "-i",
        "-",
    ] + loop_args

    images = map(lambda x: x.tobytes(), images)
    env = os.environ.copy()
    if "environment" in video_format:
        env.update(video_format["environment"])

    if "pre_pass" in video_format:
        images = [b"".join(images)]
        in_args_len = args.index("-i") + 2
        pre_pass_args = args[:in_args_len] + video_format["pre_pass"]
        merge_filter_args(pre_pass_args)
        try:
            subprocess.run(pre_pass_args, input=images[0], env=env, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(
                "An error occurred in the ffmpeg prepass:\n" + e.stderr.decode(*ENCODE_ARGS)
            )
    if "inputs_main_pass" in video_format:
        in_args_len = args.index("-i") + 2
        args = args[:in_args_len] + video_format["inputs_main_pass"] + args[in_args_len:]

    args += video_format["main_pass"] + bitrate_arg
    merge_filter_args(args)

    total_frames_output = _pump_ffmpeg(args, file_path, env, images, pbar)

    mimetype = f"video/{video_format['extension']}"
    final_path = file_path

    # ---- optional audio mux (second ffmpeg pass) --------------------------
    a_waveform = None
    if audio is not None:
        try:
            a_waveform = audio["waveform"]
        except Exception:
            a_waveform = None
    if a_waveform is not None:
        file = f"{filename}_{counter:05}-audio.{video_format['extension']}"
        output_file_with_audio_path = os.path.join(out_folder, file)
        if "audio_pass" not in video_format:
            video_format["audio_pass"] = ["-c:a", "libopus"]
        channels = audio["waveform"].size(1)
        min_audio_dur = total_frames_output / frame_rate + 1
        if video_format.get("trim_to_audio", "False") != "False":
            apad = []
        else:
            apad = ["-af", "apad=whole_dur=" + str(min_audio_dur)]
        mux_args = (
            [
                ffmpeg_path,
                "-v",
                "error",
                "-n",
                "-i",
                file_path,
                "-ar",
                str(audio["sample_rate"]),
                "-ac",
                str(channels),
                "-f",
                "f32le",
                "-i",
                "-",
                "-c:v",
                "copy",
            ]
            + video_format["audio_pass"]
            + apad
            + ["-shortest", output_file_with_audio_path]
        )
        audio_data = audio["waveform"].squeeze(0).transpose(0, 1).numpy().tobytes()
        merge_filter_args(mux_args, "-af")
        try:
            res = subprocess.run(
                mux_args, input=audio_data, env=env, capture_output=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception(
                "An error occurred in the ffmpeg subprocess:\n" + e.stderr.decode(*ENCODE_ARGS)
            )
        if res.stderr:
            print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        final_path = output_file_with_audio_path

    return final_path, mimetype


# ---------------------------------------------------------------------------
# CKVideoCombine node
# ---------------------------------------------------------------------------
class CKVideoCombine:
    """Combine an IMAGE batch into a video and output it as a native VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        formats = ["image/gif", "image/webp"] + get_video_formats()
        default = "video/h264-mp4" if "video/h264-mp4" in formats else formats[0]
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 8.0, "min": 1.0, "step": 1.0}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "CKVideo"}),
                "format": (formats, {"default": default}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "background_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Transparent (RGBA) frames are flattened onto this "
                        "hex colour, since most video codecs can't store transparency. "
                        "Ignored for opaque (RGB) input or when preserve_alpha keeps it.",
                    },
                ),
                "preserve_alpha": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep real transparency for formats that support it "
                        "(ProRes, ffv1, gif, webp). Formats that can't carry alpha "
                        "(h264/h265 mp4, webm, av1) still composite onto "
                        "background_color.",
                    },
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    FUNCTION = "combine_video"
    DESCRIPTION = (
        "Encode an IMAGE batch into a video (ffmpeg formats, or animated gif/webp) "
        "and output it directly as a ComfyUI VIDEO object. The result is held in "
        "memory and previewed from the temp folder; nothing is written to output/."
    )

    def combine_video(
        self,
        images,
        frame_rate,
        loop_count=0,
        filename_prefix="CKVideo",
        format="video/h264-mp4",
        pingpong=False,
        background_color="#000000",
        preserve_alpha=False,
        audio=None,
        **kwargs,
    ):
        # ComfyUI-only imports are deferred so the encoder above stays testable.
        import folder_paths

        try:
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:  # pragma: no cover - old ComfyUI without VIDEO type
            raise ImportError(
                "CK Video Combine needs a ComfyUI version with the native VIDEO type "
                "(comfy_api.input_impl.VideoFromFile)."
            ) from exc

        try:
            from comfy.utils import ProgressBar

            pbar = ProgressBar(len(images))
        except Exception:
            pbar = None

        # Always encode into the temp directory - never persist to output/.
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )

        final_path, mimetype = encode_to_file(
            images,
            frame_rate,
            full_output_folder,
            filename,
            counter,
            loop_count=loop_count,
            fmt=format,
            pingpong=pingpong,
            audio=audio,
            background_color=background_color,
            preserve_alpha=preserve_alpha,
            pbar=pbar,
            **kwargs,
        )

        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise RuntimeError(
                f"Encoding produced no output at {final_path}. See the ffmpeg log above."
            )

        # Read the finished container into memory so the VIDEO is self-contained
        # even after the temp file is cleaned up.
        with open(final_path, "rb") as fh:
            data = fh.read()
        video = VideoFromFile(io.BytesIO(data))

        preview = {
            "filename": os.path.basename(final_path),
            "subfolder": subfolder,
            "type": "temp",
            "format": mimetype,
            "frame_rate": frame_rate,
        }
        return {"ui": {"images": [preview], "animated": (True,)}, "result": (video,)}


NODE_CLASS_MAPPINGS = {"CKVideoCombine": CKVideoCombine}
NODE_DISPLAY_NAME_MAPPINGS = {"CKVideoCombine": "CK Video Combine"}
