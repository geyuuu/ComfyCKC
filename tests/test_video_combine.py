"""Tests for the CK Video Combine encoder.

The pure :func:`encode_to_file` encoder has no ComfyUI dependency, so the
Pillow gif/webp path runs anywhere torch + Pillow are present. The ffmpeg path
is skipped when no ffmpeg binary is discoverable. Run with
``uv run --group dev --group comfy pytest``.
"""

import os
import subprocess

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from customknight_creator import video_combine  # noqa: E402


def test_get_video_formats_lists_bundled_formats():
    fmts = video_combine.get_video_formats()
    assert "video/h264-mp4" in fmts
    assert "video/webm" in fmts
    assert all(f.startswith("video/") for f in fmts)
    # gifski needs an external binary and must be filtered out.
    assert "video/gifski" not in fmts


def test_encode_gif_frame_count(tmp_path):
    images = torch.rand(4, 16, 16, 3)
    path, mime = video_combine.encode_to_file(
        images, 8.0, str(tmp_path), "anim", 1, fmt="image/gif"
    )
    assert mime == "image/gif"
    assert path.endswith(".gif")
    with Image.open(path) as im:
        assert im.is_animated
        assert im.n_frames == 4


def test_encode_gif_pingpong_appends_reversed(tmp_path):
    images = torch.rand(4, 16, 16, 3)
    path, _ = video_combine.encode_to_file(
        images, 8.0, str(tmp_path), "pp", 1, fmt="image/gif", pingpong=True
    )
    with Image.open(path) as im:
        # 4 frames + (4 - 2) reversed middle frames.
        assert im.n_frames == 6


def test_encode_no_frames_raises(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        video_combine.encode_to_file(
            torch.zeros(0, 8, 8, 3), 8.0, str(tmp_path), "empty", 1, fmt="image/gif"
        )


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_encode_h264_mp4(tmp_path):
    images = torch.rand(6, 16, 24, 3)
    path, mime = video_combine.encode_to_file(
        images, 8.0, str(tmp_path), "vid", 1, fmt="video/h264-mp4"
    )
    assert mime == "video/mp4"
    assert path.endswith(".mp4")
    assert os.path.getsize(path) > 0


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_encode_h264_odd_dimensions_padded(tmp_path):
    # Odd width/height force the dim_alignment padding path; encoding must succeed.
    images = torch.rand(3, 15, 17, 3)
    path, _ = video_combine.encode_to_file(
        images, 8.0, str(tmp_path), "odd", 1, fmt="video/h264-mp4"
    )
    assert os.path.getsize(path) > 0


def _webp_frame_durations_ms(path):
    """Per-frame durations (ms) parsed straight from a WebP's ANMF chunks.

    Neither Pillow nor ffmpeg reliably expose animated-webp timing, so we read
    the 24-bit little-endian Frame Duration field of each ANMF chunk directly.
    """
    import struct

    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    durs = []
    off = 12
    while off + 8 <= len(data):
        fourcc = data[off : off + 4]
        size = struct.unpack("<I", data[off + 4 : off + 8])[0]
        if fourcc == b"ANMF":
            p = data[off + 8 : off + 8 + size]
            durs.append(p[12] | (p[13] << 8) | (p[14] << 16))
        off += 8 + size + (size & 1)  # chunks are padded to an even size
    return durs


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_encode_webp_honours_frame_rate(tmp_path):
    # Regression: Pillow wrote no per-frame durations for animated webp, so the
    # clip played at ffmpeg's ~25fps default and was the wrong length. Encoding
    # via ffmpeg must embed durations that match frame_rate exactly.
    path, mime = video_combine.encode_to_file(
        torch.rand(10, 16, 16, 3), 4.0, str(tmp_path), "w", 1, fmt="image/webp"
    )
    assert mime == "image/webp"
    durs = _webp_frame_durations_ms(path)
    assert len(durs) == 10
    assert all(d == 250 for d in durs)  # 4 fps -> 250 ms/frame
    assert sum(durs) == 2500  # 2.5 s total


def test_parse_color():
    assert video_combine.parse_color("#000000") == (0.0, 0.0, 0.0)
    assert video_combine.parse_color("#ffffff") == (1.0, 1.0, 1.0)
    assert video_combine.parse_color("f00") == (1.0, 0.0, 0.0)  # short form
    assert video_combine.parse_color("00ff00") == (0.0, 1.0, 0.0)  # '#' optional
    assert video_combine.parse_color("nope!!") == (0.0, 0.0, 0.0)  # invalid -> default
    assert video_combine.parse_color("") == (0.0, 0.0, 0.0)


def test_composite_over_flattens_alpha_to_background():
    # Fully transparent frames with RED garbage in the RGB channels must become
    # the pure background colour (the garbage must not survive).
    arr = np.zeros((2, 4, 4, 4), np.float32)
    arr[..., 0] = 1.0  # red garbage under transparent pixels
    arr[..., 3] = 0.0  # fully transparent
    imgs = torch.from_numpy(arr)
    _, first = video_combine.composite_over(iter(imgs), imgs[0], "#00ff00")
    assert first.shape[-1] == 3
    assert torch.allclose(first, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_composite_over_passes_rgb_through():
    imgs = torch.rand(2, 4, 4, 3)
    _, first = video_combine.composite_over(iter(imgs), imgs[0], "#123456")
    assert first.shape[-1] == 3
    assert torch.equal(first, imgs[0])


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_video_composites_transparent_pixels_over_background(tmp_path):
    # Regression: video codecs drop alpha, exposing the arbitrary RGB sprite
    # atlases leave in transparent texels as a garbage background. We must
    # alpha-composite onto background_color first. Transparent RED garbage with
    # an opaque GREEN centre -> on a BLUE background the corner must be blue.
    arr = np.zeros((3, 32, 32, 4), np.float32)
    arr[..., 0] = 1.0  # red garbage
    arr[..., 3] = 0.0  # transparent
    arr[:, 12:20, 12:20, 0] = 0.0
    arr[:, 12:20, 12:20, 1] = 1.0  # green
    arr[:, 12:20, 12:20, 3] = 1.0  # opaque
    images = torch.from_numpy(arr)
    path, _ = video_combine.encode_to_file(
        images, 8.0, str(tmp_path), "bg", 1, fmt="video/webm", background_color="#0000FF"
    )
    frame = os.path.join(str(tmp_path), "frame.png")
    subprocess.run(
        [video_combine.ffmpeg_path, "-v", "error", "-y", "-i", path, "-frames:v", "1", frame],
        check=True,
        capture_output=True,
    )
    r, g, b = Image.open(frame).convert("RGB").getpixel((0, 0))  # transparent corner
    assert b > 200 and r < 60 and g < 60  # blue background, not red garbage


def test_format_keeps_alpha():
    assert video_combine.format_keeps_alpha("image/gif")
    assert video_combine.format_keeps_alpha("image/webp")
    assert video_combine.format_keeps_alpha("video/ProRes")
    assert video_combine.format_keeps_alpha("video/ffv1-mkv")
    # ffmpeg can't encode VP8/VP9 alpha, and these have no alpha channel at all.
    assert not video_combine.format_keeps_alpha("video/webm")
    assert not video_combine.format_keeps_alpha("video/h264-mp4")
    assert not video_combine.format_keeps_alpha("video/av1-webm")


def _rgba_garbage_frames(n=3, size=32):
    # transparent RED garbage everywhere, opaque GREEN square in the centre
    arr = np.zeros((n, size, size, 4), np.float32)
    arr[..., 0] = 1.0
    arr[..., 3] = 0.0
    lo, hi = size * 3 // 8, size * 5 // 8
    arr[:, lo:hi, lo:hi, 0] = 0.0
    arr[:, lo:hi, lo:hi, 1] = 1.0
    arr[:, lo:hi, lo:hi, 3] = 1.0
    return torch.from_numpy(arr)


def _first_frame_rgba(path, tmp_path):
    frame = os.path.join(str(tmp_path), "frame.png")
    subprocess.run(
        [video_combine.ffmpeg_path, "-v", "error", "-y", "-i", path, "-frames:v", "1", frame],
        check=True,
        capture_output=True,
    )
    return Image.open(frame).convert("RGBA")


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_preserve_alpha_keeps_transparency_for_prores(tmp_path):
    path, _ = video_combine.encode_to_file(
        _rgba_garbage_frames(), 8.0, str(tmp_path), "pr", 1,
        fmt="video/ProRes", preserve_alpha=True, background_color="#0000FF",
    )
    im = _first_frame_rgba(path, tmp_path)
    assert im.getpixel((0, 0))[3] == 0  # transparent corner stays transparent
    cx = im.width // 2
    assert im.getpixel((cx, cx))[3] == 255  # opaque centre stays opaque


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_preserve_alpha_falls_back_to_composite_for_webm(tmp_path):
    # webm can't carry alpha via ffmpeg, so even with preserve_alpha it must
    # composite (no opaque red garbage, no transparency).
    path, _ = video_combine.encode_to_file(
        _rgba_garbage_frames(), 8.0, str(tmp_path), "wm", 1,
        fmt="video/webm", preserve_alpha=True, background_color="#0000FF",
    )
    r, g, b, a = _first_frame_rgba(path, tmp_path).getpixel((0, 0))
    assert b > 200 and r < 60 and g < 60  # blue background, not red garbage


@pytest.mark.skipif(video_combine.ffmpeg_path is None, reason="ffmpeg not available")
def test_encode_gif_total_duration(tmp_path):
    # GIF timing is centisecond-quantized by the format, so 8 fps can't be exact,
    # but ffmpeg's palettegen output must still total close to the expected length.
    path, _ = video_combine.encode_to_file(
        torch.rand(16, 16, 16, 3), 8.0, str(tmp_path), "g", 1, fmt="image/gif"
    )
    im = Image.open(path)
    total = 0
    for i in range(im.n_frames):
        im.seek(i)
        total += im.info.get("duration", 0)
    assert abs(total - 2000) <= 100  # within 5% of 2.0 s
