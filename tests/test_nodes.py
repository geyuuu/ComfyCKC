"""Integration tests for the ComfyUI node layer.

Skipped automatically when torch is unavailable (e.g. the lightweight `dev`
environment); run with `uv run --group dev --group comfy pytest`.
"""

import json

import pytest
from PIL import Image

pytest.importorskip("torch")

from customknight_creator import nodes  # noqa: E402


def _make_dump(tmp_path):
    base = tmp_path / "Skin"
    root = base / "Knight"
    atlases = root / "0.Atlases"
    atlases.mkdir(parents=True)
    walk = root / "Walk"
    walk.mkdir()
    Image.new("RGBA", (10, 20), (255, 0, 0, 255)).save(walk / "0.png")
    Image.new("RGBA", (12, 16), (0, 255, 0, 255)).save(walk / "1.png")
    info = {
        "sid": ["w0", "w1"],
        "sx": [0, 16],
        "sy": [0, 0],
        "sxr": [0, 0],
        "syr": [0, 0],
        "swidth": [10, 12],
        "sheight": [20, 16],
        "sfilpped": [False, False],
        "spath": ["Knight/Walk/0.png", "Knight/Walk/1.png"],
        "scollectionname": ["Knight", "Knight"],
    }
    (atlases / "SpriteInfo.json").write_text(json.dumps(info), encoding="utf-8")
    return str(root)


def test_selector_then_pack_roundtrip(tmp_path):
    root = _make_dump(tmp_path)

    selector = nodes.CKAnimationSelector()
    images, alpha, ck_frames = selector.select(root, "Knight", "Walk")

    # Two frames padded to the largest size (12 x 20).
    assert images.shape == (2, 20, 12, 3)
    assert alpha.shape == (2, 20, 12)
    assert len(ck_frames["sprites"]) == 2

    packer = nodes.CKPackAtlas()
    out = packer.pack(ck_frames, images, edited_alpha=alpha, save_to_output=False)
    atlas, saved_path = out["result"]

    # max right = 16 + 12 = 28 -> 32 ; max top = max(20, 16) -> 32
    # RGBA: the atlas output keeps its alpha channel so transparency survives.
    assert atlas.shape == (1, 32, 32, 4)
    assert saved_path == ""


def test_pack_frame_count_mismatch(tmp_path):
    root = _make_dump(tmp_path)
    selector = nodes.CKAnimationSelector()
    images, _alpha, ck_frames = selector.select(root, "Knight", "Walk")

    packer = nodes.CKPackAtlas()
    with pytest.raises(ValueError, match="Frame count mismatch"):
        packer.pack(ck_frames, images[:1], save_to_output=False)
