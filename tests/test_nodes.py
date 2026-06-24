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


def test_selector_frames_sheet_roundtrip_is_exact(tmp_path):
    root = _make_dump(tmp_path)
    frames, _alpha, _ck_frames = nodes.CKAnimationSelector().select(
        root, "Knight", "Walk"
    )

    sheet, layout = nodes.CKFramesToSheet().combine(frames, columns=1)
    restored, = nodes.CKSheetToFrames().split(sheet, layout)

    assert sheet.shape == (1, 40, 12, 3)
    assert restored.shape == frames.shape
    assert restored.dtype == frames.dtype
    assert restored.device == frames.device
    assert nodes.torch.equal(restored, frames)


def test_frames_sheet_auto_grid_and_roundtrip_are_exact():
    frames = nodes.torch.arange(5 * 2 * 3 * 4, dtype=nodes.torch.float32).reshape(
        5, 2, 3, 4
    )

    sheet, layout = nodes.CKFramesToSheet().combine(frames)
    restored, = nodes.CKSheetToFrames().split(sheet, layout)

    assert sheet.shape == (1, 4, 9, 4)
    assert layout == {
        "version": 1,
        "frame_count": 5,
        "frame_height": 2,
        "frame_width": 3,
        "channels": 4,
        "columns": 3,
        "rows": 2,
    }
    assert nodes.torch.equal(restored, frames)
    # The unused final grid cell is transparent/black rather than duplicated.
    assert nodes.torch.count_nonzero(sheet[:, 2:4, 6:9, :]) == 0


def test_sheet_split_rejects_a_resized_sheet():
    frames = nodes.torch.zeros((2, 4, 5, 3))
    sheet, layout = nodes.CKFramesToSheet().combine(frames)

    with pytest.raises(ValueError, match="Sheet shape mismatch"):
        nodes.CKSheetToFrames().split(sheet[:, :-1], layout)


def _make_numbered_dump(tmp_path):
    """Numbered animation folders split across two atlases (collections)."""
    base = tmp_path / "Skin"
    root = base / "Hornet"
    atlases = root / "0.Atlases"
    atlases.mkdir(parents=True)
    run = root / "001.Run"
    idle = root / "002.Idle"
    run.mkdir()
    idle.mkdir()
    Image.new("RGBA", (10, 20), (255, 0, 0, 255)).save(run / "001-00-000.png")
    Image.new("RGBA", (12, 16), (0, 255, 0, 255)).save(run / "001-01-001.png")
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(idle / "002-00-002.png")
    info = {
        "sid": ["r0", "r1", "i0"],
        "sx": [0, 0, 16],
        "sy": [0, 0, 0],
        "sxr": [0, 0, 0],
        "syr": [0, 0, 0],
        "swidth": [10, 12, 8],
        "sheight": [20, 16, 8],
        "sfilpped": [False, False, False],
        "spath": [
            "Hornet/001.Run/001-00-000.png",
            "Hornet/001.Run/001-01-001.png",
            "Hornet/002.Idle/002-00-002.png",
        ],
        "scollectionname": ["atlas0", "atlas1", "atlas0"],
    }
    (atlases / "SpriteInfo.json").write_text(json.dumps(info), encoding="utf-8")
    return str(root)


def test_selector_range_mode_concatenates_across_collections(tmp_path):
    root = _make_numbered_dump(tmp_path)
    selector = nodes.CKAnimationSelector()
    images, alpha, ck_frames = selector.select(
        root, "", "", mode="animation range", animation_range="1-2"
    )

    # Three frames padded to the largest (12 x 20).
    assert images.shape == (3, 20, 12, 3)
    assert alpha.shape == (3, 20, 12)
    assert ck_frames["mode"] == "animation range"
    assert ck_frames["collection"] is None
    assert ck_frames["collections"] == ["atlas0", "atlas1"]
    assert len(ck_frames["sprites"]) == 3


def test_selector_range_mode_requires_a_range(tmp_path):
    root = _make_numbered_dump(tmp_path)
    selector = nodes.CKAnimationSelector()
    with pytest.raises(ValueError, match="enter a range"):
        selector.select(root, "", "", mode="animation range", animation_range="")


def test_pack_range_writes_one_atlas_per_collection(tmp_path):
    root = _make_numbered_dump(tmp_path)
    selector = nodes.CKAnimationSelector()
    images, alpha, ck_frames = selector.select(
        root, "", "", mode="animation range", animation_range="1-2"
    )

    ext_dir = tmp_path / "skin"
    packer = nodes.CKPackAtlas()
    out = packer.pack(
        ck_frames,
        images,
        edited_alpha=alpha,
        save_to_output=False,
        external_directory=str(ext_dir),
    )
    atlas, saved_path = out["result"]

    # Two collections -> a 2-image batch, padded to the larger atlas (atlas0 is
    # 32x32: right 16+8=24 -> 32, top 20 -> 32; atlas1 is 16x16).
    assert atlas.shape == (2, 32, 32, 4)
    # One <collection>.png written to the external skin folder for each.
    assert (ext_dir / "atlas0.png").is_file()
    assert (ext_dir / "atlas1.png").is_file()
    assert str(ext_dir / "atlas0.png") in saved_path
    assert str(ext_dir / "atlas1.png") in saved_path
