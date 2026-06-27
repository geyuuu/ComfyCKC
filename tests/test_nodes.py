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
    Image.new("RGBA", (10, 20), (255, 0, 0, 128)).save(walk / "0.png")
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
    images, ck_frames = selector.select(root, "Knight", "Walk")

    # Two RGBA frames padded transparently to the largest size (12 x 20).
    assert images.shape == (2, 20, 12, 4)
    assert images[0, 0, 0, 3].item() == pytest.approx(128 / 255)
    assert images[0, 0, 10, 3].item() == 0
    assert len(ck_frames["sprites"]) == 2

    packer = nodes.CKPackAtlas()
    out = packer.pack(images, ck_frames, save_to_output=False)
    atlas, saved_path = out["result"]

    # max right = 16 + 12 = 28 -> 32 ; max top = max(20, 16) -> 32
    # RGBA: the atlas output keeps its alpha channel so transparency survives.
    assert atlas.shape == (1, 32, 32, 4)
    assert nodes.torch.any(nodes.torch.isclose(atlas[..., 3], nodes.torch.tensor(128 / 255)))
    assert saved_path == ""


def test_alpha_is_integrated_into_image_ports():
    assert nodes.CKAnimationSelector.RETURN_TYPES == ("IMAGE", "CK_FRAMES")
    assert nodes.CKAnimationSelector.RETURN_NAMES == ("frames", "ck_frames")

    merge_inputs = nodes.CKMergeEdits.INPUT_TYPES()
    assert list(merge_inputs["required"]) == [
        "frames_a",
        "ck_frames_a",
        "frames_b",
        "ck_frames_b",
    ]
    assert merge_inputs["optional"]["frames_3"] == ("IMAGE",)
    assert merge_inputs["optional"]["ck_frames_3"] == ("CK_FRAMES",)
    assert nodes.CKMergeEdits.RETURN_TYPES == ("CK_FRAMES", "IMAGE")
    assert nodes.CKMergeEdits.RETURN_NAMES == ("ck_frames", "frames")

    pack_inputs = nodes.CKPackAtlas.INPUT_TYPES()
    assert list(pack_inputs["required"]) == ["edited_frames", "ck_frames"]
    assert "edited_alpha" not in pack_inputs["optional"]

    assert nodes.CKSaveFramesDescriptor.RETURN_NAMES == ("ck_frames", "saved_path")
    assert nodes.CKLoadFramesDescriptor.RETURN_NAMES == ("ck_frames", "loaded_path")
    save_frames_inputs = nodes.CKSaveFrames.INPUT_TYPES()
    assert "ck_frames" not in save_frames_inputs["required"]
    assert save_frames_inputs["optional"]["ck_frames"] == ("CK_FRAMES",)


def test_merge_edits_preserves_rgba_and_transparent_padding():
    frames_a = nodes.torch.ones((1, 2, 3, 4))
    frames_a[..., 3] = 0.25
    frames_b = nodes.torch.ones((1, 3, 2, 4))
    frames_b[..., 3] = 0.75
    frames_c = nodes.torch.ones((1, 1, 4, 4))
    frames_c[..., 3] = 0.5
    descriptor_a = {
        "collection": "Knight",
        "collections": ["Knight"],
        "frame_size": [3, 2],
        "animation": "Walk",
        "sprites": [{"path": "Walk/0.png"}],
    }
    descriptor_b = {
        "collection": "Knight",
        "collections": ["Knight"],
        "frame_size": [2, 3],
        "animation": "Idle",
        "sprites": [{"path": "Idle/0.png"}],
    }
    descriptor_c = {
        "collection": "Knight",
        "collections": ["Knight"],
        "frame_size": [4, 1],
        "animation": "Jump",
        "sprites": [{"path": "Jump/0.png"}],
    }

    merged_descriptor, merged_frames = nodes.CKMergeEdits().merge(
        frames_a,
        descriptor_a,
        frames_b,
        descriptor_b,
        frames_3=frames_c,
        ck_frames_3=descriptor_c,
    )

    assert merged_frames.shape == (3, 3, 4, 4)
    assert nodes.torch.all(merged_frames[0, :2, :3, 3] == 0.25)
    assert nodes.torch.all(merged_frames[1, :3, :2, 3] == 0.75)
    assert nodes.torch.all(merged_frames[2, :1, :4, 3] == 0.5)
    assert nodes.torch.all(merged_frames[0, 2, :, :] == 0)
    assert nodes.torch.all(merged_frames[1, :, 2:, :] == 0)
    assert nodes.torch.all(merged_frames[2, 1:, :, :] == 0)
    assert merged_descriptor["frame_size"] == [4, 3]
    assert merged_descriptor["animation"] == "Walk+Idle+Jump"


def test_merge_edits_can_merge_multiple_collections():
    frames_a = nodes.torch.ones((1, 2, 2, 4))
    frames_b = nodes.torch.ones((1, 2, 2, 4))
    descriptor_a = {
        "root_folders": "Skin",
        "basepath": "Skin",
        "collection": "atlas0",
        "collections": ["atlas0"],
        "frame_size": [2, 2],
        "animation": "Walk",
        "sprites": [{"collection": "atlas0", "path": "Walk/0.png"}],
    }
    descriptor_b = {
        "root_folders": "Skin",
        "basepath": "Skin",
        "collection": "atlas1",
        "collections": ["atlas1"],
        "frame_size": [2, 2],
        "animation": "Idle",
        "sprites": [{"collection": "atlas1", "path": "Idle/0.png"}],
    }

    merged_descriptor, merged_frames = nodes.CKMergeEdits().merge(
        frames_a, descriptor_a, frames_b, descriptor_b
    )

    assert merged_frames.shape == (2, 2, 2, 4)
    assert merged_descriptor["collection"] is None
    assert merged_descriptor["collections"] == ["atlas0", "atlas1"]


def test_pack_frame_count_mismatch(tmp_path):
    root = _make_dump(tmp_path)
    selector = nodes.CKAnimationSelector()
    images, ck_frames = selector.select(root, "Knight", "Walk")

    packer = nodes.CKPackAtlas()
    with pytest.raises(ValueError, match="Frame count mismatch"):
        packer.pack(images[:1], ck_frames, save_to_output=False)


def test_ck_frames_descriptor_save_load_roundtrip(tmp_path):
    root = _make_dump(tmp_path)
    _images, ck_frames = nodes.CKAnimationSelector().select(root, "Knight", "Walk")
    path = tmp_path / "descriptors" / "walk.ck_frames.json"

    saved_descriptor, saved_path = nodes.CKSaveFramesDescriptor().save(
        ck_frames, str(path)
    )
    loaded_descriptor, loaded_path = nodes.CKLoadFramesDescriptor().load(str(path))

    assert saved_descriptor == ck_frames
    assert loaded_descriptor == ck_frames
    assert saved_path == str(path.resolve())
    assert loaded_path == str(path.resolve())


def test_save_load_frames_with_optional_descriptor_roundtrip(tmp_path):
    root = _make_dump(tmp_path)
    frames, ck_frames = nodes.CKAnimationSelector().select(root, "Knight", "Walk")
    parent = tmp_path / "saved"

    saved_frames, saved_descriptor, saved_path = nodes.CKSaveFrames().save(
        frames, str(parent), ck_frames=ck_frames
    )
    loaded_frames, loaded_descriptor, loaded_path = nodes.CKLoadFrames().load(saved_path)

    assert saved_path == str((parent / "Walk").resolve())
    assert loaded_path == saved_path
    assert (parent / "Walk" / "ck_frames.json").is_file()
    assert (parent / "Walk" / "0.png").is_file()
    assert (parent / "Walk" / "1.png").is_file()
    assert not (parent / "Walk" / "frame_00000.png").exists()
    assert nodes.torch.equal(saved_frames, frames)
    assert nodes.torch.equal(loaded_frames, frames)
    assert saved_descriptor == ck_frames
    assert loaded_descriptor == ck_frames


def test_save_load_frames_without_descriptor(tmp_path):
    frames = nodes.torch.zeros((2, 3, 4, 4), dtype=nodes.torch.float32)
    frames[1, ..., 3] = 1.0
    parent = tmp_path / "saved"

    _saved_frames, saved_descriptor, saved_path = nodes.CKSaveFrames().save(
        frames, str(parent)
    )
    loaded_frames, loaded_descriptor, loaded_path = nodes.CKLoadFrames().load(saved_path)

    assert saved_path == str((parent / "frames").resolve())
    assert loaded_path == saved_path
    assert saved_descriptor is None
    assert loaded_descriptor is None
    assert not (parent / "frames" / "ck_frames.json").exists()
    assert nodes.torch.equal(loaded_frames, frames)


def test_selector_frames_sheet_roundtrip_is_exact(tmp_path):
    root = _make_dump(tmp_path)
    frames, _ck_frames = nodes.CKAnimationSelector().select(
        root, "Knight", "Walk"
    )

    sheet, layout = nodes.CKFramesToSheet().combine(frames, columns=1)
    restored, = nodes.CKSheetToFrames().split(sheet, layout)

    assert sheet.shape == (1, 1488, 448, 4)
    assert sheet.shape[1] % 16 == 0
    assert sheet.shape[2] % 16 == 0
    assert 655_360 <= sheet.shape[1] * sheet.shape[2] <= 8_294_400
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

    assert sheet.shape == (1, 544, 1216, 4)
    assert layout == {
        "version": 3,
        "frame_count": 5,
        "frame_height": 2,
        "frame_width": 3,
        "channels": 4,
        "columns": 3,
        "rows": 2,
        "sheet_height": 544,
        "sheet_width": 1216,
    }
    assert nodes.torch.equal(restored, frames)
    # The unused final grid cell is transparent/black rather than duplicated.
    assert nodes.torch.count_nonzero(sheet[:, 2:4, 6:9, :]) == 0
    # Right/bottom alignment padding is transparent/black too.
    assert nodes.torch.count_nonzero(sheet[:, 4:, :, :]) == 0
    assert nodes.torch.count_nonzero(sheet[:, :, 9:, :]) == 0


def test_sheet_split_rejects_a_resized_sheet():
    frames = nodes.torch.zeros((2, 4, 5, 3))
    sheet, layout = nodes.CKFramesToSheet().combine(frames)

    with pytest.raises(ValueError, match="Sheet shape mismatch"):
        nodes.CKSheetToFrames().split(sheet[:, :-1], layout)


def test_frames_sheet_rejects_a_grid_above_custom_resolution_maximum():
    frames = nodes.torch.zeros((1, 2881, 2881, 1), dtype=nodes.torch.uint8)

    with pytest.raises(ValueError, match="exceeding the 8,294,400 maximum"):
        nodes.CKFramesToSheet().combine(frames)


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
    images, ck_frames = selector.select(
        root, "", "", mode="animation range", animation_range="1-2"
    )

    # Three frames padded to the largest (12 x 20).
    assert images.shape == (3, 20, 12, 4)
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
    images, ck_frames = selector.select(
        root, "", "", mode="animation range", animation_range="1-2"
    )

    ext_dir = tmp_path / "skin"
    packer = nodes.CKPackAtlas()
    out = packer.pack(
        images,
        ck_frames,
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
