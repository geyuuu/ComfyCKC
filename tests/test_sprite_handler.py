"""Round-trip tests for the core sprite logic (no torch / ComfyUI needed)."""

import json
import os

from PIL import Image

from customknight_creator.sprite_handler import (
    SpriteProject,
    atlas_size_for,
    pack_collection,
    parse_root_folders,
)


def _make_dump(tmp_path):
    """Create a tiny synthetic CustomKnight dump and return the root folder."""
    base = tmp_path / "Skin"
    root = base / "Knight"
    atlases = root / "0.Atlases"
    atlases.mkdir(parents=True)

    # Two animations, three frames, in one collection.
    walk = root / "Walk"
    idle = root / "Idle"
    walk.mkdir()
    idle.mkdir()

    # Source PNGs: each is exactly its sprite content (xr=yr=0).
    Image.new("RGBA", (10, 20), (255, 0, 0, 255)).save(walk / "0.png")
    Image.new("RGBA", (10, 20), (0, 255, 0, 255)).save(walk / "1.png")
    Image.new("RGBA", (30, 8), (0, 0, 255, 255)).save(idle / "0.png")

    info = {
        "sid": ["w0", "w1", "i0"],
        "sx": [0, 16, 0],
        "sy": [0, 0, 24],
        "sxr": [0, 0, 0],
        "syr": [0, 0, 0],
        "swidth": [10, 10, 30],
        "sheight": [20, 20, 8],
        "sfilpped": [False, False, True],  # idle frame is rotated
        "spath": ["Knight/Walk/0.png", "Knight/Walk/1.png", "Knight/Idle/0.png"],
        "scollectionname": ["Knight", "Knight", "Knight"],
    }
    (atlases / "SpriteInfo.json").write_text(json.dumps(info), encoding="utf-8")
    return str(root)


def test_parse_root_folders():
    assert parse_root_folders('a\n"b"\n\n a ') == [os.path.normpath("a"), os.path.normpath("b")]


def test_project_queries(tmp_path):
    root = _make_dump(tmp_path)
    project = SpriteProject.from_root_folders(root)

    assert project.collections() == ["Knight"]
    assert set(project.animations("Knight")) == {"Walk", "Idle"}
    assert [s.filename for s in project.sprites_in_animation("Walk", "Knight")] == [
        "0.png",
        "1.png",
    ]


def test_atlas_size_is_power_of_two(tmp_path):
    root = _make_dump(tmp_path)
    project = SpriteProject.from_root_folders(root)
    w, h = atlas_size_for(project.sprites_in_collection("Knight"))
    # max right = 16 + 10 = 26 -> 32 ; idle flipped top = 24 + 30 = 54 -> 64
    assert (w, h) == (32, 64)


def test_pack_places_sprites(tmp_path):
    root = _make_dump(tmp_path)
    project = SpriteProject.from_root_folders(root)
    result = pack_collection(project, "Knight")

    assert result.image.size == (32, 64)
    assert result.sprite_count == 3
    assert result.replaced_count == 0

    atlas = result.image
    H = atlas.size[1]
    # Walk frame 0 sits at x=0, bottom-left origin y=0 -> top = H - 0 - 20.
    assert atlas.getpixel((1, H - 1)) == (255, 0, 0, 255)
    # Walk frame 1 sits at x=16.
    assert atlas.getpixel((17, H - 1)) == (0, 255, 0, 255)


def test_pack_with_replacement(tmp_path):
    root = _make_dump(tmp_path)
    project = SpriteProject.from_root_folders(root)

    replacement = Image.new("RGBA", (10, 20), (123, 45, 67, 255))
    result = pack_collection(
        project, "Knight", replacements={"Knight/Walk/0.png": replacement}
    )

    assert result.replaced_count == 1
    H = result.image.size[1]
    assert result.image.getpixel((1, H - 1)) == (123, 45, 67, 255)
