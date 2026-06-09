"""Round-trip tests for the core sprite logic (no torch / ComfyUI needed)."""

import json
import os

import pytest
from PIL import Image

from customknight_creator.sprite_handler import (
    SpriteProject,
    animation_number,
    atlas_size_for,
    pack_collection,
    parse_index_range,
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


# ---------------------------------------------------------------------------
# Animation-range selection (numbered animation folders across collections)
# ---------------------------------------------------------------------------
def _make_numbered_dump(tmp_path):
    """A dump whose animations are number-prefixed folders spanning 2 atlases.

    Mirrors a real Silksong dump: a single numbered animation folder may have
    frames packed into more than one collection (atlas).
    """
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
        # 001.Run is split across two atlases; 002.Idle lives in atlas0.
        "scollectionname": ["atlas0", "atlas1", "atlas0"],
    }
    (atlases / "SpriteInfo.json").write_text(json.dumps(info), encoding="utf-8")
    return str(root)


def test_animation_number():
    assert animation_number("034.Idle") == 34
    assert animation_number("001.AirDash Burst") == 1
    assert animation_number("098.Hornet Superjump Charge Lace") == 98
    assert animation_number("Walk") is None
    assert animation_number("") is None


def test_parse_index_range():
    assert parse_index_range("1-10") == list(range(1, 11))
    assert parse_index_range("34") == [34]
    assert parse_index_range("1-3,5,7-9") == [1, 2, 3, 5, 7, 8, 9]
    assert parse_index_range("  3 , 1-2 ") == [1, 2, 3]  # dedup + sort
    assert parse_index_range("") == []


def test_parse_index_range_errors():
    with pytest.raises(ValueError, match="reversed"):
        parse_index_range("10-1")
    with pytest.raises(ValueError):
        parse_index_range("abc")


def test_sprites_in_animation_range_spans_collections(tmp_path):
    root = _make_numbered_dump(tmp_path)
    project = SpriteProject.from_root_folders(root)

    # Range 1-2 grabs both animations, ordered by animation number then
    # SpriteInfo order, and crosses both collections.
    sprites = project.sprites_in_animation_range([1, 2])
    assert [s.id for s in sprites] == ["r0", "r1", "i0"]
    assert [s.collection for s in sprites] == ["atlas0", "atlas1", "atlas0"]

    # A range that matches only animation 002.
    only_idle = project.sprites_in_animation_range([2])
    assert [s.id for s in only_idle] == ["i0"]

    # Nothing matches -> empty (and empty input -> empty).
    assert project.sprites_in_animation_range([99]) == []
    assert project.sprites_in_animation_range([]) == []
