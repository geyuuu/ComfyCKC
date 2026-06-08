# ComfyUI CustomKnight Creator

A ComfyUI custom-node port of
[CustomKnight-Creator](https://github.com/cmot17/CustomKnight-Creator) — the
desktop tool for building [Hollow Knight](https://www.hollowknight.com/)
[CustomKnight](https://github.com/Clazex/HollowKnight.CustomKnight) skins.

Instead of a standalone PyQt app, you now edit your skins **inside a ComfyUI
graph**: pick an animation, run the frames through any ComfyUI image
pipeline (recolour, palette swap, AI restyle, …), then repack everything back
into a game-ready atlas.

```
Root Folders ──▶ [CK Animation Selector] ──IMAGE──▶ (your edits) ──▶ [CK Pack Atlas] ──▶ atlas .png
                          │  ▲ live preview                                  ▲
                          └──CK_FRAMES──────────────────────────────────────┘
```

## Nodes

### CK Animation Selector
The core node. **Input is Root Folders** (one top-level dump folder per line,
each containing `0.Atlases/SpriteInfo.json`). Just like the original tool you:

- pick a **collection / atlas** (`scollectionname`),
- pick an **animation** (a folder of frames),
- watch a **live animated preview** on the node (play/pause + frame scrubber).

Dropdowns cascade automatically as you type/change Root Folders → collection →
animation (served from disk by the bundled HTTP routes).

**Outputs**

| Output | Type | Description |
| --- | --- | --- |
| `frames` | `IMAGE` | The animation's frames as a PNG image sequence (batch). |
| `alpha` | `MASK` | The frames' alpha channels (sprites are transparent). |
| `ck_frames` | `CK_FRAMES` | Layout descriptor used by the packer. |

Frames of different sizes are padded (top-left anchored, transparent) to a
common size so they fit in one batch; the original per-frame size is recorded
in `ck_frames` and restored on pack.

### CK Pack Atlas
Takes the (edited) `frames` + `ck_frames` and rebuilds the **entire** atlas:
modified frames are dropped in place, every other sprite of the collection is
read from disk unchanged, and the whole thing is packed — applying the same
crop / 90° rotation / flip and power-of-two sizing as the original packer.

- The `atlas` output is an RGBA `IMAGE`, so its transparency previews exactly
  like the saved PNG (no white boxes / colour noise from dropped alpha).
- A thumbnail always renders on the node. `save_to_output` only controls whether
  the PNG is persisted to ComfyUI's `output/` folder (on) or written to the
  temp folder for preview only (off).
- Optional `external_directory` also writes `<collection>.png` straight into
  your CustomKnight skin folder.
- `override_width` / `override_height` force exact atlas dimensions if needed.

If you don't feed an `alpha`/`MASK` back in, the original sprite's transparency
is preserved automatically.

### CK Merge Edits
Combine two edited animations from the **same** collection into one
`frames` + `ck_frames` pair so they pack into a single atlas. Chain several of
these to edit many animations at once.

### CK Project Info
Inspect a dump: prints the base path, every collection (with computed atlas
size) and its animations / frame counts. Handy for finding the right names.

## Installation

Clone into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> comfyui-customknight-creator
```

ComfyUI installs the requirements (`pillow`, `numpy`) on next launch; `torch`,
`aiohttp` and the server come from ComfyUI itself. Restart ComfyUI.

## Development (uv)

This project is managed with [uv](https://docs.astral.sh/uv/). The `dev`
dependency group adds `torch`, `pytest` and `ruff` so the nodes can be imported
and tested outside a ComfyUI install:

```bash
uv sync                # create .venv and install deps + dev group
uv run pytest          # run the tests
uv run ruff check .    # lint
```

`pillow` + `numpy` are the only runtime dependencies; everything else is
provided by the ComfyUI host environment.

## How it maps to the original

| Original (`spritehandler.py`)        | Here |
| --- | --- |
| `loadSpriteInfo` / categories         | `SpriteProject.collections()` |
| `loadAnimations` / `loadSprites`      | `SpriteProject.animations()` / `sprites_in_animation()` |
| crop box + `sfilpped` rotate/flip     | `SpriteProject.crop_content()` + `place_sprite()` |
| `packSprites` (power-of-two atlas)    | `pack_collection()` / `atlas_size_for()` |
| Qt preview + play button              | `web/js/customknight.js` canvas preview |

The data model is unchanged: `SpriteInfo.json` parallel arrays
(`sx/sy/sxr/syr/swidth/sheight/sfilpped/spath/scollectionname`), bottom-left
origin, atlas per collection.

## Layout

```
comfyui-customknight-creator/
├── __init__.py                 # ComfyUI entry: mappings + WEB_DIRECTORY
├── pyproject.toml              # uv project + [tool.comfy] registry metadata
├── requirements.txt
├── web/js/customknight.js      # dynamic dropdowns + animated preview
├── src/customknight_creator/
│   ├── __init__.py             # node mappings, registers HTTP routes
│   ├── nodes.py                # CK* node classes
│   ├── sprite_handler.py       # core port (no torch / ComfyUI deps)
│   └── server_routes.py        # /customknight/* endpoints
└── tests/
```

## License

This project is licensed under the **GNU General Public License v3.0**
(see [`LICENSE`](LICENSE)).

It is a derivative work — a node port — of
[CustomKnight-Creator](https://github.com/cmot17/CustomKnight-Creator) by
cmot17, which is itself licensed under the GPLv3. Because the GPL is a copyleft
license, this port inherits the same terms and cannot be relicensed under a
more permissive license such as MIT.

```
Copyright (C) 2026 ComfyUI CustomKnight Creator contributors
Copyright (C) cmot17 and the CustomKnight-Creator contributors (original work)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
```

CustomKnight and Hollow Knight are property of their respective owners.
