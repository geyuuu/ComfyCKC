# ComfyUI CustomKnight Creator

![ComfyUI CustomKnight Creator](assets/ckc.gif)

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

**Animation-range mode.** Set `mode` to **`animation range`** and the node
ignores the collection/animation dropdowns and instead concatenates *every*
animation whose folder-name number falls in `animation_range` into one
sequence. The number is the prefix CustomKnight / Silksong dumps put on each
animation folder (e.g. `034.Idle` → `34`). Accepts inclusive ranges and lists:

```
1-10        animations 001..010
34          just animation 034
1-3,5,7-9   mix ranges and singles
```

Frames are emitted in animation order (then SpriteInfo order within each), and
the live preview plays the whole concatenated clip. Because one numbered
animation can be packed into more than one atlas, a range commonly spans
**several collections** — that's fine, the frames carry their own collection
and **CK Pack Atlas** rebuilds one atlas per collection (see below).

**Outputs**

| Output | Type | Description |
| --- | --- | --- |
| `frames` | `IMAGE` | The animation's RGBA frames as a PNG image sequence (batch); alpha is the fourth channel. |
| `ck_frames` | `CK_FRAMES` | Layout descriptor used by the packer. |

Frames of different sizes are padded (top-left anchored, transparent) to a
common size so they fit in one batch; the original per-frame size is recorded
in `ck_frames` and restored on pack.

### CK Frames to Sheet / CK Sheet to Frames

**CK Frames to Sheet** arranges an `IMAGE` frame batch (including the `frames`
output of **CK Animation Selector**) into one grid image. `columns = 0` chooses
a compact near-square grid automatically; set it to `1` for a vertical strip,
or to the frame count for a horizontal strip. It outputs both the single
`sheet` image and a `sheet_layout` descriptor.

The sheet is transparently padded on its right and bottom edges so both its
width and height are multiples of 16 and its total pixel count is between
**655,360 and 8,294,400**. When the frame grid is smaller than the minimum, both
canvas dimensions grow while approximately preserving the grid's aspect ratio.
No frame pixels are resized or moved. If the aligned frame grid itself exceeds
the maximum, the node reports an error because it cannot meet the limit without
losing pixels; changing `columns`, reducing the frame count, or reducing the
source frame size is then required.

Connect both outputs to **CK Sheet to Frames** to recover the original ordered
`IMAGE` batch. A direct combine/split round trip does not resize, colour-convert,
or quantise pixels, so the recovered frames have exactly the same shape, dtype,
device, ordering, and values as the selector output. If the sheet is resized or
cropped between the two nodes, the splitter reports a clear shape-mismatch error
instead of silently producing incorrect frames.

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
- `override_width` / `override_height` force exact atlas dimensions if needed
  (single-collection only; ignored when the input spans several collections).

Transparency is carried in the fourth channel of `frames`. If an image operation
drops that channel and sends RGB frames to the packer, the original sprite's
transparency is restored automatically.

When the input frames span **multiple collections** (from the selector's
`animation range` mode, possibly via **CK Merge Edits**), the node groups the
frames by collection and writes **one atlas PNG per collection** — saved as
`<prefix>_<collection>.png` in `output/` and/or `<collection>.png` in
`external_directory`. The `atlas` output then carries all of them as one image
batch, and `saved_path` lists every written path (newline-separated).

### CK Merge Edits
Combine two edited animations from the **same** collection into one
`frames` + `ck_frames` pair so they pack into a single atlas. RGBA transparency
stays inside `frames`; there are no separate alpha inputs or output. Chain
several of these to edit many animations at once.

### CK Project Info
Inspect a dump: prints the base path, every collection (with computed atlas
size) and its animations / frame counts. Handy for finding the right names.

### CK Video Combine
Turn an `IMAGE` batch (e.g. the `frames` output of **CK Animation Selector**, or
any edited image sequence) into a **video — right inside the graph**. It's a port
of [VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite)'s
*Video Combine* node, but instead of saving a file to `output/` it outputs a
native ComfyUI **`VIDEO`** object you can wire into the built-in **Save Video** /
preview nodes.

**Inputs**

| Input | Type | Description |
| --- | --- | --- |
| `images` | `IMAGE` | The frame sequence (batch) to encode. |
| `frame_rate` | `FLOAT` | Playback FPS. |
| `loop_count` | `INT` | Extra loops (`0` = play once). |
| `filename_prefix` | `STRING` | Temp filename prefix (used for the preview file). |
| `format` | combo | `image/gif`, `image/webp`, or a video container: `video/h264-mp4`, `video/h265-mp4`, `video/webm`, `video/av1-webm`, `video/ffmpeg-gif`, `video/ProRes`, `video/ffv1-mkv`. |
| `pingpong` | `BOOLEAN` | Append the sequence reversed so it ping-pongs. |
| `background_color` | `STRING` | Hex colour (e.g. `#000000`) that transparent (RGBA) frames are flattened onto. Ignored for opaque RGB input or when `preserve_alpha` keeps it. |
| `preserve_alpha` | `BOOLEAN` | Keep real transparency for formats that can store it (`ProRes`, `ffv1-mkv`, `image/gif`, `image/webp`). Other formats composite onto `background_color`. |
| `audio` | `AUDIO` *(optional)* | Muxed into the video. |

**Output**

| Output | Type | Description |
| --- | --- | --- |
| `video` | `VIDEO` | The encoded clip, held **in memory** (no `output/` write). |

The clip is encoded into ComfyUI's **temp** folder (so it can be previewed on the
node and so ffmpeg can mux audio), then loaded into memory and wrapped as a
`VIDEO`; nothing is persisted to `output/` unless you connect a Save node.
Encoding uses ffmpeg for every format — including `image/gif` and `image/webp` —
so the chosen `frame_rate` is honoured exactly (Pillow drops animated-webp frame
timing, which made clips the wrong length). The bundled `imageio-ffmpeg` provides
the binary, or a system `ffmpeg` on `PATH` works too; if neither is present,
gif/webp fall back to Pillow. (Per-format options such as `crf`/`pix_fmt` use each
format's defaults.)

Transparent sprites (RGBA) are **alpha-composited onto `background_color`** before
encoding: most video codecs can't store transparency, and sprite atlases leave
arbitrary RGB in fully-transparent texels — without compositing that garbage RGB
leaks through as an ugly background. Set `background_color` to whatever you want
behind the sprite (default black). Opaque RGB input is encoded unchanged.

To keep **real transparency** instead, enable `preserve_alpha`. It applies only to
formats that can actually carry an alpha channel — `ProRes` (4444, `.mov`),
`ffv1-mkv` (`.mkv`), `image/gif` and `image/webp`. `h264`/`h265` mp4 and `av1`
have no alpha, and **webm can't either** — ffmpeg can *decode* VP8/VP9 alpha but
not *encode* it — so those always composite onto `background_color`. Note the
in-node `<video>` preview shows transparent areas against the page background, so
transparency is most obvious once you use the `VIDEO` downstream.

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
│   ├── server_routes.py        # /customknight/* endpoints
│   ├── video_combine.py        # CK Video Combine -> in-memory VIDEO (VHS port)
│   └── video_formats/          # ffmpeg format definitions (.json)
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
