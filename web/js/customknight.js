/**
 * CustomKnight Creator - web extension for the CKAnimationSelector node.
 *
 * Adds, like the original desktop tool:
 *   - dynamic "collection" and "animation" dropdowns driven by the typed
 *     Root Folders (cascading: root folders -> collections -> animations),
 *   - a "Refresh" button to (re)load collections from disk,
 *   - an animated preview that plays the selected animation's frames, with
 *     play/pause and a frame scrubber.
 *
 * The preview is a <canvas> mounted as a DOM widget (addDOMWidget) so it
 * renders under both the classic LiteGraph canvas renderer and the Vue-based
 * "Modern Node Design" (Nodes 2.0) renderer, keeping a constant aspect ratio
 * as the node is resized. (A legacy canvas-drawn widget is kept as a fallback
 * for ComfyUI builds old enough to lack addDOMWidget.)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLACEHOLDER = "<refresh>";
const SELECTOR_NODE_NAME = "CKAnimationSelector";
const MERGE_NODE_NAME = "CKMergeEdits";

// Preview box height as a fraction of its width. Keeping this constant means
// the preview area's aspect ratio stays fixed when the node is resized.
const PREVIEW_ASPECT = 1.0;
const PREVIEW_MIN_HEIGHT = 120;
const PREVIEW_MARGIN = 8;

async function getJSON(url, params) {
  const qs = new URLSearchParams(params).toString();
  const res = await api.fetchApi(`${url}?${qs}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

function imageURL(rootFolders, relPath) {
  const qs = new URLSearchParams({
    root_folders: rootFolders,
    path: relPath,
  }).toString();
  return api.apiURL(`/customknight/image?${qs}`);
}

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/** Replace a combo widget's option list and keep a sensible current value. */
function setComboValues(widget, values, keepValue) {
  if (!widget) return;
  const list = values && values.length ? values : [PLACEHOLDER];
  widget.options.values = list;
  if (!(keepValue && list.includes(widget.value))) {
    widget.value = list[0];
  }
}

// --- transparency checkerboard pattern -------------------------------------
// A CanvasPattern belongs to the context that created it, and each DOM-widget
// node owns its own canvas/context, so cache one pattern per context (the
// 16x16 source tile is built once and shared).
let _checkerTile = null;
const _checkerPatterns = new WeakMap();
function checkerPattern(ctx) {
  let pattern = _checkerPatterns.get(ctx);
  if (pattern) return pattern;
  if (!_checkerTile) {
    const c = document.createElement("canvas");
    c.width = c.height = 16;
    const cx = c.getContext("2d");
    cx.fillStyle = "#3a3a3a";
    cx.fillRect(0, 0, 16, 16);
    cx.fillStyle = "#2b2b2b";
    cx.fillRect(0, 0, 8, 8);
    cx.fillRect(8, 8, 8, 8);
    _checkerTile = c;
  }
  pattern = ctx.createPattern(_checkerTile, "repeat");
  _checkerPatterns.set(ctx, pattern);
  return pattern;
}

app.registerExtension({
  name: "CustomKnight.Nodes",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === SELECTOR_NODE_NAME) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        setupSelectorNode(this);
      };
    }

    if (nodeData.name === MERGE_NODE_NAME) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        setupMergeNode(this);
      };
    }
  },
});

function setupMergeNode(node) {
  if (node._ckMergeControlsReady) return;
  node._ckMergeControlsReady = true;
  node.addWidget("button", "Add edit pair", null, () => addMergeEditPair(node), {
    serialize: false,
  });
}

function addMergeEditPair(node) {
  const index = nextMergePairIndex(node);
  node.addInput(`frames_${index}`, "IMAGE");
  node.addInput(`ck_frames_${index}`, "CK_FRAMES");

  if (node.computeSize && node.setSize) {
    const [width, height] = node.computeSize();
    node.setSize([Math.max(node.size?.[0] ?? width, width), height]);
  }
  app.graph?.setDirtyCanvas(true, true);
}

function nextMergePairIndex(node) {
  let maxIndex = 2;
  for (const input of node.inputs || []) {
    const match = /^(?:ck_frames|frames|images)_(\d+)$/.exec(input.name || "");
    if (match) maxIndex = Math.max(maxIndex, Number(match[1]));
  }
  return maxIndex + 1;
}

function setupSelectorNode(node) {
  const rootWidget = findWidget(node, "root_folders");
  const collectionWidget = findWidget(node, "collection");
  const animationWidget = findWidget(node, "animation");

  node._ckPreview = {
    frames: [],
    images: [],
    index: 0,
    playing: true,
    fps: 12,
    timer: null,
  };

  // --- control widgets -----------------------------------------------------
  node.addWidget("button", "Refresh", null, () => reloadCollections(node), {
    serialize: false,
  });

  const playWidget = node.addWidget(
    "toggle",
    "play",
    true,
    (v) => {
      node._ckPreview.playing = v;
    },
    { on: "playing", off: "paused", serialize: false }
  );

  node.addWidget(
    "slider",
    "frame",
    0,
    (v) => {
      const p = node._ckPreview;
      if (p.images.length) {
        p.index = Math.max(0, Math.min(p.images.length - 1, Math.round(v)));
        p.playing = false;
        playWidget.value = false;
        renderPreview(node);
      }
    },
    { min: 0, max: 0, step: 1, precision: 0, serialize: false }
  );
  node._ckFrameWidget = findWidget(node, "frame");

  // --- preview widget (added last so it sits at the bottom of the node) ----
  addPreviewWidget(node);

  // --- wire change handlers ------------------------------------------------
  hookWidget(collectionWidget, () => reloadAnimations(node));
  hookWidget(animationWidget, () => reloadFrames(node));
  // Switching mode or editing the range re-loads the preview.
  hookWidget(findWidget(node, "mode"), () => reloadFrames(node));
  hookWidget(findWidget(node, "animation_range"), () => reloadFrames(node));

  startPreviewLoop(node);

  // The multiline textarea (inputEl) may be created lazily, so attach the
  // change/blur listeners on the next tick. Also auto-load once if the root
  // folders were restored from a saved graph, and normalise the node height
  // (fixes nodes that were left oversized by an earlier build). The Refresh
  // button always works regardless.
  setTimeout(() => {
    const el = rootWidget?.inputEl;
    if (el && !el._ckHooked) {
      el._ckHooked = true;
      el.addEventListener("change", () => reloadCollections(node));
      el.addEventListener("blur", () => reloadCollections(node));
    }
    const rangeEl = findWidget(node, "animation_range")?.inputEl;
    if (rangeEl && !rangeEl._ckHooked) {
      rangeEl._ckHooked = true;
      rangeEl.addEventListener("change", () => reloadFrames(node));
      rangeEl.addEventListener("blur", () => reloadFrames(node));
    }
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
    renderPreview(node); // draw the placeholder before any frames load
    if (rootWidget?.value?.trim()) reloadCollections(node);
  }, 100);
}

// Reserve a preview area of constant aspect ratio under the node's width.
function previewHeightFor(width) {
  const w = width || PREVIEW_MIN_HEIGHT;
  return Math.max(PREVIEW_MIN_HEIGHT, Math.round(w * PREVIEW_ASPECT));
}

function addPreviewWidget(node) {
  // Prefer a real DOM widget: the "Modern Node Design" (Nodes 2.0) renderer is
  // Vue/DOM based and no longer paints custom canvas-drawn widgets, so a plain
  // `draw(ctx)` widget shows up blank/broken there. A `<canvas>` mounted via
  // addDOMWidget renders correctly under BOTH the classic and Nodes 2.0
  // renderers. Fall back to the legacy canvas widget only on very old ComfyUI.
  if (node.addDOMWidget) {
    // ComfyUI lays out (and, in Nodes 2.0, measures) the *widget element* to
    // size the slot. Hand it a plain <div>: a <div> has no intrinsic size, so
    // nothing we paint can feed back into the layout. The <canvas> lives
    // *inside* it, absolutely positioned to fill it. Because the canvas is out
    // of flow, resizing its backing store never changes the <div>'s measured
    // height -- which is exactly what used to loop with the ResizeObserver and
    // make the preview slowly grow/shrink while zooming under Nodes 2.0.
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
      position: "relative",
      width: "100%",
      // Guarantee a visible height even if the Nodes 2.0 layout sizes the
      // widget slot by content rather than by computeSize().
      minHeight: `${PREVIEW_MIN_HEIGHT}px`,
      boxSizing: "border-box",
    });

    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
      position: "absolute",
      inset: "0",
      display: "block",
      borderRadius: "4px",
      backgroundColor: "#2b2b2b",
      // Non-interactive: let clicks/drags fall through to the node.
      pointerEvents: "none",
    });
    wrap.appendChild(canvas);
    node._ckCanvas = canvas;

    const widget = node.addDOMWidget("ck_preview", "ckpreview", wrap, {
      serialize: false,
      hideOnZoom: false,
    });
    widget.computeSize = function (width) {
      const w = width || node.size?.[0] || PREVIEW_MIN_HEIGHT;
      return [w, previewHeightFor(w)];
    };
    node._ckPreviewWidget = widget;

    // Repaint crisply whenever the slot (and thus the canvas) is resized.
    // Observe the wrapper, not the canvas: the wrapper's size is driven purely
    // by the layout, so its resize signal can't loop back through our paint.
    if (typeof ResizeObserver !== "undefined") {
      node._ckResizeObserver = new ResizeObserver(() => renderPreview(node));
      node._ckResizeObserver.observe(wrap);
    }
    return;
  }

  // --- legacy fallback: canvas-drawn LiteGraph widget ----------------------
  const widget = {
    name: "ck_preview",
    type: "ckpreview",
    value: "",
    options: { serialize: false },
    computeSize(width) {
      return [width, previewHeightFor(width)];
    },
    draw(ctx, n, width, y) {
      const x = PREVIEW_MARGIN;
      const w = width - PREVIEW_MARGIN * 2;
      const h = this.computeSize(width)[1] - PREVIEW_MARGIN;
      if (w <= 0 || h <= 0) return;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.translate(x, y);
      drawPreviewInto(ctx, node, w, h);
      ctx.restore();
    },
  };
  if (node.addCustomWidget) node.addCustomWidget(widget);
  else (node.widgets ||= []).push(widget);
  node._ckPreviewWidget = widget;
}

function hookWidget(widget, cb) {
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function () {
    const r = original?.apply(this, arguments);
    cb();
    return r;
  };
}

async function reloadCollections(node) {
  const rootWidget = findWidget(node, "root_folders");
  const collectionWidget = findWidget(node, "collection");
  const root = rootWidget?.value || "";
  if (!root.trim()) return;
  try {
    const { collections } = await getJSON("/customknight/collections", {
      root_folders: root,
    });
    setComboValues(collectionWidget, collections, true);
    app.graph.setDirtyCanvas(true, true);
    await reloadAnimations(node);
  } catch (e) {
    toast(node, `Collections: ${e.message}`);
  }
}

async function reloadAnimations(node) {
  const root = findWidget(node, "root_folders")?.value || "";
  const collection = findWidget(node, "collection")?.value || "";
  const animationWidget = findWidget(node, "animation");
  if (!root.trim() || !collection || collection === PLACEHOLDER) return;
  try {
    const { animations } = await getJSON("/customknight/animations", {
      root_folders: root,
      collection,
    });
    setComboValues(animationWidget, animations, true);
    app.graph.setDirtyCanvas(true, true);
    await reloadFrames(node);
  } catch (e) {
    toast(node, `Animations: ${e.message}`);
  }
}

async function reloadFrames(node) {
  const root = findWidget(node, "root_folders")?.value || "";
  const mode = findWidget(node, "mode")?.value || "single animation";

  // "animation range" mode: preview the concatenated frames of every animation
  // whose folder-name number is in the range, regardless of collection.
  if (mode === "animation range") {
    const range = findWidget(node, "animation_range")?.value || "";
    if (!root.trim() || !range.trim()) {
      setFrames(node, [], root);
      return;
    }
    try {
      const { frames } = await getJSON("/customknight/range_frames", {
        root_folders: root,
        range,
      });
      setFrames(node, frames, root);
    } catch (e) {
      toast(node, `Frames: ${e.message}`);
    }
    return;
  }

  const collection = findWidget(node, "collection")?.value || "";
  const animation = findWidget(node, "animation")?.value || "";
  if (!animation || animation === PLACEHOLDER) return;
  try {
    const { frames } = await getJSON("/customknight/frames", {
      root_folders: root,
      collection,
      animation,
    });
    setFrames(node, frames, root);
  } catch (e) {
    toast(node, `Frames: ${e.message}`);
  }
}

/** Load `frames` (path/name/w/h list) into the preview and reset the scrubber. */
function setFrames(node, frames, root) {
  const p = node._ckPreview;
  p.frames = frames || [];
  p.index = 0;
  p.images = p.frames.map((f) => {
    const img = new Image();
    img.onload = () => renderPreview(node);
    img.src = imageURL(root, f.path);
    return img;
  });
  if (node._ckFrameWidget) {
    node._ckFrameWidget.options.max = Math.max(0, p.frames.length - 1);
    node._ckFrameWidget.value = 0;
  }
  renderPreview(node);
}

function startPreviewLoop(node) {
  const p = node._ckPreview;
  if (p.timer) clearInterval(p.timer);
  p.timer = setInterval(() => {
    if (!p.playing || p.images.length === 0) return;
    p.index = (p.index + 1) % p.images.length;
    if (node._ckFrameWidget) node._ckFrameWidget.value = p.index;
    renderPreview(node);
  }, 1000 / p.fps);

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    if (p.timer) clearInterval(p.timer);
    node._ckResizeObserver?.disconnect();
    onRemoved?.apply(this, arguments);
  };
}

/** Repaint the preview for the current frame.
 *
 * With a DOM widget we own a real <canvas> and draw straight onto it. With the
 * legacy canvas widget we just flag the LiteGraph canvas dirty and its `draw`
 * handler calls drawPreviewInto for us.
 */
function renderPreview(node) {
  const canvas = node._ckCanvas;
  if (!canvas) {
    node.setDirtyCanvas(true, false);
    return;
  }
  // Match the backing store to the element's displayed size for crisp output.
  const cw = Math.max(1, Math.round(canvas.clientWidth || canvas.width || 1));
  const ch = Math.max(1, Math.round(canvas.clientHeight || canvas.height || 1));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  drawPreviewInto(canvas.getContext("2d"), node, cw, ch);
}

/** Draw the current frame into the rect (0, 0, w, h) of `ctx`. */
function drawPreviewInto(ctx, node, w, h) {
  const p = node._ckPreview;

  // transparency checkerboard
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = checkerPattern(ctx) || "#2b2b2b";
  ctx.fillRect(0, 0, w, h);

  const img = p.images[p.index];
  if (!img || !img.complete || !img.naturalWidth) {
    ctx.fillStyle = "#aaa";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("no animation selected", w / 2, h / 2);
    return;
  }

  // Fit the frame inside the rect, preserving aspect ratio.
  const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  ctx.imageSmoothingEnabled = false;
  try {
    ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  } catch (e) {
    /* image not ready yet */
  }

  // frame counter overlay
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, h - 16, w, 16);
  ctx.fillStyle = "#fff";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const name = p.frames[p.index]?.name || "";
  ctx.fillText(`${p.index + 1}/${p.images.length}  ${name}`, 5, h - 8);
}

function toast(node, message) {
  console.warn("[CustomKnight]", message);
  if (app.extensionManager?.toast) {
    app.extensionManager.toast.add({
      severity: "warn",
      summary: "CustomKnight",
      detail: message,
      life: 4000,
    });
  }
}
