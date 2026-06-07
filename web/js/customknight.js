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
 * The preview is drawn as a LiteGraph custom widget (not a DOM element), so it
 * is laid out and clipped by the node itself and keeps a constant aspect ratio
 * as the node is resized.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLACEHOLDER = "<refresh>";
const NODE_NAME = "CKAnimationSelector";

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

// --- transparency checkerboard pattern (created lazily, cached) ------------
let _checker = null;
function checkerPattern(ctx) {
  if (_checker) return _checker;
  const c = document.createElement("canvas");
  c.width = c.height = 16;
  const cx = c.getContext("2d");
  cx.fillStyle = "#3a3a3a";
  cx.fillRect(0, 0, 16, 16);
  cx.fillStyle = "#2b2b2b";
  cx.fillRect(0, 0, 8, 8);
  cx.fillRect(8, 8, 8, 8);
  _checker = ctx.createPattern(c, "repeat");
  return _checker;
}

app.registerExtension({
  name: "CustomKnight.AnimationSelector",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      setupSelectorNode(this);
    };
  },
});

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
        node.setDirtyCanvas(true, false);
      }
    },
    { min: 0, max: 0, step: 1, precision: 0, serialize: false }
  );
  node._ckFrameWidget = findWidget(node, "frame");

  // --- preview widget (drawn by LiteGraph, last so it sits at the bottom) ---
  addPreviewWidget(node);

  // --- wire change handlers ------------------------------------------------
  hookWidget(collectionWidget, () => reloadAnimations(node));
  hookWidget(animationWidget, () => reloadFrames(node));

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
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
    if (rootWidget?.value?.trim()) reloadCollections(node);
  }, 100);
}

function addPreviewWidget(node) {
  const widget = {
    name: "ck_preview",
    type: "ckpreview",
    value: "",
    options: { serialize: false },
    computeSize(width) {
      const h = Math.max(PREVIEW_MIN_HEIGHT, Math.round(width * PREVIEW_ASPECT));
      return [width, h];
    },
    draw(ctx, n, width, y) {
      const x = PREVIEW_MARGIN;
      const w = width - PREVIEW_MARGIN * 2;
      const h = this.computeSize(width)[1] - PREVIEW_MARGIN;
      if (w <= 0 || h <= 0) return;
      drawPreview(ctx, node, x, y, w, h);
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
  const collection = findWidget(node, "collection")?.value || "";
  const animation = findWidget(node, "animation")?.value || "";
  if (!animation || animation === PLACEHOLDER) return;
  try {
    const { frames } = await getJSON("/customknight/frames", {
      root_folders: root,
      collection,
      animation,
    });
    const p = node._ckPreview;
    p.frames = frames;
    p.index = 0;
    p.images = frames.map((f) => {
      const img = new Image();
      img.onload = () => node.setDirtyCanvas(true, false);
      img.src = imageURL(root, f.path);
      return img;
    });
    if (node._ckFrameWidget) {
      node._ckFrameWidget.options.max = Math.max(0, frames.length - 1);
      node._ckFrameWidget.value = 0;
    }
    node.setDirtyCanvas(true, false);
  } catch (e) {
    toast(node, `Frames: ${e.message}`);
  }
}

function startPreviewLoop(node) {
  const p = node._ckPreview;
  if (p.timer) clearInterval(p.timer);
  p.timer = setInterval(() => {
    if (!p.playing || p.images.length === 0) return;
    p.index = (p.index + 1) % p.images.length;
    if (node._ckFrameWidget) node._ckFrameWidget.value = p.index;
    node.setDirtyCanvas(true, false);
  }, 1000 / p.fps);

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    if (p.timer) clearInterval(p.timer);
    onRemoved?.apply(this, arguments);
  };
}

/** Draw the current frame into the rect (x, y, w, h) in node-local coords. */
function drawPreview(ctx, node, x, y, w, h) {
  const p = node._ckPreview;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();

  // transparency checkerboard
  ctx.fillStyle = checkerPattern(ctx) || "#2b2b2b";
  ctx.fillRect(x, y, w, h);

  const img = p.images[p.index];
  if (!img || !img.complete || !img.naturalWidth) {
    ctx.fillStyle = "#aaa";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("no animation selected", x + w / 2, y + h / 2);
    ctx.restore();
    return;
  }

  // Fit the frame inside the rect, preserving aspect ratio.
  const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  ctx.imageSmoothingEnabled = false;
  try {
    ctx.drawImage(img, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
  } catch (e) {
    /* image not ready yet */
  }

  // frame counter overlay
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(x, y + h - 16, w, 16);
  ctx.fillStyle = "#fff";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const name = p.frames[p.index]?.name || "";
  ctx.fillText(`${p.index + 1}/${p.images.length}  ${name}`, x + 5, y + h - 8);

  ctx.restore();
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
