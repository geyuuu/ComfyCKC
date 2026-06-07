/**
 * CustomKnight Creator - web extension for the CKAnimationSelector node.
 *
 * Adds, like the original desktop tool:
 *   - dynamic "collection" and "animation" dropdowns driven by the typed
 *     Root Folders (cascading: root folders -> collections -> animations),
 *   - a "Refresh" button to (re)load collections from disk,
 *   - an animated preview that plays the selected animation's frames, with
 *     play/pause and a frame scrubber.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLACEHOLDER = "<refresh>";
const NODE_NAME = "CKAnimationSelector";

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
  if (keepValue && list.includes(widget.value)) {
    // leave as-is
  } else {
    widget.value = list[0];
  }
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

  // --- preview canvas (DOM widget) ----------------------------------------
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  canvas.style.width = "100%";
  canvas.style.borderRadius = "6px";
  canvas.style.background =
    "repeating-conic-gradient(#444 0% 25%, #333 0% 50%) 50% / 16px 16px";
  canvas.style.imageRendering = "pixelated";

  const previewWidget = node.addDOMWidget("ck_preview", "preview", canvas, {
    serialize: false,
    hideOnZoom: false,
  });
  previewWidget.computeSize = function (width) {
    return [width, 220];
  };
  node._ckCanvas = canvas;

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

  const frameWidget = node.addWidget(
    "slider",
    "frame",
    0,
    (v) => {
      const p = node._ckPreview;
      if (p.images.length) {
        p.index = Math.max(0, Math.min(p.images.length - 1, Math.round(v)));
        p.playing = false;
        playWidget.value = false;
        drawPreview(node);
      }
    },
    { min: 0, max: 0, step: 1, precision: 0, serialize: false }
  );
  node._ckFrameWidget = frameWidget;

  // --- wire change handlers ------------------------------------------------
  hookWidget(collectionWidget, () => reloadAnimations(node));
  hookWidget(animationWidget, () => reloadFrames(node));

  // Reload collections when the user finishes editing the root folders box.
  if (rootWidget?.inputEl) {
    rootWidget.inputEl.addEventListener("change", () => reloadCollections(node));
    rootWidget.inputEl.addEventListener("blur", () => reloadCollections(node));
  }

  startPreviewLoop(node);

  // Populate on next tick if root folders were restored from a saved graph.
  setTimeout(() => {
    if (rootWidget?.value?.trim()) reloadCollections(node);
  }, 50);
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
      img.src = imageURL(root, f.path);
      return img;
    });
    node._ckFrameWidget.options.max = Math.max(0, frames.length - 1);
    node._ckFrameWidget.value = 0;
    drawPreview(node);
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
    drawPreview(node);
  }, 1000 / p.fps);

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    if (p.timer) clearInterval(p.timer);
    onRemoved?.apply(this, arguments);
  };
}

function drawPreview(node) {
  const canvas = node._ckCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const p = node._ckPreview;
  const img = p.images[p.index];

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!img || !img.complete || !img.naturalWidth) {
    ctx.fillStyle = "#aaa";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("no animation selected", canvas.width / 2, canvas.height / 2);
    return;
  }

  // Fit the frame inside the canvas, preserving aspect ratio.
  const scale = Math.min(
    canvas.width / img.naturalWidth,
    canvas.height / img.naturalHeight
  );
  const w = img.naturalWidth * scale;
  const h = img.naturalHeight * scale;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, canvas.height - 18, canvas.width, 18);
  ctx.fillStyle = "#fff";
  ctx.font = "12px monospace";
  ctx.textAlign = "left";
  const name = p.frames[p.index]?.name || "";
  ctx.fillText(`${p.index + 1}/${p.images.length}  ${name}`, 6, canvas.height - 5);
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
