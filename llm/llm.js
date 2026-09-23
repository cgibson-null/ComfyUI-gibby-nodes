import { app } from "../../../scripts/app.js";

// Gibby Nodes - LLM Connect
// -------------------------
// Model-list population for the Connectivity node's "model" dropdown:
// fetches the server's /v1/models via this pack's /gibby_llm/get_models
// route (registered in the pack __init__), caches the result per URL in
// localStorage (10-minute staleness window), and auto-refreshes on
// url/api_key edits and on node create / graph load.
//
// The auto-growing image/video sockets on Generate are native
// (COMFY_AUTOGROW_V3) - no JS needed for those.

const CONNECTIVITY_NODE = "GibbyConnectivity";

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

async function fetchModels(url, apiKey) {
  const resp = await fetch("/gibby_llm/get_models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, api_key: apiKey || "" }),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  return Array.isArray(data) ? data : [];
}

const refreshSeq = new WeakMap();
const refreshTimer = new WeakMap();
const CACHE_KEY = "gibby_llm_models_";

function cacheGet(url) {
  try {
    const raw = localStorage.getItem(CACHE_KEY + url);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.models) || typeof parsed.fetchedAt !== "number") {
      return null;
    }
    // 10-minute staleness window: old enough that a model added to the
    // server config shows up without a manual refresh, young enough that
    // a stale list from a dead server is never trusted silently.
    if (Date.now() - parsed.fetchedAt > 10 * 60 * 1000) return null;
    return parsed.models;
  } catch {
    return null;
  }
}

function cacheSet(url, models) {
  try {
    localStorage.setItem(CACHE_KEY + url, JSON.stringify({ models, fetchedAt: Date.now() }));
  } catch {
    // Quota / private mode - the dropdown still works, it just won't cache.
  }
}

function applyModels(node, models) {
  const modelWidget = findWidget(node, "model");
  if (!modelWidget) return;
  modelWidget.options.values = models;
  const current = modelWidget.value;
  if (models.length && !models.includes(current)) {
    modelWidget.value = models[0];
    modelWidget.callback?.();
  }
  modelWidget.refresh?.();
  node.setDirtyCanvas(true, false);
}

async function updateModelsList(node) {
  const urlWidget = findWidget(node, "url");
  const modelWidget = findWidget(node, "model");
  if (!urlWidget || !modelWidget) return false;

  const url = urlWidget.value;
  const apiKey = findWidget(node, "api_key")?.value || "";
  const seq = (refreshSeq.get(node) || 0) + 1;
  refreshSeq.set(node, seq);

  try {
    const models = await fetchModels(url, apiKey);
    if (refreshSeq.get(node) !== seq) return false;
    cacheSet(url, models);
    applyModels(node, models);
    return models.length > 0;
  } catch {
    if (refreshSeq.get(node) !== seq) return false;
    modelWidget.options.values = [];
    modelWidget.value = "";
    node.setDirtyCanvas(true, false);
    return false;
  }
}

function debouncedAutoLoad(node) {
  clearTimeout(refreshTimer.get(node));
  refreshTimer.set(node, setTimeout(() => updateModelsList(node), 400));
}

app.registerExtension({
  name: "Gibby.LLMConnect",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CONNECTIVITY_NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      this.addWidget("button", "Reconnect", null, () => updateModelsList(this));
      for (const name of ["url", "api_key"]) {
        const w = findWidget(this, name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = (...args) => {
          const res = orig?.apply(w, args);
          debouncedAutoLoad(this);
          return res;
        };
      }
      const nodeId = this.id;
      setTimeout(() => {
        const node = app.graph?.nodes?.find((n) => n.id === nodeId) ?? this;
        const url = findWidget(node, "url")?.value;
        const cached = url ? cacheGet(url) : null;
        if (cached?.length) applyModels(node, cached);
        else updateModelsList(node);
      }, 3000);
      return r;
    };

    const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
    nodeType.prototype.onAfterGraphConfigured = function () {
      const r = onAfterGraphConfigured?.apply(this, arguments);
      const url = findWidget(this, "url")?.value;
      const cached = url ? cacheGet(url) : null;
      if (cached?.length) applyModels(this, cached);
      else updateModelsList(this);
      return r;
    };
  },
});
