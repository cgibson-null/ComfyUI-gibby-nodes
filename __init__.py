"""
Standalone Power Lora Loader
-----------------------------
A self-contained reimplementation of rgthree's "Power Lora Loader" node,
built on ComfyUI's modern V3 node schema, with a growable/shrinkable row
count via a small "lora slots" number field that updates live.

WHY IT'S BUILT THIS WAY:
The original Power Lora Loader draws its own custom widget (the rows, the
toggles, the "+ Add Lora" button) directly onto the old canvas system with
hand-written JavaScript, doing its own click/drag handling from scratch.
That's exactly the kind of custom node that breaks under Nodes 2.0.

This version instead uses a JS helper (see js/dynamic_rows.js) that builds
real HTML elements via ComfyUI's own node.addDOMWidget() function - not
hand-drawn canvas graphics. It isn't reinventing any rendering, which is
why it survives frontend changes that break fully hand-drawn widgets.

INSTALL:
Put this whole folder (including the js/ subfolder) in
ComfyUI/custom_nodes/, then restart ComfyUI.
"""

import hashlib
import json
import logging
import os
import struct
import urllib.error
import urllib.request
from asyncio import get_event_loop

import folder_paths
from nodes import LoraLoader
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./js"

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_JS_FILE_PATH = os.path.join(_NODE_DIR, "js", "dynamic_rows.js")
_FOLDER_NAME = os.path.basename(_NODE_DIR)
_INFO_CACHE_PATH = os.path.join(_NODE_DIR, "lora_info_cache.json")

# --- Serve dynamic_rows.js ourselves, with an explicit Content-Type -------
# Two earlier attempts tried to fix Windows' broken .js MIME-type guessing
# by patching the databases ComfyUI's web server consults - but there's no
# way to be fully sure which database (if any) is actually being read on
# any given setup. Rather than keep guessing at that indirectly, this
# registers a route for this exact file's URL and serves it ourselves, with
# the Content-Type set directly and explicitly. No guessing involved at
# all, so it can't be derailed by any registry/database problem.
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get(f"/extensions/{_FOLDER_NAME}/dynamic_rows.js")
    async def _serve_dynamic_rows_js(request):
        with open(_JS_FILE_PATH, "rb") as f:
            body = f.read()
        return web.Response(
            body=body, content_type="text/javascript", charset="utf-8"
        )

    logging.info(
        "[Standalone Power Lora Loader] Registered dedicated route for dynamic_rows.js"
    )
except Exception as e:
    logging.warning(
        f"[Standalone Power Lora Loader] Could not register dedicated JS route, "
        f"falling back to default static file serving: {e}"
    )


# --- Lora info / Civitai lookup -------------------------------------------
# Hashing is cached (keyed by file size + modified time) so a multi-GB lora
# only gets hashed once, and it runs in a background thread so hashing a
# large file doesn't freeze the rest of ComfyUI's server while it works.
# Civitai lookups (and any manual edits to Name/Strength/Notes) are cached
# in a small local JSON file, keyed by lora filename, so they're shared by
# every node instance and survive restarts.

def _load_info_cache():
    try:
        with open(_INFO_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_info_cache(cache):
    try:
        with open(_INFO_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logging.warning(f"[Standalone Power Lora Loader] Could not save info cache: {e}")


def _hash_file_sync(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_safetensors_metadata(path):
    """Reads just the small JSON header safetensors files store up front -
    not the (potentially multi-GB) tensor data after it."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return {}
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    return header.get("__metadata__", {}) or {}


def _extract_trained_words(metadata, top_n=50):
    """kohya_ss/sd-scripts (the tool most LoRA trainers use) embeds a tag
    frequency table (ss_tag_frequency) recording every caption word seen
    during training and how often. This is the LoRA file's own record of
    what it was trained on - separate from, and often more complete than,
    whatever "trigger words" a Civitai uploader typed into the site."""
    raw = metadata.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        freq_by_dataset = json.loads(raw)
    except Exception:
        return []

    counts = {}
    for _dataset, tags in freq_by_dataset.items():
        if not isinstance(tags, dict):
            continue
        for tag, count in tags.items():
            if isinstance(count, (int, float)):
                counts[tag] = counts.get(tag, 0) + count

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [tag for tag, _count in ranked[:top_n]]


def _read_local_lora_metadata_sync(path):
    """Runs in a background thread: hashes the file and, if present, reads
    its embedded training-tag data. Combined into one step since both need
    to open/read the same file and should be invalidated together."""
    digest = _hash_file_sync(path)
    trained_words = []
    try:
        metadata = _read_safetensors_metadata(path)
        trained_words = _extract_trained_words(metadata)
    except Exception:
        trained_words = []
    return digest, trained_words


async def _get_local_lora_metadata(lora_name):
    path = folder_paths.get_full_path("loras", lora_name)
    if not path or not os.path.isfile(path):
        return None, []

    stat = os.stat(path)
    cache = _load_info_cache()
    entry = cache.get(lora_name, {})

    if (
        entry.get("hash")
        and entry.get("size") == stat.st_size
        and entry.get("mtime") == stat.st_mtime
    ):
        return entry["hash"], entry.get("trained_words", [])

    digest, trained_words = await get_event_loop().run_in_executor(
        None, _read_local_lora_metadata_sync, path
    )
    entry.update(
        {
            "hash": digest,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "trained_words": trained_words,
        }
    )
    cache[lora_name] = entry
    _save_info_cache(cache)
    return digest, trained_words


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/standalone_power_lora_loader/lora_info")
    async def _get_lora_info(request):
        lora_name = request.query.get("lora", "")
        if not lora_name:
            return web.json_response({"error": "missing lora"}, status=400)

        digest, trained_words = await _get_local_lora_metadata(lora_name)
        if digest is None:
            return web.json_response({"error": "file not found"}, status=404)

        entry = _load_info_cache().get(lora_name, {})
        return web.json_response(
            {
                "file": lora_name,
                "hash": digest,
                "name": entry.get("name", ""),
                "strength_min": entry.get("strength_min", ""),
                "strength_max": entry.get("strength_max", ""),
                "notes": entry.get("notes", ""),
                "trained_words": trained_words,
                "civitai": entry.get("civitai"),
            }
        )

    @PromptServer.instance.routes.post("/standalone_power_lora_loader/lora_info")
    async def _save_lora_info(request):
        data = await request.json()
        lora_name = data.get("lora")
        if not lora_name:
            return web.json_response({"error": "missing lora"}, status=400)

        cache = _load_info_cache()
        entry = cache.get(lora_name, {})
        for key in ("name", "strength_min", "strength_max", "notes"):
            if key in data:
                entry[key] = data[key]
        cache[lora_name] = entry
        _save_info_cache(cache)
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.get("/standalone_power_lora_loader/civitai_fetch")
    async def _fetch_civitai_info(request):
        lora_name = request.query.get("lora", "")
        if not lora_name:
            return web.json_response({"error": "missing lora"}, status=400)

        digest, trained_words = await _get_local_lora_metadata(lora_name)
        if digest is None:
            return web.json_response({"error": "file not found"}, status=404)

        url = f"https://civitai.com/api/v1/model-versions/by-hash/{digest}"

        def _do_request():
            req = urllib.request.Request(
                url, headers={"User-Agent": "ComfyUI-StandalonePowerLoraLoader"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            data = await get_event_loop().run_in_executor(None, _do_request)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return web.json_response({"error": "No match found on Civitai for this file."}, status=404)
            return web.json_response({"error": f"Civitai returned an error ({e.code})."}, status=502)
        except Exception as e:
            return web.json_response({"error": f"Could not reach Civitai: {e}"}, status=502)

        model = data.get("model") or {}
        images = []
        for img in (data.get("images") or [])[:24]:
            meta = img.get("meta") or {}
            images.append(
                {
                    "url": img.get("url"),
                    "prompt": meta.get("prompt", ""),
                    "type": img.get("type", "image"),
                }
            )

        model_id = model.get("id") or data.get("modelId")
        civitai_info = {
            "url": f"https://civitai.com/models/{model_id}" if model_id else None,
            "name": model.get("name") or data.get("name"),
            "base_model": data.get("baseModel"),
            # Civitai calls this field "trigger words" on their own site -
            # kept separate from trained_words (the file's own embedded
            # training-tag data), which is a different thing entirely.
            "trigger_words": data.get("trainedWords", []),
            "images": images,
        }

        cache = _load_info_cache()
        entry = cache.get(lora_name, {})
        entry["civitai"] = civitai_info
        if not entry.get("name"):
            entry["name"] = civitai_info.get("name") or ""
        cache[lora_name] = entry
        _save_info_cache(cache)

        return web.json_response(
            {"civitai": civitai_info, "name": entry["name"], "trained_words": trained_words}
        )

    logging.info("[Standalone Power Lora Loader] Registered lora info / Civitai routes")
except Exception as e:
    logging.warning(f"[Standalone Power Lora Loader] Could not register info routes: {e}")


class StandalonePowerLoraLoader(io.ComfyNode):
    """Apply multiple LoRAs to a MODEL/CLIP pair, standalone, Nodes-2.0-safe."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Standalone_PowerLoraLoader",
            display_name="Standalone Power Lora Loader",
            category="loaders",
            description=(
                "Apply any number of LoRAs to a model/clip. Standalone (no "
                "rgthree-comfy required), built for Nodes 2.0. Adjust "
                "'lora slots' to add or remove rows."
            ),
            # All the actual lora rows, and the "lora slots" count control
            # itself, live entirely in JS-built widgets, not declared here -
            # accept_all_inputs lets execute() receive lora_rows anyway.
            accept_all_inputs=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.String.Output(display_name="LoRA Names"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, **kwargs) -> io.NodeOutput:
        rows = kwargs.get("lora_rows")

        # Values received through accept_all_inputs (i.e. not a formally
        # declared/typed schema input) apparently arrive wrapped as
        # {"__value__": actual_data} rather than as the raw data directly -
        # unwrap that before using it.
        if isinstance(rows, dict) and "__value__" in rows:
            rows = rows["__value__"]

        if not isinstance(rows, list):
            rows = []

        lora_tags = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            enabled = row.get("on", True)
            lora_name = row.get("lora", "None")
            strength = row.get("strength", 1.0)

            if not enabled:
                continue
            if not lora_name or lora_name == "None":
                continue
            if not strength:
                continue

            model, clip = LoraLoader().load_lora(
                model, clip, lora_name, strength, strength
            )

            # ForgeUI-style tag, e.g. <lora:watercolorstyle:1.00> - this is
            # for embedding in a saved image's prompt text (via a separate
            # save node), since sites like Civitai can't read a workflow's
            # actual node graph out of a PNG the way ComfyUI itself can.
            clean_name = os.path.splitext(os.path.basename(lora_name))[0]
            lora_tags.append(f"<lora:{clean_name}:{strength:.2f}>")

        lora_names_string = ", ".join(lora_tags)

        return io.NodeOutput(model, clip, lora_names_string)


class StandalonePowerLoraLoaderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [StandalonePowerLoraLoader]


async def comfy_entrypoint() -> StandalonePowerLoraLoaderExtension:
    return StandalonePowerLoraLoaderExtension()
