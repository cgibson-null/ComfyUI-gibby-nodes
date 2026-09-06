"""
Lora Loader
-----------
A multi-lora loader built on ComfyUI's modern V3 node schema, with a
growable/shrinkable row count via a small "lora slots" number field that
updates live.

WHY IT'S BUILT THIS WAY:
rgthree's original version draws its own custom widget (the rows, the
toggles, the "+ Add Lora" button) directly onto the old canvas system with
hand-written JavaScript, doing its own click/drag handling from scratch.
That's exactly the kind of custom node that breaks under Nodes 2.0.

This version instead uses a JS helper (see lora_loader.js) that builds real
HTML elements via ComfyUI's own node.addDOMWidget() function - not hand-
drawn canvas graphics. It isn't reinventing any rendering, which is why it
survives frontend changes that break fully hand-drawn widgets.
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
from comfy_api.latest import io

_lora_stack = io.Custom("LORA_STACK")
_CONTEXT_TYPE = io.Custom("CONTEXT")


def _apply_lora(model, clip, name, strength_model, strength_clip):
    """Apply one LoRA to model/clip via ComfyUI's LoraLoader."""
    if model is not None and clip is not None:
        return LoraLoader().load_lora(model, clip, name, strength_model, strength_clip)
    elif model is not None:
        m, _ = LoraLoader().load_lora(model, None, name, strength_model, 0)
        return m, clip
    elif clip is not None:
        _, c = LoraLoader().load_lora(None, clip, name, 0, strength_clip)
        return model, c
    return model, clip

# The info cache lives at the plugin root (shared runtime state, and where it
# has always been so existing user edits keep working across upgrades).
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_INFO_CACHE_PATH = os.path.join(
    os.path.dirname(_NODE_DIR), "lora_info_cache.json"
)

# Settings loaded from ComfyUI frontend (via API calls)
_civitai_settings = {
    "exclude_words": [],
    "media_limit": 5
}

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
        logging.warning(f"[Lora Loader] Could not save info cache: {e}")


def _hash_file_sync(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lora_hash_for(lora_name):
    """Returns the sha256 hex digest for a lora filename, using the info
    cache when valid and computing + saving it otherwise."""
    path = folder_paths.get_full_path("loras", lora_name)
    if not path or not os.path.isfile(path):
        return None

    stat = os.stat(path)
    cache = _load_info_cache()
    entry = cache.get(lora_name, {})

    if (
        entry.get("hash")
        and entry.get("size") == stat.st_size
        and entry.get("mtime") == stat.st_mtime
    ):
        return entry["hash"]

    digest = _hash_file_sync(path)
    entry.update({"hash": digest, "size": stat.st_size, "mtime": stat.st_mtime})
    cache[lora_name] = entry
    _save_info_cache(cache)
    return digest


def _format_lora_tag(name, strength):
    """ForgeUI-style <lora:name:weight> tag for embedding in saved prompt text."""
    clean_name = os.path.splitext(os.path.basename(name))[0]
    return f"<lora:{clean_name}:{strength:.2f}>"


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


# --- Civitai lookup (owned here, shared with Image Saver) ---------------------

def _fetch_civitai_by_hash(digest):
    """Fetch a model version's raw Civitai API data by hash; raises on failure."""
    url = f"https://civitai.red/api/v1/model-versions/by-hash/{digest.upper()}"
    headers = {"User-Agent": "ComfyUI-GibbyLoraLoader"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_excluded_words():
    """Get excluded words from ComfyUI settings."""
    return set(_civitai_settings.get("exclude_words", []))


def _should_exclude_media(prompt, tags):
    """Check if media should be excluded based on prompt or tags."""
    excluded_words = _get_excluded_words()
    
    if not excluded_words:
        return False
    
    # Check tags list (case-insensitive)
    if tags:
        tag_set = {t.lower().strip() for t in tags if isinstance(t, str)}
        for excluded in excluded_words:
            if excluded in tag_set:
                return True

    # Check prompt text (case-insensitive)
    if prompt:
        prompt_lower = prompt.lower()
        for excluded in excluded_words:
            if excluded in prompt_lower:
                return True

    return False


def _civitai_info_from_api(data):
    """Shape a raw Civitai by-hash response into the cached lora civitai entry."""
    model = data.get("model") or {}
    images = []
    for img in (data.get("images") or [])[:24]:
        meta = img.get("meta") or {}
        prompt = meta.get("prompt", "")
        tags = meta.get("tags", [])
        # Skip images with excluded tags
        if _should_exclude_media(prompt, tags):
            continue
        images.append(
            {
                "url": img.get("url"),
                "prompt": prompt,
                "type": img.get("type", "image"),
            }
        )

    model_id = model.get("id") or data.get("modelId")
    model_version_id = data.get("id")
    # Create URL with model version ID to link to the specific version
    if model_id:
        url = f"https://civitai.red/models/{model_id}"
        if model_version_id:
            url += f"?modelVersionId={model_version_id}"
    else:
        url = None
    return {
        "url": url,
        "name": model.get("name") or data.get("name"),
        "base_model": data.get("baseModel"),
        # Civitai calls this field "trigger words" on their own site -
        # kept separate from trained_words (the file's own embedded
        # training-tag data), which is a different thing entirely.
        "trigger_words": data.get("trainedWords", []),
        "images": images,
        # Identifiers Image Saver needs for its civitai resource metadata.
        "version_name": data.get("name"),
        "air": data.get("air"),
        "model_version_id": model_version_id,
    }


def _cache_lora_civitai(lora_name, civitai_info):
    cache = _load_info_cache()
    entry = cache.get(lora_name, {})
    entry["civitai"] = civitai_info
    if not entry.get("name"):
        entry["name"] = civitai_info.get("name") or ""
    cache[lora_name] = entry
    _save_info_cache(cache)


def _lora_name_for_path(filepath):
    """Relative lora filename (info-cache key form) for a full path, or None."""
    if not filepath:
        return None
    target = os.path.abspath(filepath)
    # normcase is case-insensitive on Windows - use it only for the prefix test,
    # never for building the name itself.
    target_norm = os.path.normcase(target)
    for base in folder_paths.folder_names_and_paths.get("loras", [[], set()])[0]:
        base_abs = os.path.abspath(base)
        if target_norm.startswith(os.path.normcase(base_abs) + os.sep):
            return os.path.relpath(target, base_abs)
    return None


def _fetch_lora_civitai_info(lora_name, digest):
    """Fetch and cache Civitai info for a lora by hash; returns it or None."""
    try:
        data = _fetch_civitai_by_hash(digest)
    except urllib.error.HTTPError as e:
        logging.warning(f"[Lora Loader] Civitai error ({e.code}) fetching info for '{lora_name}'")
        return None
    except Exception as e:
        logging.warning(f"[Lora Loader] Could not reach Civitai for '{lora_name}': {e}")
        return None

    civitai_info = _civitai_info_from_api(data)
    _cache_lora_civitai(lora_name, civitai_info)
    return civitai_info


def _get_or_fetch_lora_civitai_info(filepath, digest):
    """Civitai info for a lora by full path; fetches and caches when missing."""
    name = _lora_name_for_path(filepath)
    if not name:
        return None

    info = _load_info_cache().get(name, {}).get("civitai")
    # Re-fetch entries that predate the metadata fields Image Saver needs.
    if info is None or not (info.get("air") or info.get("model_version_id")):
        info = _fetch_lora_civitai_info(name, digest)
    return info


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/gibby_nodes/lora_info")
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

    @PromptServer.instance.routes.post("/gibby_nodes/lora_info")
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
        
        # Handle civitai data
        if "civitai" in data:
            entry["civitai"] = data["civitai"]
        
        cache[lora_name] = entry
        _save_info_cache(cache)
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.get("/gibby_nodes/civitai_fetch")
    async def _fetch_civitai_info(request):
        lora_name = request.query.get("lora", "")
        if not lora_name:
            return web.json_response({"error": "missing lora"}, status=400)

        digest, trained_words = await _get_local_lora_metadata(lora_name)
        if digest is None:
            return web.json_response({"error": "file not found"}, status=404)

        try:
            data = await get_event_loop().run_in_executor(None, _fetch_civitai_by_hash, digest)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return web.json_response({"error": "No match found on Civitai for this file."}, status=404)
            return web.json_response({"error": f"Civitai returned an error ({e.code})."}, status=502)
        except Exception as e:
            return web.json_response({"error": f"Could not reach Civitai: {e}"}, status=502)

        civitai_info = _civitai_info_from_api(data)
        _cache_lora_civitai(lora_name, civitai_info)
        entry_name = _load_info_cache().get(lora_name, {}).get("name", "")

        return web.json_response(
            {"civitai": civitai_info, "name": entry_name, "trained_words": trained_words}
        )

    @PromptServer.instance.routes.get("/gibby_nodes/civitai_settings")
    async def _get_civitai_settings(request):
        return web.json_response(_civitai_settings)

    @PromptServer.instance.routes.post("/gibby_nodes/civitai_settings")
    async def _save_civitai_settings(request):
        global _civitai_settings
        data = await request.json()
        _civitai_settings.update(data)
        return web.json_response({"ok": True, "settings": _civitai_settings})

    logging.info("[Lora Loader] Registered lora info / Civitai routes")
except Exception as e:
    logging.warning(f"[Lora Loader] Could not register info routes: {e}")


class GibbyLoraLoader(io.ComfyNode):
    """Apply multiple LoRAs to a MODEL/CLIP pair (Nodes 2.0 safe)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_LoraLoader",
            display_name="Lora Loader",
            category="gibby",
            description=(
                "Apply any number of LoRAs to a model/clip. No rgthree-comfy "
                "required; built for Nodes 2.0. Adjust 'lora slots' to add "
                "or remove rows."
            ),
            # All the actual lora rows, and the "lora slots" count control
            # itself, live entirely in JS-built widgets, not declared here -
            # accept_all_inputs lets execute() receive lora_rows anyway.
            accept_all_inputs=True,
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                _lora_stack.Input("lora_stack", optional=True),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                _lora_stack.Output(display_name="lora_stack"),
                io.String.Output(display_name="lora_names"),
            ],
        )

    @classmethod
    def execute(cls, context=None, model=None, clip=None, lora_stack=None, **kwargs) -> io.NodeOutput:
        rows = kwargs.get("lora_rows")

        # Values received through accept_all_inputs (i.e. not a formally
        # declared/typed schema input) apparently arrive wrapped as
        # {"__value__": actual_data} rather than as the raw data directly -
        # unwrap that before using it.
        if isinstance(rows, dict) and "__value__" in rows:
            rows = rows["__value__"]

        if not isinstance(rows, list):
            rows = []

        # Start from the incoming context's values (if any).
        ctx = {}
        if isinstance(context, dict):
            ctx.update(context)

        # A stack carried by the context is assumed already baked into that
        # context's model/clip - valid only while we keep using them.
        stack_already_applied = (
            isinstance(ctx.get("lora_stack"), list)
            and (ctx.get("model") is not None or ctx.get("clip") is not None)
        )

        # Directly-connected model/clip override the context's values, which
        # invalidates the "already applied" assumption for that stack.
        if model is not None:
            ctx["model"] = model
            stack_already_applied = False
        if clip is not None:
            ctx["clip"] = clip
            stack_already_applied = False

        work_model = ctx.get("model")
        work_clip = ctx.get("clip")

        # Build accumulated stack like CR_LoRAStack does.
        stacked_loras = []
        lora_tags = []

        if stack_already_applied:
            for item in ctx["lora_stack"]:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                stacked_loras.append(item)
                lora_tags.append(_format_lora_tag(item[0], item[1]))

        # Apply the incoming lora_stack entries (not yet applied).
        if isinstance(lora_stack, list):
            for item in lora_stack:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                name, sm, sc = item[0], item[1], item[2]

                work_model, work_clip = _apply_lora(work_model, work_clip, name, sm, sc)
                stacked_loras.append(item)

                # Add tag for incoming stack LoRAs too.
                lora_tags.append(_format_lora_tag(name, sm))

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

            # Skip loras that don't exist on disk
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.isfile(lora_path):
                continue

            # Apply LoRAs to whichever of model/clip are connected (independently).
            work_model, work_clip = _apply_lora(work_model, work_clip, lora_name, strength, strength)
            stacked_loras.append((lora_name, strength, strength))

            # Tag for embedding in a saved image's prompt text - sites like
            # Civitai can't read the workflow graph out of a PNG.
            lora_tags.append(_format_lora_tag(lora_name, strength))

        # All outputs reflect the model/clip after every LoRA is applied, and
        # the context carries that state forward for downstream nodes.
        ctx["model"] = work_model
        ctx["clip"] = work_clip
        ctx["lora_stack"] = stacked_loras

        lora_names_string = ", ".join(lora_tags)

        return io.NodeOutput(ctx, work_model, work_clip, stacked_loras, lora_names_string)
