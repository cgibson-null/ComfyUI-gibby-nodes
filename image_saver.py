"""Image Saver (Context).

Saves images with civitai-compatible generation metadata, adapted for context
workflows:

- Generation settings (model name, seed, steps, sampler, prompts, size) come
  from the linked CONTEXT instead of widgets; an optional image input overrides
  the context's image.
- LoRA names are pulled from the context's lora_stack and appended to the
  positive prompt as <lora:name:weight> tags before metadata processing.
- Resource hashes and Civitai data live in JSON caches at the plugin root
  (lora_info_cache.json via Lora Loader, model_info_cache.json here) instead
  of .sha256/.civitai.info files next to models; embeddings are hashed in
  memory. LoRA Civitai data is read from Lora Loader's cache.

Based on ComfyUI-Image-Saver by Luciano Cirino:
https://github.com/LucianoCirino/ComfyUI-Image-Saver
"""

import json
import os
import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import piexif
import piexif.helper
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.sd1_clip import escape_important, unescape_important, token_weights
from comfy_api.latest import io

from .context import _CONTEXT_TYPE
from .lora_loader import _format_lora_tag, _get_or_fetch_lora_civitai_info, _hash_file_sync, _lora_hash_for


# --- File path matching (ComfyUI-Image-Saver/utils.py) -----------------------

def _sanitize_filename(filename: str) -> str:
    """Remove characters that are unsafe for filenames."""
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
    return sanitized.rstrip('. ')


def _full_embedding_path_for(embedding: str):
    matching_embedding = _get_file_path_match("embeddings", embedding)
    if matching_embedding is None:
        print(f'Gibby Image Saver: could not find full path to embedding "{embedding}"')
        return None
    return folder_paths.get_full_path("embeddings", matching_embedding)


def _full_checkpoint_path_for(model_name: str) -> str:
    if not model_name:
        return ''

    supported_extensions = set(folder_paths.supported_pt_extensions) | {".gguf"}

    matching_checkpoint = _get_file_path_match("checkpoints", model_name, supported_extensions)
    if matching_checkpoint is not None:
        return folder_paths.get_full_path("checkpoints", matching_checkpoint)

    matching_model = _get_file_path_match("diffusion_models", model_name, supported_extensions)
    if matching_model:
        return folder_paths.get_full_path("diffusion_models", matching_model)

    print(f'Could not find full path to checkpoint "{model_name}"')
    return ''


def _get_file_path_iterator(folder_name: str, supported_extensions=None):
    """Returns an iterator over valid file paths for the specified model folder."""
    if supported_extensions is None:
        return (Path(x) for x in folder_paths.get_filename_list(folder_name))
    else:
        return _custom_file_path_generator(folder_name, supported_extensions)


def _custom_file_path_generator(folder_name: str, supported_extensions: Collection[str]):
    """Generator function for file paths, allowing for a customized extension check."""
    model_paths = folder_paths.folder_names_and_paths.get(folder_name, [[], set()])[0]
    for path in model_paths:
        if os.path.exists(path):
            base_path = Path(path)
            for root, _, files in os.walk(path):
                root_path = Path(root).relative_to(base_path)
                for file in files:
                    file_path = root_path / file
                    if file_path.suffix.lower() in supported_extensions:
                        yield file_path


def _get_file_path_match(folder_name: str, file_name: str, supported_extensions=None):
    supported_extensions_fallback = supported_extensions if supported_extensions is not None else folder_paths.supported_pt_extensions
    file_path = Path(file_name)

    # first try full path match, then fallback to just name match, matching the extension if appropriate
    if file_path.suffix.lower() not in supported_extensions_fallback:
        matching_file_path = next((p for p in _get_file_path_iterator(folder_name, supported_extensions) if p.with_suffix('') == file_path), None)
        matching_file_path = (matching_file_path if matching_file_path is not None else
            next((p for p in _get_file_path_iterator(folder_name, supported_extensions) if p.stem == file_path.name), None))
    else:
        matching_file_path = next((p for p in _get_file_path_iterator(folder_name, supported_extensions) if p == file_path), None)
        matching_file_path = (matching_file_path if matching_file_path is not None else
            next((p for p in _get_file_path_iterator(folder_name, supported_extensions) if p.name == file_path.name), None))

    return str(matching_file_path) if matching_file_path is not None else None


def _http_get_json(url: str):
    try:
        response = requests.get(url, timeout=300)
    except requests.exceptions.Timeout:
        print(f"Gibby Image Saver: HTTP GET Request timed out for {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Gibby Image Saver: Warning - Network connection error for {url}: {e}")
        return None

    if not response.ok:
        print(f"Gibby Image Saver: HTTP GET Request failed with error code: {response.status_code}: {response.reason}")
        return None

    try:
        return response.json()
    except ValueError as e:
        print(f"Gibby Image Saver: HTTP Response JSON error: {e}")
    return None


# --- Model info cache (hashes + Civitai data; JSON instead of .sha256/.civitai.info files)

_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_INFO_CACHE_PATH = os.path.join(_PLUGIN_ROOT, "model_info_cache.json")


def _load_model_info_cache() -> dict:
    try:
        with open(_MODEL_INFO_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_model_info_cache(cache: dict):
    try:
        with open(_MODEL_INFO_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Gibby Image Saver: could not save model info cache: {e}")


def _model_hash_for(model_path: str):
    """sha256 for a model file via the plugin-root JSON cache; computes and saves when absent or stale."""
    if not model_path or not os.path.isfile(model_path):
        return None

    stat = os.stat(model_path)
    name = os.path.basename(model_path)
    cache = _load_model_info_cache()
    entry = cache.get(name, {})

    if (
        entry.get("hash")
        and entry.get("size") == stat.st_size
        and entry.get("mtime") == stat.st_mtime
    ):
        return entry["hash"]

    digest = _hash_file_sync(model_path)
    entry.update({"hash": digest, "size": stat.st_size, "mtime": stat.st_mtime})
    cache[name] = entry
    _save_model_info_cache(cache)
    return digest


# --- Civitai helpers (ComfyUI-Image-Saver/utils_civitai.py) ------------------

MAX_HASH_LENGTH = 16 # skip larger unshortened hashes, such as full sha256 or blake3


def _civitai_embedding_key_name(embedding: str) -> str:
    return f'embed:{embedding}'


def _civitai_lora_key_name(lora: str) -> str:
    return f'LORA:{lora}'


CIVITAI_SAMPLER_MAP = {
    'euler_ancestral': 'Euler a',
    'euler': 'Euler',
    'lms': 'LMS',
    'heun': 'Heun',
    'dpm_2': 'DPM2',
    'dpm_2_ancestral': 'DPM2 a',
    'dpmpp_2s_ancestral': 'DPM++ 2S a',
    'dpmpp_2m': 'DPM++ 2M',
    'dpmpp_sde': 'DPM++ SDE',
    'dpmpp_2m_sde': 'DPM++ 2M SDE',
    'dpmpp_3m_sde': 'DPM++ 3M SDE',
    'dpm_fast': 'DPM fast',
    'dpm_adaptive': 'DPM adaptive',
    'ddim': 'DDIM',
    'plms': 'PLMS',
    'uni_pc_bh2': 'UniPC',
    'uni_pc': 'UniPC',
    'lcm': 'LCM',
}


def _get_civitai_sampler_name(sampler_name: str, scheduler: str) -> str:
    # based on: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts#L122
    if sampler_name in CIVITAI_SAMPLER_MAP:
        civitai_name = CIVITAI_SAMPLER_MAP[sampler_name]

        if scheduler == "karras":
            civitai_name += " Karras"
        elif scheduler == "exponential":
            civitai_name += " Exponential"

        return civitai_name
    else:
        if scheduler != 'normal':
            return f"{sampler_name}_{scheduler}"
        else:
            return sampler_name


def _civitai_resource_data(model_name: Any, version_name: Any, weight: float | None, air: Any, model_version_id: Any) -> dict[str, str | float]:
    resource_data: dict[str, str | float] = {
        # Optional data - modelName, versionName
        "modelName": model_name or "",
        "versionName": version_name or "",
    }

    # Weight/strength (for LoRA or embedding)
    if weight is not None:
        resource_data["weight"] = weight

    # Required data - AIR or modelVersionId (unique resource identifier)
    # https://github.com/civitai/civitai/wiki/AIR-%E2%80%90-Uniform-Resource-Names-for-AI
    if air:
        resource_data["air"] = air
    else:
        # Fallback if AIR is not found
        resource_data["modelVersionId"] = model_version_id

    return resource_data


def _get_civitai_metadata(
        modelname: str,
        ckpt_path: str,
        modelhash: str,
        loras: dict[str, tuple[str, float, str]],
        embeddings: dict[str, tuple[str, float, str]],
        manual_entries: dict[str, tuple[str | None, float | None, str]],
        download_civitai_data: bool) -> tuple[list[dict[str, str | float]], dict[str, str], str | None]:
    """Download or load cache of Civitai data, save specially-formatted data to metadata"""
    civitai_resources: list[dict[str, str | float]] = []
    hashes = {}
    add_model_hash = None

    if download_civitai_data:
        # LoRAs - their Civitai data is owned by Lora Loader's info cache.
        for name, (filepath, weight, hash) in loras.items():
            civitai_info = _get_or_fetch_lora_civitai_info(filepath, hash)
            if civitai_info is not None:
                civitai_resources.append(_civitai_resource_data(civitai_info.get("name"), civitai_info.get("version_name"), weight, civitai_info.get("air"), civitai_info.get("model_version_id")))
            else:
                # Fallback in case the data wasn't loaded to add to the "Hashes" section
                hashes[name] = hash.upper()

        # Model + embeddings + manual hashes - via the plugin-root Civitai cache.
        for name, (filepath, weight, hash) in ({ modelname: ( ckpt_path, None, modelhash ) } | embeddings | manual_entries).items():
            civitai_info = _get_model_civitai_info(filepath, hash)
            if civitai_info is not None:
                civitai_resources.append(_civitai_resource_data(civitai_info["model"]["name"], civitai_info.get("name"), weight, civitai_info.get("air"), civitai_info.get("id")))
            else:
                # Fallback in case the data wasn't loaded to add to the "Hashes" section
                if name == modelname:
                    add_model_hash = hash.upper()
                else:
                    hashes[name] = hash.upper()
    else:
        # Convert all hashes to JSON format
        hashes = {key: value[2] for key, value in embeddings.items()} | {key: value[2] for key, value in loras.items()} | {key: value[2] for key, value in manual_entries.items()} | {"model": modelhash}
        add_model_hash = modelhash

    return civitai_resources, hashes, add_model_hash


def _get_model_civitai_info(filepath: str | None, model_hash: str) -> dict[str, Any] | None:
    """Civitai data for a model/embedding/manual hash via the plugin-root JSON cache."""
    if not model_hash:
        print("Gibby Image Saver: Error: Missing hash.")
        return None

    try:
        cache = _load_model_info_cache()
        # Manual hashes (no local file) are keyed by hash.
        key = os.path.basename(filepath) if filepath else model_hash.upper()
        info = cache.get(key, {}).get("civitai")
        if info is not None:
            return info

        content = _download_model_info(model_hash)
        if content is None:
            return None
        cache[key] = cache.get(key, {}) | {"civitai": content}
        _save_model_info_cache(cache)
        return content
    except Exception as e:
        print(f"Gibby Image Saver: Civitai info error: {e}")
    return None


def _download_model_info(model_hash: str) -> dict[str, object] | None:
    print(f"Gibby Image Saver: Downloading model info for '{model_hash}'.")

    content = _http_get_json(f'https://civitai.red/api/v1/model-versions/by-hash/{model_hash.upper()}')
    if content is None:
        return None
    model_id = content["modelId"]
    parent_model = _http_get_json(f'https://civitai.red/api/v1/models/{model_id}')
    if not parent_model:
        parent_model = {}

    content["creator"] = parent_model.get("creator", "{}")
    model_metadata = content["model"]
    for metadata in [ "description", "tags", "allowNoCredit", "allowCommercialUse", "allowDerivatives", "allowDifferentLicense" ]:
        model_metadata[metadata] = parent_model.get(metadata, "")

    return content


# --- Prompt metadata extraction (ComfyUI-Image-Saver/prompt_metadata_extractor.py)

class PromptMetadataExtractor:
    # Anything that follows embedding:<characters except , or whitespace
    EMBEDDING: str = r'embedding:([^,\s\(\)\:]+)'
    # Anything that follows <lora:NAME> with allowance for :weight, :weight.fractal or LBW
    LORA: str = r'<lora:([^>:]+)(?::([^>]+))?>'

    def __init__(self, prompts: list[str]) -> None:
        self.__embeddings: dict[str, tuple[str, float, str]] = {}
        self.__loras: dict[str, tuple[str, float, str]] = {}
        self.__perform(prompts)

    def get_embeddings(self) -> dict[str, tuple[str, float, str]]:
        """Returns the embeddings used in the given prompts in a format as known by civitAI"""
        return self.__embeddings

    def get_loras(self) -> dict[str, tuple[str, float, str]]:
        """Returns the lora's used in the given prompts in a format as known by civitAI"""
        return self.__loras

    # Private API
    def __perform(self, prompts: list[str]) -> None:
        for prompt in prompts:
            # Use ComfyUI's built-in attention parser to get accurate weights for embeddings
            parsed = ((unescape_important(value), weight) for value, weight in token_weights(escape_important(prompt), 1.0))
            for text, weight in parsed:
                embeddings = re.findall(self.EMBEDDING, text, re.IGNORECASE | re.MULTILINE)
                for embedding in embeddings:
                    self.__extract_embedding_information(embedding, weight)
            loras = re.findall(self.LORA, prompt, re.IGNORECASE | re.MULTILINE)
            for lora in loras:
                self.__extract_lora_information(lora)

    def __extract_embedding_information(self, embedding: str, weight: float) -> None:
        embedding_name = _civitai_embedding_key_name(embedding)
        embedding_path = _full_embedding_path_for(embedding)
        if embedding_path is None:
            return
        # Embeddings are small; hash in memory instead of writing .sha256 files
        sha = _hash_file_sync(embedding_path)[:10]
        self.__embeddings[embedding_name] = (embedding_path, weight, sha)

    def __extract_lora_information(self, lora: tuple[str, str]) -> None:
        lora_name = _civitai_lora_key_name(lora[0])
        # Match by full filename (with subfolder) so hashing resolves the same
        # way Lora Loader does and reuses its info cache entries.
        matching_lora = _get_file_path_match("loras", lora[0])
        if not matching_lora:
            print(f'Gibby Image Saver: could not find full path to lora "{lora[0]}"')
            return
        matching_lora = os.path.normpath(matching_lora)
        lora_path = folder_paths.get_full_path("loras", matching_lora)
        try:
            lora_weight = float(lora[1].split(':')[0])
        except (ValueError, TypeError):
            lora_weight = 1.0
        # Hash via the shared lora info cache instead of writing .sha256 files
        digest = _lora_hash_for(matching_lora)
        if not digest:
            return
        self.__loras[lora_name] = (lora_path, lora_weight, digest[:10])


# --- Filename templating (ComfyUI-Image-Saver/nodes.py) ----------------------

def _parse_checkpoint_name(ckpt_name: str) -> str:
    return os.path.basename(ckpt_name)


def _parse_checkpoint_name_without_extension(ckpt_name: str) -> str:
    filename = _parse_checkpoint_name(ckpt_name)
    name_without_ext, ext = os.path.splitext(filename)
    supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}

    # Only remove extension if it's a known model file extension
    if ext.lower() in supported_extensions:
        return name_without_ext
    else:
        return filename # Keep full name if extension isn't recognized


def _get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except Exception:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def _apply_custom_time_format(filename: str) -> str:
    """Replace %time_format<strftime_format> patterns with formatted datetime."""
    now = datetime.now()
    pattern = r'%time_format<([^>]*)>'
    def replace_format(match):
        format_str = match.group(1)
        try:
            return now.strftime(format_str)
        except Exception:
            # If format is invalid, return original
            return match.group(0)

    return re.sub(pattern, replace_format, filename)


def _apply_custom_counter_format(filename: str, counter: int) -> str:
    """Replace %counter<padding> patterns with formatted counter."""
    pattern = r'%counter<([0-9]+)>'
    def replace_format(match):
        padding_str = match.group(1)
        try:
            # Create format string like "{:03d}"
            fmt = "{:0" + padding_str + "d}"
            return fmt.format(counter)
        except Exception:
            return match.group(0)

    return re.sub(pattern, replace_format, filename)


def _save_json(image_info: dict[str, Any] | None, filename: str) -> None:
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')


def _make_pathname(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, custom: str) -> str:
    # Process custom format patterns first
    filename = _apply_custom_time_format(filename)
    filename = _apply_custom_counter_format(filename, counter)
    filename = filename.replace("%date", _get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", _get_timestamp(time_format))
    filename = filename.replace("%model", _parse_checkpoint_name(modelname))
    filename = filename.replace("%width", str(width))
    filename = filename.replace("%height", str(height))
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    filename = filename.replace("%sampler_name", sampler_name)
    filename = filename.replace("%steps", str(steps))
    filename = filename.replace("%cfg", str(cfg))
    filename = filename.replace("%scheduler_name", scheduler_name)
    filename = filename.replace("%basemodelname", _parse_checkpoint_name_without_extension(modelname))
    filename = filename.replace("%denoise", str(denoise))
    filename = filename.replace("%custom", custom)

    directory, basename = os.path.split(filename)
    sanitized_basename = _sanitize_filename(basename)
    return os.path.join(directory, sanitized_basename)


def _make_filename(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, custom: str) -> str:
    filename = _make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler_name, denoise, custom)
    return _get_timestamp(time_format) if filename == "" else filename


@dataclass
class Metadata:
    modelname: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_name: str
    denoise: float
    custom: str
    additional_hashes: str
    ckpt_path: str
    a111_params: str
    final_hashes: str


# Match 'anything' or 'anything:anything' with trimmed white space
re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
# Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')


def _get_multiple_models(modelname: str, additional_hashes: str) -> tuple[str, str]:
    model_names = [m.strip() for m in modelname.split(',')]
    modelname = model_names[0] # Use the first model as the primary one

    # Process additional model names and add to additional_hashes
    for additional_model in model_names[1:]:
        additional_ckpt_path = _full_checkpoint_path_for(additional_model)
        if additional_ckpt_path:
            additional_digest = _model_hash_for(additional_ckpt_path)
            if additional_digest:
                # Add to additional_hashes in "name:HASH" format
                if additional_hashes:
                    additional_hashes += ","
                additional_hashes += f"{additional_model}:{additional_digest[:10]}"
    return modelname, additional_hashes


def _parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
    """Process additional_hashes input (a string) by normalizing, removing extra spaces/newlines, and splitting by comma"""
    manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
    unnamed_count = 0

    additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
    for entry in additional_hash_split:
        match = (re_manual_hash_weights if download_civitai_data else re_manual_hash).search(entry)
        if match is None:
            print(f"Gibby Image Saver: Invalid additional hash string: '{entry}'")
            continue

        groups = tuple(group for group in match.groups() if group)

        # Read weight and remove from groups, if needed
        weight = None
        if download_civitai_data and len(groups) > 1:
            try:
                weight = float(groups[-1])
                groups = groups[:-1]
            except (ValueError, TypeError):
                pass

        # Read hash, optionally preceded by name
        name, hash = groups if len(groups) > 1 else (None, groups[0])

        if len(hash) > MAX_HASH_LENGTH:
            print(f"Gibby Image Saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
            continue

        if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
            print(f"Gibby Image Saver: Skipping duplicate hash: {hash}")
            continue  # Skip duplicates

        if hash.lower() in existing_hashes:
            print(f"Gibby Image Saver: Skipping manual hash already present in resources: {hash}")
            continue

        if name is None:
            unnamed_count += 1
            name = f"manual{unnamed_count}"
        elif name in manual_entries:
            print(f"Gibby Image Saver: Duplicate manual hash name '{name}' is being overwritten.")

        manual_entries[name] = (None, weight, hash)

        if len(manual_entries) > 29:
            print("Gibby Image Saver: Reached maximum limit of 30 manual hashes. Skipping the rest.")
            break

    return manual_entries


def _clean_prompt(prompt: str, metadata_extractor: PromptMetadataExtractor) -> str:
    """Clean prompts for easier remixing by removing LoRAs and simplifying embeddings."""
    # Strip loras
    prompt = re.sub(metadata_extractor.LORA, "", prompt)
    # Shorten 'embedding:path/to/my_embedding' -> 'my_embedding'
    # Note: Possible inaccurate embedding name if the filename has been renamed from the default
    prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
    # Remove prompt control edits. e.g., 'STYLE(A1111, mean)', 'SHIFT(1)`, etc.`
    prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
    return prompt


def _make_metadata(modelname: str, positive: str, negative: str, width: int, height: int, seed_value: int, steps: int, cfg: float, sampler_name: str, scheduler_name: str, denoise: float, custom: str, additional_hashes: str, download_civitai_data: bool, easy_remix: bool) -> Metadata:
    modelname, additional_hashes = _get_multiple_models(modelname, additional_hashes)

    ckpt_path = _full_checkpoint_path_for(modelname)
    model_digest = _model_hash_for(ckpt_path) if ckpt_path else None
    modelhash = model_digest[:10] if model_digest else ""

    metadata_extractor = PromptMetadataExtractor([positive, negative])
    embeddings = metadata_extractor.get_embeddings()
    loras = metadata_extractor.get_loras()
    civitai_sampler_name = _get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler_name)
    basemodelname = _parse_checkpoint_name_without_extension(modelname)

    # Get existing hashes from model, loras, and embeddings
    existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
    # Parse manual hashes
    manual_entries = _parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
    # Get Civitai metadata
    civitai_resources, hashes, add_model_hash = _get_civitai_metadata(modelname, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data)

    if easy_remix:
        positive = _clean_prompt(positive, metadata_extractor)
        negative = _clean_prompt(negative, metadata_extractor)

    positive_a111_params = positive.strip()
    negative_a111_params = f"\nNegative prompt: {negative.strip()}"
    custom_str = f", {custom}" if custom else ""
    model_hash_str = f", Model hash: {add_model_hash}" if add_model_hash else ""
    hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

    a111_params = (
        f"{positive_a111_params}{negative_a111_params}\n"
        f"Steps: {steps}, Sampler: {civitai_sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, "
        f"Size: {width}x{height}{custom_str}{model_hash_str}, Model: {basemodelname}{hashes_str}, Version: ComfyUI"
    )

    # Add Civitai resource listing
    if download_civitai_data and civitai_resources:
        a111_params += f", Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}"

    # Combine all resources (model, loras, embeddings, manual entries) for final hash string
    all_resources = { modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries

    hash_parts = []
    for name, (_, weight, hash_value) in (all_resources.items() if isinstance(all_resources, dict) else all_resources):
        # Format: "name:hash" or "name:hash:weight" depending on download_civitai_data
        if name:
            # Extract clean name (only remove actual model file extensions, preserve dots in model names)
            filename = name.split(':')[-1]
            name_without_ext, ext = os.path.splitext(filename)
            supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}

            # Only remove extension if it's a known model file extension
            if ext.lower() in supported_extensions:
                clean_name = name_without_ext
            else:
                clean_name = filename  # Keep full name if extension isn't recognized

            name_part = f"{clean_name}:"
        else:
            name_part = ""

        # Skip entries without a valid hash
        if not hash_value:
            continue

        weight_part = f":{weight}" if weight is not None and download_civitai_data else ""
        hash_parts.append(f"{name_part}{hash_value}{weight_part}")

    final_hashes = ",".join(hash_parts)

    return Metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, custom, additional_hashes, ckpt_path, a111_params, final_hashes)


# --- Image saving (ComfyUI-Image-Saver/saver/saver.py + nodes.py) ------------

def _save_image(image: Image.Image, filepath: str, extension: str, quality_jpeg_or_webp: int, lossless_webp: bool, optimize_png: bool, a111_params: str, prompt: dict[str, Any] | None, extra_pnginfo: dict[str, Any] | None, embed_workflow: bool) -> None:
    if extension == 'png':
        metadata = PngInfo()
        if a111_params:
            metadata.add_text("parameters", a111_params)

        if embed_workflow:
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v, separators=(',', ':')))
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt, separators=(',', ':')))

        image.save(filepath, pnginfo=metadata, optimize=optimize_png)
    else: # webp & jpeg
        image.save(filepath, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)

        # Native example adding workflow to exif:
        # https://github.com/comfyanonymous/ComfyUI/blob/095610717000bffd477a7e72988d1fb2299afacb/comfy_extras/nodes_images.py#L113
        pnginfo_json = {}
        prompt_json = {}
        if embed_workflow:
            if extra_pnginfo is not None:
                pnginfo_json = {piexif.ImageIFD.Make - i: f"{k}:{json.dumps(v, separators=(',', ':'))}" for i, (k, v) in enumerate(extra_pnginfo.items())}
            if prompt is not None:
                prompt_json = {piexif.ImageIFD.Model: f"prompt:{json.dumps(prompt, separators=(',', ':'))}"}

        def get_exif_bytes() -> bytes:
            exif_dict = ({
                "0th": pnginfo_json | prompt_json
                } if pnginfo_json or prompt_json else {}) | ({
                "Exif": {
                    piexif.ExifIFD.UserComment: cast(bytes, piexif.helper.UserComment.dump(a111_params, encoding="unicode"))
                },
            } if a111_params else {})
            return cast(bytes, piexif.dump(exif_dict))

        exif_bytes = get_exif_bytes()

        # JPEG format limits the EXIF bytes to a maximum of 65535 bytes
        if extension == "jpg" or extension == "jpeg":
            MAX_EXIF_SIZE = 65535
            if len(exif_bytes) > MAX_EXIF_SIZE and embed_workflow:
                print("Gibby Image Saver: Error: Workflow is too large, removing client request prompt.")
                prompt_json = {}
                exif_bytes = get_exif_bytes()
                if len(exif_bytes) > MAX_EXIF_SIZE:
                    print("Gibby Image Saver: Error: Workflow is still too large, cannot embed workflow!")
                    pnginfo_json = {}
                    exif_bytes = get_exif_bytes()
            if len(exif_bytes) > MAX_EXIF_SIZE:
                print("Gibby Image Saver: Error: Metadata exceeds maximum size for JPEG. Cannot save metadata.")
                return

        piexif.insert(exif_bytes, filepath)


def _get_base_suffix(output_path: str, filename_prefix: str, extension: str, batch_size: int) -> int | None:
    """Calculate the base suffix for batch naming. Returns None for single images with no existing files."""
    existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]

    # For single images with no existing files, return None (no suffix needed)
    if batch_size == 1 and not existing_files:
        return None

    # For batches or when files exist, calculate base suffix
    suffixes: list[int] = []
    for f in existing_files:
        name, _ = os.path.splitext(f)
        parts = name.split('_')
        if parts[-1].isdigit():
            suffixes.append(int(parts[-1]))

    if suffixes:
        return max(suffixes) + 1
    else:
        return 1


def _format_batch_filename(filename_prefix: str, base_suffix: int | None, batch_index: int) -> str:
    """Format filename with batch suffix. If base_suffix is None, returns plain filename."""
    if base_suffix is None:
        return filename_prefix
    return f"{filename_prefix}_{base_suffix + batch_index:02d}"


def _save_images(
    images,
    filename_pattern: str,
    extension: str,
    path: str,
    quality_jpeg_or_webp: int,
    lossless_webp: bool,
    optimize_png: bool,
    prompt: dict[str, Any] | None,
    extra_pnginfo: dict[str, Any] | None,
    save_workflow_as_json: bool,
    embed_workflow: bool,
    counter: int,
    time_format: str,
    metadata: Metadata
) -> list[str]:
    filename_prefix = _make_filename(filename_pattern, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.custom)

    output_path = os.path.join(folder_paths.output_directory, path)

    if output_path.strip() != '':
        if not os.path.exists(output_path.strip()):
            print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path, exist_ok=True)

    result_paths: list[str] = list()
    num_images = len(images)
    # Calculate base suffix once before the loop to avoid re-scanning after each save
    base_suffix = _get_base_suffix(output_path, filename_prefix, extension, num_images)
    for idx, image in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        current_filename_prefix = _format_batch_filename(filename_prefix, base_suffix, idx)
        final_filename = f"{current_filename_prefix}.{extension}"
        filepath = os.path.join(output_path, final_filename)

        _save_image(img, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, metadata.a111_params, prompt, extra_pnginfo, embed_workflow)

        if save_workflow_as_json:
            _save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

        result_paths.append(final_filename)
    return result_paths


# --- Node --------------------------------------------------------------------

class GibbyImageSaverContext(io.ComfyNode):
    """Save images with civitai-compatible metadata from a context workflow."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_ImageSaver_Context",
            display_name="Image Saver (Context)",
            category="gibby",
            description=(
                "Save images with civitai-compatible generation metadata. "
                "Generation settings come from the linked context; an optional image input overrides its image."
            ),
            is_output_node=True,
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Image.Input("image", optional=True),
                io.String.Input(
                    "filename",
                    default='%time_%basemodelname_%seed',
                    tooltip="filename (available variables: %date, %time, %time_format<format>, %model, %width, %height, %seed, %counter, %counter<padding>, %sampler_name, %steps, %cfg, %scheduler_name, %basemodelname, %denoise)",
                ),
                io.String.Input("path", default='', tooltip="path to save the images (under Comfy's save directory)"),
                io.Combo.Input("extension", options=['png', 'jpeg', 'jpg', 'webp'], default='png', tooltip="file extension/type to save image as"),
                io.Boolean.Input("lossless_webp", default=True, tooltip="if True, saved WEBP files will be lossless"),
                io.Int.Input("quality_jpeg_or_webp", default=100, min=1, max=100, tooltip="quality setting of JPEG/WEBP"),
                io.Boolean.Input("optimize_png", default=False, tooltip="if True, saved PNG files will be optimized (can reduce file size but is slower)"),
                io.Int.Input("counter", default=0, min=0, max=0xffffffffffffffff, tooltip="counter"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, tooltip="denoise value"),
                io.String.Input("time_format", default="%Y-%m-%d-%H%M%S", tooltip="timestamp format"),
                io.Boolean.Input("save_workflow_as_json", default=False, tooltip="if True, also saves the workflow as a separate JSON file"),
                io.Boolean.Input("embed_workflow", default=True, tooltip="if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"),
                io.String.Input(
                    "additional_hashes",
                    default="",
                    tooltip="hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')",
                ),
                io.Boolean.Input("download_civitai_data", default=True, tooltip="Download and cache data from civitai.red to save correct metadata. Allows LoRA weights to be saved to the metadata."),
                io.Boolean.Input("easy_remix", default=True, tooltip="Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.red more seamless."),
                io.Boolean.Input("show_preview", default=True, tooltip="if True, displays saved images in the UI preview"),
                io.String.Input("custom", default="", tooltip="custom string to add to the metadata, inserted into the a111 string before the model hash"),
                io.Boolean.Input("save_image", default=True, tooltip="if True, saves the image to disk"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.String.Output(id="hashes", display_name="hashes", tooltip="Comma-separated list of the hashes to chain with other Image Saver additional_hashes"),
                io.String.Output(id="a111_params", display_name="a1111_params", tooltip="Written parameters to the image metadata"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, filename='%time_%basemodelname_%seed', path='', extension='png',
                lossless_webp=True, quality_jpeg_or_webp=100, optimize_png=False, counter=0, denoise=1.0,
                time_format="%Y-%m-%d-%H%M%S", save_workflow_as_json=False, embed_workflow=True, additional_hashes='',
                download_civitai_data=True, easy_remix=True, show_preview=True, custom='', save_image=True) -> io.NodeOutput:
        # Generation settings from the context instead of widgets.
        ctx = context if isinstance(context, dict) else {}
        has_context = bool(ctx)

        modelname = str(ctx.get("model_name") or '')
        seed_value = int(ctx.get("seed") or 0)
        steps = int(ctx.get("steps") or 20)
        cfg = float(ctx.get("cfg") or 7.0)
        sampler_name = str(ctx.get("sampler") or '')
        scheduler_name = str(ctx.get("scheduler") or 'normal')
        width = int(ctx.get("width") or 512)
        height = int(ctx.get("height") or 512)
        positive = str(ctx.get("positive_prompt") or '')
        negative = str(ctx.get("negative_prompt") or '')

        if image is not None:
            ctx["image"] = image
        images = ctx.get("image")
        if images is None:
            print("Gibby Image Saver: no images found in context or input, nothing to save.")
            return io.NodeOutput(ctx, '', '')

        # Resolve size from the actual images when the context doesn't have it.
        if width == 0 or height == 0:
            img_h, img_w = images.shape[1], images.shape[2]
            width = int(img_w)
            height = int(img_h)

        # LoRA names from the context's lora_stack, appended to the positive prompt.
        lora_stack = ctx.get("lora_stack")
        if isinstance(lora_stack, list):
            tags = []
            for item in lora_stack:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                tags.append(_format_lora_tag(item[0], item[1]))
            if tags:
                positive = f"{positive}, {', '.join(tags)}" if positive else ', '.join(tags)

        metadata = _make_metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, custom, additional_hashes, download_civitai_data, easy_remix)

        path = _make_pathname(path, width, height, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler_name, denoise, custom)
        
        # Only save image if save_image is True
        if not save_image:
            print("Gibby Image Saver: save_image is False, skipping save.")
            return io.NodeOutput(ctx, metadata.final_hashes, metadata.a111_params, ui=None)
        
        # When no context, create minimal metadata; otherwise use full context metadata
        if not has_context:
            minimal_metadata = Metadata(
                modelname="",
                positive="",
                negative="",
                width=width,
                height=height,
                seed=seed_value,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler_name=scheduler_name,
                denoise=denoise,
                custom="",
                additional_hashes="",
                ckpt_path="",
                a111_params="",
                final_hashes=""
            )
            filenames = _save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, cls.hidden.prompt, cls.hidden.extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, minimal_metadata)
        else:
            filenames = _save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, cls.hidden.prompt, cls.hidden.extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, metadata)

        subfolder = os.path.normpath(path)
        ui = None
        if show_preview:
            ui = {"images": [{"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for filename in filenames]}

        return io.NodeOutput(ctx, metadata.final_hashes, metadata.a111_params, ui=ui)
