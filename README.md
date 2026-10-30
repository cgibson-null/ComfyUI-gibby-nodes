Bunch of nodes to reduce noodles significantly.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/cgibson-null/ComfyUI-gibby-nodes
```

## Context system

The core idea: a **Context** node is a bag with sampling and metadata related values that can be overridden. Has Any slot added with idea of using Any Pipe node. Every field is optional. Spiritual inheritor of [rgthree's](https://github.com/rgthree/rgthree-comfy) Context.

**Context Loader** - loads model(ckpt or diff)/clip/vae(s - for audio too), sets prompts and sampling params, and outputs a ready-to-use context. The starting point of most workflows. Detects model category and creates appropriate latent for it (flux2/flux3/fibo2). Spiritual inheritor of [Efficient loader](https://github.com/jags111/efficiency-nodes-comfyui).

**Context Override** - provides widgets to override context values. Toggles let you choose which categories to override. Useful when you want to separate setting per sampling stage (steps 1-3 higher cfg, 1st prompt, remaining steps - 1cfg, 2nd prompt; etc)

**Sampling Parameters (Context)** - acts like override for sampling parameters.

**Merge contexts** - dynamic `context1…context10` inputs, single context output. Takes the 1st context as base, then goes through the other provided contexts in order and overrides its values with theirs - a value overrides only when it is set (None, zero and empty string keep the base's value).

For ggufs use **Context** and connect loaders to it, then use **Context Override** or **Sampling Parameters (Context)** to provide initial context values.

If **Context Loader** and **Context Override** are provided with image and/or audio, they are encoded using respective VAE, and if both provided - combined into AVlatent. Respective masks also applied, allowing easy i2i or masking audio to generate audio-driven video.

## KSampler (Context)

Unifies native KSampler, KSamplerAdvanced and KSamplerCustom nodes' features.

Use **inputs** to override context values (override happens first, so output context will contain same values)

**Latent** used in priority:
- provided latent;
- encode image and/or audio with provided masks using respective VAE, combine if both;
- create empty latent based on context width and height, respecting model type (flux2/flux3/fibo2).

**Steps:** widget `steps` overrides context's steps and refiner_steps when >0. refiner_steps are used for i2i. start/end inside defined by widget or context step will ignore refiner_steps.

**Start / end step:** three modes per value:
- `0 < value < 1` - fraction of total steps (e.g. 0.5 = halfway)
- `value < 0` - offset from the end (e.g. -5 = last 5 steps)
- `value ≥ 1` - absolute step number

**Denoise:** only applies when an image is present, otherwise forced to 1.0. Always slices the schedule, so start/end step narrow the denoised range instead of cancelling the denoise. flux2 models sample with the Flux2 schedule for the sampled size, denoise takes its low-noise tail exactly like `Flux2Scheduler` -> `SplitSigmasDenoise` (`low_sigmas`); flux3 keeps its own constant shift and the context's scheduler.

**CFG schedule:** optional link-only float input that replaces the context's cfg on every sampling step - one value per step of the context's steps, from `String to Float List`, a float batch, or a single float. Past its end the last value is held for the remaining steps, so `3, 2, 1` over 5 steps samples 3, 2, 1, 1, 1; extra values are ignored. Like the travels, it follows the context's step numbers, so start/end step shift it with the steps they slice (`2, 1` with start_step 1 samples at 1); each crop-inpaint crop samples it from the first value. The context's cfg is left alone, so the verbose log and saved metadata keep it.

**Decode:** on by default. Decodes the output latent to image/audio/both and removes latent from context. An image whose size is not a multiple of the VAE's downscale ratio is pre-scaled to it before encoding, and the decoded image is stretched back to the original size, so the content stays aligned with the original. Disable to skip decoding in case you're chaining ksamplers (wan high pass).

**Latent output:** the latent the returned image was sampled from - the sampled latent, or the last iterative step's latent. Crop-Inpaint samples each crop at its own size, so its latent output is the latent of the inpainted area (the last region's when the mask splits into several), at the size that crop was sampled at, never the whole image.

**Options:** connect options nodes to the `options` input - **Crop-Inpaint options**, **Iterative Options**, **Lora Travel options**, **Tiled VAE options**, **Clear VRAM options**, **Color Match options**, **DyPE options**, or **Merge KSampler Options** (dynamic `option1…option10` inputs that append several options lists into one).

**Crop-Inpaint options** - allows crop area per mask, configure boundaries and resolution, and inpaint it at proper size, then paste back onto original.

**Iterative Options** - sample for set amount iterations, can upscale; an optional target size image overrides the upscale factor and sets the final size. `latent_upscale` grows the latent between steps instead of decode -> upscale -> encode, so the VAE runs at the ends of the run; the upscale model is an image model and is not used then.

Crop-Inpaint options and Iterative Options are inclusive, meaning can be used together. Both ideas from [Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
**Lora Travel options** - dynamically change loras strength per step; str values in node are multipliers for value set in lora loader, unless it's 0 - then they are applied as actual str; can be used per iteration.

**Clear VRAM options** frees VRAM like [Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)'s Clean VRAM Used.

**Color Match options** - after sampling (a run from a provided image, crop-inpainting, or the final iterative step) the result's color is matched back to the original with Transfer Color (`strength` 0 = no match).

**DyPE options** - needs the [DyPE](https://github.com/wildminder/ComfyUI-DyPE) pack. dype/sega patch the model for high-resolution generation, so every other feature applies to them: the patch is re-applied per sample at the size actually being sampled, which makes it combine with the iterative upscale, and it never leaves that step.

**Tiled VAE options** - self-explanatory.

Supports prompt travel/scheduling, whatever it is called, from a1111/forge webui:

[a:b] - alternates between prompts a and b each step.

[a:b:3] or [a:b:.4] - uses prompt a at start, switches to b at step 3 or total_steps*.4

[a:b:8:4] - crossfade, a up to step 5, step 5 a at str .875 b - .125, step 6 - a:.75, b:.25, step 8 - both .5, step 9 a:.375, b:.625, step 12 - prompt b

[a:b|3] or [a:b|.1] - iteration like [a:b] but instead of alternating each step, does it each 3 steps or total_steps*.1

Groups nest, and only square brackets travel: `(prompt:1.5)` stays ComfyUI's weight syntax and is passed through untouched, colons and all.

The context conditions with the first prompt of the plan - never the literal `[a:b:3]` text - and only the prompt that travels is encoded; the other keeps the context's own conditioning. Each unique prompt of the plan is encoded once and cached with its text encoder (last 8 per clip), so re-running an unchanged prompt skips the text encoder entirely.

## Other nodes

**Lora Loader** - spiritual successor of [rgthree's](https://github.com/rgthree/rgthree-comfy) PowerLoraLoader initially made by Divine. Nodes2.0 compatible and extends functionality.

**Any Switch** - [rgthree's](https://github.com/rgthree/rgthree-comfy) Any Switch but with dynamic number of inputs.

**Any Batch/List** - dynamic-input batch of any type: `Batch` concatenates the connected inputs the way Batch Images does (tensors along dim 0, conditioning and other lists joined), `List` outputs them as a Python list like `Make List (Any)` from [Impact](https://github.com/ltdrdata/ComfyUI-Impact-Pack).

**Any Count** - how many items a value holds: the batch size of a tensor or latent, the length of a conditioning or a list, 1 for a single object. Nothing connected and a null both count 0, so an optional branch can be measured without being wired - the case [Essentials'](https://github.com/cubiq/ComfyUI_essentials) `Batch Count` can't do, as its input is required.

**Pipe Any** - bundles up to 10 `Any` values into a dict. Mostly inspired by [Crystools'](https://github.com/crystian/ComfyUI-Crystools) one, just unified into single node like context instead of in/out split.

**Fast Groups Muter / Bypasser / Toggle** - like [rgthree's](https://github.com/rgthree/rgthree-comfy) but with cookies - have mass toggle, enable, mute/bypass; better settings window (actual window, not menu) with color picker (not text input). Toggle merges bot options with a toggle between bypass/mute.

**Group Header Toggles** - three small buttons (queue / bypass / mute) in the top-right corner of each group header. Active by default, available in settings. Stolen from [rgthree](https://github.com/rgthree/rgthree-comfy).

**Resize Image / Empty Latent (Context)** - creates empty latent or resizes connected image/mask allowing to keep proportions. Optional vae overrides the context's vae; encoded_latent outputs the image encoded with it (empty_latent when no image or vae). Always outputs a context - without a context input it creates a new one with width, height and the empty latent. `Enable preview` toggle (off by default, hidden while load image / crop image is on) writes and shows the output image's temp preview in the node on execution; off writes no preview file at all (while load / crop is on the preview is always written - downstream crop views read the context's image from it).

**Reference Latent (Context)** - faster flux2/klein/flux3/fibo2 references, rescale all to the `megapixels` target then to `scale` (0 = own size), floored to the VAE's downscale ratio (2x for qwen image 2.1, so the latents splice at the vision slots). `ref_latents` (on) VAE-encodes them and references them on the conditioning - the edit path the edit models were trained for; turn it off for a t2i model that only has the latent-concat path by accident (krea). A text encoder with a vision tower (qwen image, qwen image 2.1, krea2, mage, boogu, ideogram4, fibo 2.0) also sees the images and the prompts are re-encoded with them, at `vl_megapixels` (0 = the VAE-sized image; qwen image 2.1 always shares one size so every vision slot covers 2x2 latents, fibo 2.0 because its tokenizer does its own 768px / Qwen3-VL-grid resize) - that is how a model with no edit path conditions on a reference at all. Flux3's encoder is text-only by training, so it takes latents alone. A negative the context carries is kept as it is at the context's `cfg` 1 (it only gains the references) - cfg 1 never runs it, so a turbo model (fibo 2.0) pays for no re-encode and a negative built outside the prompts survives the node; only a negative that exists as prompt text alone becomes the zeroed positive, like KSampler (Context) makes it. HiDream-O1 and SenseNova read the reference as a raw image rather than a latent, so they get the resized image itself and need no VAE. `ref_ctx_img` additionally references the context's own image at its own size (always when no image is linked, otherwise only when enabled); with no context image the first image becomes the context image and is stored back on the output context. `tiled_vae_options` takes the same Tiled VAE options node the sampler takes and encodes a reference at or above `above_megapixels` in tiles instead of whole - what a huge image costs with `megapixels` 0 or `ref_ctx_img`; only that option of the list is read. A media pipe's reference images are referenced too - the wired images append to them in the output pipe. Without a context input it is just a media pipe creator: the wired images append to the pipe and the context passes through empty.

**Mask/Segment (Context)** - detects or segments objects on the image (connected image wins over the context one) and stores the resulting image and mask in the context; the context's model and clip pass through untouched. A mode selector reveals the widgets of the chosen approach: `bbox` (ultralytics detector), `segm` (SAM refined by the detector's boxes), `sam3` (the selected SAM3 checkpoint, which can track through a batch of frames; the tracker is seeded by the first-frame detection, or by a connected mask instead, which skips that detection outright - one mask per object, so a merged mask tracks as a single object). The mask is stored as the detection produced it - clean it up, grow it or preview it with **Mask Edit (Context)**. Also outputs crop info for pasting results back with **Image Paste By Mask (Batch) (Context)**.

**Mask Edit (Context)** - the mask manipulation the detection and crop nodes used to carry, in one node. The mask is cleaned first (`remove_isolated_pixels` drops isolated specks, `fill_holes` closes the holes the mask encloses), its edge is then feathered by `mask_blur` pixels and it is grown and inverted (`mask_grow` / `mask_invert`, the native Grow Mask / Invert Mask), so growing enlarges what is left instead of the specks. `remove_bg` replaces the unmasked area with `rembg_color` (an alpha 0 color makes it transparent instead - RGBA output). `preview` shows the image with the mask tinted in `mask_color`; a batch of frames previews as a video at `preview_fps`. Connected image/mask win over the context's; the output context carries the edited image and mask. A crop info that describes the image being edited is refreshed to the edited mask, so a paste back honors the edit; a crop's own - whose mask is the full frame the crop came from - does not match and passes through untouched, so **Image Paste By Mask (Batch) (Context)** still pastes onto the frame the crop was taken from.

**Image Crop By Mask (Batch) (Context)** - crops each image by its own mask (image i by mask i); every crop is scaled up until it fills the total size, so subjects keep a consistent scale across the batch - when a crop's shape differs from the canvas its edges are trimmed. Optional temporal stabilization for video batches: constant crop size per shot and a smoothed crop center, with cuts detected from mask jumps. The mask is cropped as it arrives - grow, invert, feather or knock the background out of the crop with **Mask Edit (Context)**. Outputs the crop info for pasting back; with a context, its image and mask are used when none are connected and the output context carries the cropped image, mask and crop info.

**Image Paste By Mask (Batch) (Context)** - pastes images back onto the originals by mask: from a crop info (the exact inverse of the crop), or standalone with images_original and masks, where each mask's bbox is the paste area; when the batches differ, the smaller size wins. With composite on it instead pastes like Image Composite Masked: the pasted image is resized to the original's size when it differs and a source-sized mask is applied to it (no mask pastes opaquely).

**Split Tiles (Context)** - splits the context/connected image into a batch of uniform, overlapping tiles (rows x cols, overlap as a fraction of the tile size) and outputs the tiles, a matching batch of masks for the Crop-Inpaint options, and the tiling info. The output context carries the tiling info too. Stills only.

**Combine Tiles (Context)** - recombines a batch of tiles from the tiling info into the image: the layout is scaled to the tiles' size (which may be larger than at the split - the upscaled case) and the tiles are crossfaded across the overlap bands. Pair with Split Tiles to sample or upscale a still tile by tile.

**Image Saver (Context)** - saves image with a1111 metadata (civit compatible) created from context values. `save_metadata` off writes no generation-parameters block and does no hashing or Civitai lookups; the workflow embed and the JSON sidecar keep their own toggles. Ripoff from [ImageSaver](https://github.com/alexopus/ComfyUI-Image-Saver) pack.

**SeedVR2 Upscale (Context)** - the entire [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) upscale in one node: resize, pad, tiled VAE encode, sample with the encoded latent as the conditioning, decode, then align and color-correct the result to the resized input - replaces Pre-Process SeedVR2 Input, Apply SeedVR2 Conditioning, Post-Process SeedVR2 Output, the KSampler and the tiled VAE nodes. Model and VAE lists are filtered to the seedvr files; `None` takes them from the context. Target size is `megapixels` (0 = the input's own size) times `scale_factor`, like **Resize Image / Empty Latent (Context)**, and a `target_size_image` overrides both with its own size, like **Iterative Options**. `chunking_mode` samples a video a temporal chunk at a time (none / auto sized by free VRAM / manual `frames_per_chunk`) with `temporal_overlap` crossfaded at the merge - Split and Merge SeedVR2 Latents. Sampling is euler/simple at cfg 1 and denoise 1.0, `steps` 1 - the step the model was distilled for. The tile widgets are VAE Encode (Tiled)'s and drive the encode and the decode. `clean_vram` frees VRAM before the model loads (off by default).

## MiniMax H3

Idea of it is to use single reference list for any amount of passes. if you for example decide to 2nd pass upscale with h3, you have to recondition to avoid errors, which would mean duplicating conditioning node and potentially reattaching references. H3 Pipe approach resolves that issue, establishing single source of truth.

**H3 Pipe Create** - builds a config dict (H3 Pipe) from reference images/videos/audios, keyframes, prompt, and dimensions. Stores raw tensors, no encoding yet. The H3 Pipe is a media pipe - `ref_images`, `keyframes` with their positions, `ref_videos` (+ soundtracks), `ref_audios` - plus the h3 params, so the same dict feeds Reference Latent (Context) and Generate as well. Keyframes anchor the video at given positions: `indices` takes a comma-separated list - an integer is a frame index (negative counts from the end, -1 = last), a decimal is a percentage of the video (0.5 = middle); `even_distribution` spreads the full batch (first/last frame included) evenly from 0 to the last frame; `loop` anchors a duplicate of the first keyframe at the last frame. first/last frame join the batch at its start/end. Shamelessly stolen from [H3-Hybrid-Cond](https://github.com/kitsune123150/minimax-h3-hybrid-cond).

**H3 Pipe Apply** - encodes pipe. target_width and target_height override the pipe's width and height for the keyframes and the latent, which is required for i2v 2nd pass upscale. `drop_ref_images` / `drop_keyframes` / `drop_ref_videos` / `drop_ref_video_audios` / `drop_ref_audios` prune the pipe before applying; without clip/vae (wired or from the context) it just outputs the pruned h3_pipe.

## LLM Connect

Nodes for talking to a llama.cpp server / llama-swap proxy (OpenAI-compatible API); Unsloth Studio works too.

**Connectivity** - server URL + model dropdown (auto-fetched from the server's /v1/models, cached per URL, Reconnect button) + optional API key. `keep_alive` 0 unloads the model right after each response; other values are the server's idle policy.

**Sampling Options** - per-request sampler settings (temperature, top_k/top_p/min_p, mirostat, repeat penalty, stop, n_predict, seed, reasoning budget/effort), each field gated by an enable toggle - only enabled fields are sent.

**Unsloth Load Options** - per-load parameters (context size, speculative/MTP, KV-cache quant, extra server flags); Generate applies them via Unsloth's /load API (local copy first, else the Hub repo); ignored on llama-swap, where those flags are fixed at server start (config.yaml cmd, one entry per variant).

**Generate** - single-shot text generation with auto-growing image/video reference slots (videos sampled to still frames, first + last always included); prints a tokens/sec line with the time it took for every run. Can take a media pipe - its refs and keyframes are sent to the model, labeled as such - and outputs a media pipe carrying the wired refs and the generated text as its prompt.