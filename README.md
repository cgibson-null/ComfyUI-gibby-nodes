Bunch of nodes to reduce noodles significantly.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/cgibson-null/ComfyUI-gibby-nodes
```

## Context system

The core idea: a **Context** node is a bag with sampling and metadata related values that can be overridden. Has Any slot added with idea of using Any Pipe node. Every field is optional. Spiritual inheritor of [rgthree's](https://github.com/rgthree/rgthree-comfy) Context.

**Context Loader** - loads model(ckpt or diff)/clip(s up to 4)/vae(s - for audio too), sets prompts and sampling params, and outputs a ready-to-use context. The starting point of most workflows. Detects model category and creates appropriate latent for it (flux2). Spiritual inheritor of [Efficient loader](https://github.com/jags111/efficiency-nodes-comfyui).

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
- create empty latent based on context width and height, respectin model type (flux2).

**Steps:** widget `steps` overrides context's steps and refiner_steps when >0. refiner_steps are used for i2i. start/end inside defined by widget or context step will ignore refiner_steps.

**Start / end step:** three modes per value:
- `0 < value < 1` - fraction of total steps (e.g. 0.5 = halfway)
- `value < 0` - offset from the end (e.g. -5 = last 5 steps)
- `value ≥ 1` - absolute step number

**Denoise:** only applies when an image is present, otherwise forced to 1.0.

**Decode:** on by default. Decodes the output latent to image/audio/both and removes latent from context. Disable to skip decoding in case you're chaining ksamplers (wan high pass).

**Options:** connect options nodes to the `options` input - **Crop-Inpaint options**, **Iterative Options**, **Lora Travel options**, **Tiled VAE options**, **Clear VRAM options**, or **Merge KSampler Options** (dynamic `option1…option10` inputs that append several options lists into one).

**Crop-Inpaint options** - allows crop area per mask, configure boundaries and resolution, and inpaint it at proper size, then paste back onto original.

**Iterative Options** - sample for set amount iterations, can upscale.

Crop-Inpaint options and Iterative Options are inclusive, meaning can be used together. Both ideas from [Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
**Lora Travel options** - dynamically change loras strength per step; str values in node are multipliers for value set in lora loader, unless it's 0 - then they are applied as actual str; can be used per iteration.

**Clear VRAM options** frees VRAM like [Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)'s Clean VRAM Used.

**Tiled VAE options** - self-explanatory.

Supports prompt travel/scheduling, whatever it is called, from a1111/forge webui:

[a:b] - alternates between prompts a and b each step.

[a:b:3] or [a:b:.4] - uses prompt a at start, switches to b at step 3 or total_steps*.4

[a:b:8:4] - crossfade, a up to step 5, step 5 a at str .875 b - .125, step 6 - a:.75, b:.25, step 8 - both .5, step 9 a:.375, b:.625, step 12 - prompt b

[a:b|3] or [a:b|.1] - iteration like [a:b] but instead of alternating each step, does it each 3 seps or total_steps*.1

## Other nodes

**Lora Loader** - spiriual successor of [rgthree's](https://github.com/rgthree/rgthree-comfy) PowerLoraLoader initially made by Divine. Nodes2.0 compatible and extends functionality.

**Any Switch** - [rgthree's](https://github.com/rgthree/rgthree-comfy) Any Switch but with dynamic number of inputs.

**Pipe Any** - bundles up to 10 `Any` values into a dict. Mostly inspired by [Crystools'](https://github.com/crystian/ComfyUI-Crystools) one, just unified into single node like context instead of in/out split.

**Fast Groups Muter / Bypasser / Toggle** - like [rgthree's](https://github.com/rgthree/rgthree-comfy) but with cookies - have mass toggle, enable, mute/bypass; better settings window (actual window, not menu) with color picker (not text input). Toggle merges bot options with a toggle between bypass/mute.

**Group Header Toggles** - three small buttons (queue / bypass / mute) in the top-right corner of each group header. Active by default, available in settings. Stolen from [rgthree](https://github.com/rgthree/rgthree-comfy).

**Resize Image / Empty Latent (Context)** - creates empty latent or resizes connectd image/mask allowing to keep proportions. Encodes image with mask into latent and assigns latent to context. Always outputs a context - without a context input it creates a new one with width, height and the empty latent.

**Reference Latent (Context)** - faster flux2/klein referenes, rescale all to the `megapixels` target then to `scale` (0 = own size). With qwen image 2.1 the size is a multiple of 32 and the prompts are also re-encoded with the images so the latents splice at the vision slots.

**Image Saver (Context)** - saves image with a1111 metadata (civit compatible) created from context values. Ripoff from [ImageSaver](https://github.com/alexopus/ComfyUI-Image-Saver) pack.

**Crop Image (Context)** - (temp name) like native Crop Image, but outputs mask, bbox, pos and neg coords to be used with segm.

**Pause Execution** - allows to pause execution; throws error at first run, ignore it.

## MiniMax H3

Idea of it is to use single reference list for any amount of passes. if you for example decide to 2nd pass upscale with h3, you have to recondition to avoid errors, which would mean duplicating conditioning node and potentially reattaching references. H3 Pipe approach resolves that issue, establishing single source of truth.

**H3 Pipe Create** - builds a config dict (H3 Pipe) from reference images/videos/audios, prompt, and dimensions. Stores raw tensors, no encoding yet. Shamelessly stolen from [H3-Hybrid-Cond](https://github.com/kitsune123150/minimax-h3-hybrid-cond).

**H3 Pipe Apply** - encodes pipe. target_width and target_height allow to override initial settings for first/last frame, which is required for i2v 2nd pass upscale.