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

**Options:** connect options nodes to the `options` input - **Crop-Inpaint options**, **Iterative Upscale options**, **Tiled VAE options**, or **Merge KSampler Options** (dynamic `option1…option10` inputs that append several options lists into one). The sampler applies each option in the list in order: crop-inpaint and iterative upscale run their dedicated paths, and Tiled VAE options switches the encode/decode to the tiled variants (moved here from the sampler's advanced widgets). Iterative Upscale steps through a linear scale path in pixel space (like Impact Pack's Iterative Upscale (Image), simple step mode): each step resizes the image to `initial size * (1 + (factor-1)*i/steps)` - with an optional upscale model (connect Load Upscale Model to the options node; it travels inside the options) - encodes, ksamples and decodes it. Denoise ramps from `start_denoise` (1st step) to `target_denoise` (last step). `mode: total` runs all steps at once; `mode: single` runs one step per queue. The sampler writes `next_step` and the initial image size into the options and returns them from the new `options` output (bottom) - feed it back in to continue: steps at/after `steps` keep upscaling with `target_denoise`, which is how you refine each stage individually. After the final step (`total` mode), the result's color is matched back to the original image with Transfer Color - method and strength are set on the options node, `color_match` disables it. `verbose` prints each step's index, scale, image size and denoise to the console. Step sizes are rounded to the VAE's spatial downscale ratio (8/16/32 by VAE type) so each encode/decode round-trips exactly. With a mask present, Iterative Upscale resizes it with the image and applies it on every step - only the masked area is sampled. Provide **both** Iterative Upscale options and Crop-Inpaint options (e.g. through Merge KSampler Options) for iterative inpaint upscale: the mask regions are cropped per the crop-inpaint options, each crop is upscaled iteratively with the mask applied on every step (inpaint mode and mask scaling from the crop-inpaint options), and the results are composited back into the full image upscaled by the total factor.

## Other nodes

**Lora Loader** - spiriual successor of [rgthree's](https://github.com/rgthree/rgthree-comfy) PowerLoraLoader initially made by Divine. Nodes2.0 compatible and extends functionality.

**Any Switch** - [rgthree's](https://github.com/rgthree/rgthree-comfy) Any Switch but with dynamic number of inputs.

**Pipe Any** - bundles up to 10 `Any` values into a dict. Mostly inspired by [Crystools'](https://github.com/crystian/ComfyUI-Crystools) one, just unified into single node like context instead of in/out split.

**Fast Groups Muter / Bypasser** - like [rgthree's](https://github.com/rgthree/rgthree-comfy) but with cookies - have mass toggle, enable, mute/bypass; better settings window (actual window, not menu) with color picker (not text input).

**Group Header Toggles** - three small buttons (queue / bypass / mute) in the top-right corner of each group header. Active by default, available in settings. Stolen from [rgthree](https://github.com/rgthree/rgthree-comfy).

**Resize Image / Empty Latent (Context)** - creates empty latent or resizes connectd image/mask allowing to keep proportions. Encodes image with mask into latent and assigns latent to context. Always outputs a context - without a context input it creates a new one with width, height and the empty latent.

**Reference Latent (Context)** - faster flux2 referenes, rescale all to 1mp then to `scale`.

**Image Saver (Context)** - saves image with a1111 metadata (civit compatible) created from context values. Ripoff from [ImageSaver](https://github.com/alexopus/ComfyUI-Image-Saver) pack.

## MiniMax H3

Idea of it is to use single reference list for any amount of passes. if you for example decide to 2nd pass upscale with h3, you have to recondition to avoid errors, which would mean duplicating conditioning node and potentially reattaching references. H3 Pipe approach resolves that issue, establishing single source of truth.

**H3 Pipe Create** - builds a config dict (H3 Pipe) from reference images/videos/audios, prompt, and dimensions. Stores raw tensors, no encoding yet. Shamelessly stolen from [H3-Hybrid-Cond](https://github.com/kitsune123150/minimax-h3-hybrid-cond).

**H3 Pipe Apply** - encodes pipe. target_width and target_height allow to override initial settings for first/last frame, which is required for i2v 2nd pass upscale.