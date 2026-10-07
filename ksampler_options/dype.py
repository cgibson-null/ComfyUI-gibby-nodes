from comfy_api.latest import io

from .crop_inpaint import _KSAMPLER_OPTIONS_TYPE

#: DyPE method families: the first three patch the model, the last three take
#: over sampling. Mirrors dype_helper.DYPE_PATCH_METHODS / DYPE_CASCADE_METHODS.
_DYPE_METHODS = ["dype", "sega", "spa", "pixelrush", "freescale", "hiflow"]

_MODEL_TYPES = ["auto", "flux", "nunchaku", "qwen", "qwen21", "krea2", "zimage", "anima"]


class GibbyDypeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyDypeOptions",
            display_name="DyPE options",
            category="gibby/flow",
            search_aliases=["dype", "sega", "spa", "pixelrush", "freescale", "hiflow", "highres", "upscale", "options"],
            description=(
                "Outputs DyPE options (needs the ComfyUI-DyPE pack). Feed into KSampler (Context): "
                "dype/sega/spa patch the model for high-resolution generation, "
                "pixelrush/freescale/hiflow take over sampling and cascade the context's latent "
                "(the context mask keeps everything outside it from the pre-cascade latent). "
                "The target size comes from the context."
            ),
            inputs=[
                io.Combo.Input("model_type", default="auto", options=_MODEL_TYPES,
                               tooltip="DyPE architecture; auto detects it (krea2 and qwen21 by class name, flux2/klein as flux). The native training resolution is derived from it"),
                io.DynamicCombo.Input("method", options=[
                    io.DynamicCombo.Option("dype", [
                        io.Combo.Input("dype_method", default="vision_yarn", options=["vision_yarn", "yarn", "ntk", "pi", "base"],
                                       tooltip="Position encoding extrapolation method; vision_yarn has the best aspect-ratio robustness"),
                        io.Boolean.Input("yarn_alt_scaling", default=False, label_on="Anisotropic (High-Res)", label_off="Isotropic (Stable Default)",
                                         tooltip="[yarn only] Scale H and W independently (can stretch) instead of uniformly. Ignored by vision_yarn"),
                        io.Boolean.Input("enable_dype", default=True, label_on="dynamic", label_off="static",
                                         tooltip="What stays switched on. dynamic: the extrapolation is re-scaled at every denoising step (the DyPE algorithm) and the noise schedule shifts with the target resolution. static: the chosen method applied at one fixed scale with the model's default noise schedule - plain YaRN/NTK extrapolation, the dynamic part off"),
                        io.Float.Input("dype_start_sigma", default=1.0, min=0.0, max=1.0, step=0.01,
                                       tooltip="When the modulation starts decaying (1.0 = from the start)"),
                        io.Float.Input("dype_scale", default=2.0, min=0.0, max=8.0, step=0.1, tooltip="Modulation magnitude"),
                        io.Float.Input("dype_exponent", default=2.0, min=0.0, max=1000.0, step=0.1,
                                       tooltip="Decay strength: 2.0 for 4K+, 1.0 for 2K-3K, 0.5 just above native; raise to 3-4 against speckle"),
                        io.Float.Input("base_shift", default=0.5, min=0.0, max=10.0, step=0.01, tooltip="Noise-schedule base shift (mu)"),
                        io.Float.Input("max_shift", default=1.15, min=0.0, max=10.0, step=0.01, tooltip="Noise-schedule max shift (mu); Qwen-Image 2.1 wants its own native value"),
                    ]),
                    io.DynamicCombo.Option("sega", [
                        io.Combo.Input("sega_method", default="sega", options=["sega", "ntk"],
                                       tooltip="sega: NTK base + spectral per-dimension mscale. ntk: base NTK only"),
                        io.Float.Input("mscale_alpha", default=0.15, min=0.0, max=1.0, step=0.01, tooltip="Spectral redistribution amplitude"),
                        io.Float.Input("mscale_beta", default=1.5, min=0.0, max=10.0, step=0.1, tooltip="tanh sharpness of the redistribution"),
                        io.Float.Input("mscale_min", default=1.0, min=0.1, max=2.0, step=0.05, tooltip="Floor of the per-frequency mscale"),
                        io.Float.Input("spread_min", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Spectral spread on the early steps"),
                        io.Float.Input("spread_max", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Spectral spread on the late steps"),
                        io.Float.Input("spread_alpha", default=1.5, min=0.1, max=5.0, step=0.1, tooltip="Spread schedule non-linearity"),
                        io.Combo.Input("base_mscale_formula", default="power_res", options=["power_res", "log_res"]),
                        io.Float.Input("base_mscale_coefficient", default=0.08, min=0.0, max=1.0, step=0.01, tooltip="Kappa of the base mscale (paper 0.08)"),
                        io.Float.Input("base_shift", default=0.5, min=0.0, max=10.0, step=0.01),
                        io.Float.Input("max_shift", default=1.15, min=0.0, max=10.0, step=0.01),
                    ]),
                    io.DynamicCombo.Option("spa", [
                        io.Int.Input("bundle_size", default=0, min=0, max=256, step=1,
                                     tooltip="Tokens per bundle (paper N): 0=auto, 1=off, 3 at 2K, 5 at 4K. No-op inside the model's trained extent"),
                        io.Float.Input("spa_start_sigma", default=1.0, min=0.0, max=1.0, step=0.05,
                                       tooltip="Only run SPA while sigma is above this (1.0 = no gating)"),
                        io.Int.Input("spa_steps", default=3, min=0, max=100, step=1,
                                     tooltip="Leading denoising steps SPA runs on; 0 = every step (slower)"),
                        io.String.Input("spa_layer_filter", default="",
                                        tooltip="Restrict to layers, e.g. 0-18,38-57. Empty = every layer"),
                        io.Boolean.Input("proportional_attention", default=False,
                                         tooltip="HRDiT proportional attention scaling for long sequences; no-op at or below 1024px"),
                    ]),
                    io.DynamicCombo.Option("pixelrush", [
                        io.Float.Input("cfg", default=7.0, min=0.0, max=20.0, step=0.1, tooltip="Guidance of the refinement passes (0 = use the context cfg)"),
                        io.Int.Input("cascade_stages", default=1, min=1, max=5, step=1, tooltip="Number of 2x stages: 1=2x, 2=4x, 3=8x"),
                        io.Int.Input("k_timestep", default=249, min=1, max=999, step=1, tooltip="Partial inversion timestep; must align with the model schedule"),
                        io.Float.Input("noise_lambda", default=0.95, min=0.0, max=1.0, step=0.01, tooltip="Weight of the model's prediction (0.95 = 95% prediction + 5% noise)"),
                        io.Combo.Input("noise_injection", default="slerp", options=["slerp", "additive"], tooltip="slerp = paper formula, additive = legacy"),
                        io.Float.Input("overlap", default=0.5, min=0.0, max=0.75, step=0.05, tooltip="Patch overlap; blends the seams"),
                        io.Float.Input("gaussian_sigma", default=24.0, min=1.0, max=128.0, step=0.5, tooltip="Patch feather sigma (rule of thumb: patch_size / 5)"),
                        io.Int.Input("patch_h", default=0, min=0, max=512, step=8, tooltip="Latent patch height (0 = native size)"),
                        io.Int.Input("patch_w", default=0, min=0, max=512, step=8, tooltip="Latent patch width (0 = native size)"),
                    ]),
                    io.DynamicCombo.Option("freescale", [
                        io.Float.Input("cfg", default=7.5, min=0.0, max=20.0, step=0.1, tooltip="Guidance (0 = use the context cfg)"),
                        io.Int.Input("steps", default=50, min=0, max=200, step=1, tooltip="Sampling steps per cascade level (0 = the context steps)"),
                        io.Int.Input("target_resolution", default=2048, min=1024, max=8192, step=128, tooltip="Target size; the cascade doubles until it is reached"),
                        io.Int.Input("noise_timestep", default=700, min=1, max=999, step=1, tooltip="Re-noising timestep: lower keeps more of the input image"),
                        io.Boolean.Input("fast_mode", default=True, tooltip="4 global attention windows: faster, slightly lower quality"),
                    ]),
                    io.DynamicCombo.Option("hiflow", [
                        io.Float.Input("denoise", default=1.0, min=0.05, max=1.0, step=0.05, tooltip="Img2img strength of a content latent; 1.0 regenerates from noise"),
                        io.Float.Input("cfg", default=3.5, min=0.0, max=20.0, step=0.1, tooltip="Base-stage guidance (0 = use the context cfg); leave at 1.0 for guidance-free models"),
                        io.Int.Input("steps", default=30, min=0, max=200, step=1, tooltip="Base-stage steps; their predictions form the reference trajectory (0 = the context steps)"),
                        io.Float.Input("guidance", default=4.5, min=0.0, max=20.0, step=0.1, tooltip="Guided-stage guidance (paper 4.5-6)"),
                        io.Int.Input("steps_per_stage", default=16, min=1, max=50, step=1, tooltip="Guided steps per cascade stage"),
                        io.Float.Input("tau", default=0.6, min=0.05, max=0.95, step=0.05, tooltip="Stage-entry noise level: lower preserves more of the base image"),
                        io.Float.Input("filter_ratio", default=0.2, min=0.05, max=0.95, step=0.05, tooltip="Butterworth low-pass cutoff of the direction alignment"),
                        io.Float.Input("alpha_scale", default=1.0, min=0.0, max=2.0, step=0.05, tooltip="Direction-alignment strength"),
                        io.Float.Input("beta_scale", default=0.5, min=0.0, max=2.0, step=0.05, tooltip="Acceleration-alignment strength"),
                        io.Combo.Input("upsampling", default="latent", options=["latent", "pixel"], tooltip="Per-step reference upsample: latent bicubic, or decode-sharpen-encode"),
                        io.Float.Input("sharpen", default=1.0, min=0.0, max=3.0, step=0.05, tooltip="Unsharp on the stage anchor; 0 for turbo models"),
                        io.Float.Input("scale_factor", default=2.0, min=0.25, max=8.0, step=0.05,
                                       tooltip="Output scale of the context latent: 2 doubles each side, 1 refines in place, 0.5 halves"),
                    ]),
                ], tooltip="High-resolution method. dype/sega/spa patch the model (the KSampler samples normally). pixelrush/freescale/hiflow cascade the context's latent instead of sampling it."),
                io.Boolean.Input("verbose", default=False, tooltip="Log the patch, the resolved architecture and the cascade result to the console"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, method, model_type="auto", verbose=False, **kwargs):
        # A DynamicCombo arrives as {"method": <selection>, **its widgets}
        if isinstance(method, dict):
            params = dict(method)
            method = params.pop("method", "dype")
        else:
            params = {}
        option = {
            "type": "dype",
            "method": method if method in _DYPE_METHODS else "dype",
            "model_type": model_type,
            "verbose": verbose,
        }
        option.update(params)
        return io.NodeOutput([option])
