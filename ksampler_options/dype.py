from comfy_api.latest import io

from .crop_inpaint import _KSAMPLER_OPTIONS_TYPE

#: Both methods patch the model and the KSampler samples normally.
#: Mirrors dype_helper.DYPE_PATCH_METHODS.
_DYPE_METHODS = ["dype", "sega"]

_MODEL_TYPES = ["auto", "flux", "nunchaku", "qwen", "qwen21", "krea2", "zimage", "anima"]


class GibbyDypeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyDypeOptions",
            display_name="DyPE options",
            category="gibby/flow",
            search_aliases=["dype", "sega", "highres", "upscale", "options"],
            description=(
                "Outputs DyPE options (needs the ComfyUI-DyPE pack). Feed into KSampler (Context): "
                "dype/sega patch the model for high-resolution generation. "
                "The target size comes from the context."
            ),
            inputs=[
                io.Combo.Input("model_type", default="auto", options=_MODEL_TYPES,
                               tooltip="DyPE architecture; auto detects it (krea2 and qwen21 by class name, flux2/klein/flux3 as flux). The native training resolution is derived from it"),
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
                ], tooltip="High-resolution method: the model is patched and the KSampler samples normally."),
                io.Boolean.Input("verbose", default=False, tooltip="Log the patch and the resolved architecture to the console"),
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
