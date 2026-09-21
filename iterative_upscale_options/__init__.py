from comfy_api.latest import io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyIterativeUpscaleOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyIterativeUpscaleOptions",
            display_name="Iterative Options",
            category="gibby/flow",
            search_aliases=["upscale", "iterative", "hires", "options", "config"],
            description="Outputs iterative upscale options. Feed into KSampler (Context) options input; the upscale model travels inside the options.",
            inputs=[
                io.Float.Input("upscale_factor", default=2.0, min=1.0, max=100.0, step=0.1, tooltip="Total resolution multiplier across all steps"),
                io.Int.Input("steps", default=2, min=1, max=100, step=1, tooltip="Number of iterative upscale steps; with no image in the context the 1st step is the basic generation (denoise 1.0)"),
                io.Float.Input("start_denoise", default=0.6, min=0.0, max=1.0, step=0.01, tooltip="Denoise of the first step; ramps down to target_denoise by the last step"),
                io.Float.Input("target_denoise", default=0.3, min=0.0, max=1.0, step=0.01, tooltip="Denoise of the last step; kept for steps beyond 'steps'"),
                io.Combo.Input("upscale_method", default="lanczos", options=["bilinear", "area", "nearest", "lanczos"], tooltip="Resize method used on each step when no upscale model is connected"),
                io.Boolean.Input("mode", default=True, label_on="total", label_off="single", tooltip="total: run all remaining steps at once. single: run one step per KSampler (Context) run."),
                io.Boolean.Input("verbose", default=False, tooltip="Print each step's index, scale, image size and denoise to the console"),
                io.UpscaleModel.Input("upscale_model", optional=True, tooltip="Optional upscale model (Load Upscale Model) used on each step"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, upscale_factor=2.0, steps=2, start_denoise=0.6, target_denoise=0.3,
                upscale_method="lanczos", mode=True, verbose=False, upscale_model=None):
        option = {
            "type": "iterative_upscale",
            "upscale_factor": upscale_factor,
            "steps": steps,
            "start_denoise": start_denoise,
            "target_denoise": target_denoise,
            "upscale_method": upscale_method,
            "mode": "total" if mode else "single",
            "verbose": verbose,
            "upscale_model": upscale_model,
        }
        return io.NodeOutput([option])
