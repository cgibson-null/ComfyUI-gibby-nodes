import os
import time

import comfy.model_management
from aiohttp import web
from comfy_api.latest import io
from server import PromptServer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# node_id -> "wait" | "stop", set by the frontend's Continue / Stop buttons.
_NODE_STATES = {}


@PromptServer.instance.routes.post("/gibby/pause/action")
async def _pause_action(request):
    data = await request.json()
    node_id = data.get("node_id")
    action = data.get("action")
    if node_id:
        _NODE_STATES[node_id] = action
    return web.json_response({"status": "ok"})


@PromptServer.instance.routes.get("/gibby/pause/ding.mp3")
async def _pause_ding(request):
    file_path = os.path.join(_THIS_DIR, "ding.mp3")
    if os.path.exists(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404)


class GibbyPauseExecution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyPauseExecution",
            display_name="Pause Execution",
            category="gibby/flow",
            search_aliases=["pause", "stop", "break", "wait", "continue", "resume"],
            description="Blocks execution in place until the user clicks Continue (requeues the prompt, pausing nodes let it through) or Stop (interrupts the run); a timeout auto-resolves the pause. Passes input through unchanged.",
            inputs=[
                io.AnyType.Input("input"),
                io.Boolean.Input("pause_flow", default=True, tooltip="When false, passes through without pausing"),
                io.Int.Input("timeout_seconds", default=60, min=1, max=3600, step=1, tooltip="Auto-resolve the pause after this many seconds"),
                io.Combo.Input("on_timeout", options=["continue", "stop workflow"], default="continue", tooltip="What to do when the timeout expires"),
                io.Boolean.Input("emit_sound", default=False, tooltip="Play a ding when execution pauses at this node"),
                io.String.Input("run_id", default="0_manual", tooltip="Managed by the frontend: a run id ending in _continue (the requeue after a Continue click) passes through without pausing"),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.AnyType.Output("output"),
            ],
        )

    @classmethod
    def execute(cls, input, pause_flow, timeout_seconds, on_timeout, emit_sound, run_id):
        # emit_sound is consumed by the node's frontend (it plays the ding), not here.
        unique_id = cls.hidden.unique_id

        if not pause_flow:
            print(f"Gibby Pause: node {unique_id} bypassed (pause_flow is off).")
            return io.NodeOutput(input)

        if run_id.endswith("_continue"):
            print(f"Gibby Pause: node {unique_id} post-edit transit, passing through.")
            return io.NodeOutput(input)

        _NODE_STATES[unique_id] = "wait"
        start_time = time.time()
        print(f"Gibby Pause: node {unique_id} paused for {timeout_seconds}s...")

        while True:
            if _NODE_STATES.get(unique_id, "wait") == "stop":
                print(f"Gibby Pause: node {unique_id} stopped by user.")
                comfy.model_management.interrupt_current_processing()
                return io.NodeOutput(input)

            if time.time() - start_time >= timeout_seconds:
                if ontimeout == "continue":
                    print(f"Gibby Pause: node {unique_id} timeout expired, continuing.")
                    return io.NodeOutput(input)
                print(f"Gibby Pause: node {unique_id} timeout expired, stopping workflow.")
                comfy.model_management.interrupt_current_processing()
                return io.NodeOutput(input)

            time.sleep(0.2)
