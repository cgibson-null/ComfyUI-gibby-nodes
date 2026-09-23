"""Manual end-to-end test for the LLM Connect subpack.

Run from the pack root with the ComfyUI venv python (comfy_api must be
importable):

    <ComfyUI>/venv/Scripts/python.exe llm/test_llm_manual.py

Covers:
  * The pack loads standalone and the four LLM nodes register with the
    V3 schema (node ids, category, custom socket types, autogrow refs).
  * A real single-shot generation through GibbyGenerate against the
    running llama-swap server.
"""

import importlib.util
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent
# ComfyUI root, so comfy_api imports stand alone (ComfyUI adds it itself
# when it loads the pack).
sys.path.insert(0, str(PACK_DIR.parent.parent))

spec = importlib.util.spec_from_file_location(
    "gibby_nodes", PACK_DIR / "__init__.py",
    submodule_search_locations=[str(PACK_DIR)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["gibby_nodes"] = mod
spec.loader.exec_module(mod)

from gibby_nodes.llm import (
    GibbyConnectivity,
    GibbyGenerate,
    GibbyLoadOptions,
    GibbySamplingOptions,
)

SWAP = "http://127.0.0.1:8080"
MODEL = "qw3.8"


def main():
    classes = (GibbyConnectivity, GibbySamplingOptions, GibbyLoadOptions, GibbyGenerate)
    for cls in classes:
        schema = cls.define_schema()
        assert schema.category == "gibby/llm connect", cls.__name__
    assert GibbyConnectivity.define_schema().outputs[0].get_io_type() == "LLAMACPP_CONNECTIVITY"
    assert GibbySamplingOptions.define_schema().outputs[0].get_io_type() == "LLAMACPP_OPTIONS"
    autogrows = [i.template for i in GibbyGenerate.define_schema().inputs
                 if getattr(i, "template", None) is not None]
    assert any(t.names[:1] == ["images_1"] for t in autogrows)
    assert any(t.names[:1] == ["video_1"] for t in autogrows)
    print("[ok] four LLM nodes registered under gibby/llm connect")

    # Real single-shot run through Generate.
    out = GibbyGenerate.execute(
        connectivity={"url": SWAP, "model": MODEL, "keep_alive": 5,
                      "keep_alive_unit": "minutes", "api_key": ""},
        system="You are an AI artist.",
        prompt="Say hello in five words.",
        think=False,
        clear_vram=False,
        format="text",
        frames_per_video=4,
        max_image_size=0,
        options={"enable_n_predict": True, "n_predict": 16, "debug": False},
        images={},
        video={},
    )
    assert out[0], out.args
    print("[ok] live generation through GibbyGenerate")


if __name__ == "__main__":
    main()
    print("ALL OK")
