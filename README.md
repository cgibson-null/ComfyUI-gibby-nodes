# Standalone Power Lora Loader

A self-contained "Power Lora Loader"-style node for ComfyUI. No rgthree-comfy
required, and it's built to work correctly with Nodes 2.0.

## Install (no coding required)

1. Find your ComfyUI folder, then go into `custom_nodes` inside it.
2. Copy this whole `standalone_power_lora_loader` folder into `custom_nodes`,
   so you end up with:
   `ComfyUI/custom_nodes/standalone_power_lora_loader/__init__.py`
3. Restart ComfyUI completely (close it and reopen, don't just refresh the
   browser tab).
4. Right-click the canvas → Add Node → loaders → **Standalone Power Lora
   Loader**. Or just double-click the canvas and search "Standalone Power
   Lora Loader".

## How to use it

- Connect a `MODEL` and `CLIP` into the node (same as any lora loader).
- You'll see 10 numbered rows, each with:
  - a checkbox (on/off)
  - a dropdown to pick a lora file
  - a strength slider
- Leave a row's dropdown on "None" to ignore it - it won't do anything even
  if its checkbox is on.
- Output `MODEL` and `CLIP` connect onward just like the original node.

## Want more than 10 lora slots?

Open `__init__.py` in any text editor (even Notepad), find this line near
the top:

```python
NUM_LORA_SLOTS = 10
```

Change `10` to whatever number you want, save, and restart ComfyUI.

## Why it only has fixed slots, not an infinite "+ Add Lora" button

ComfyUI's current node system doesn't yet have a clean, official way to grow
a *group* of fields (checkbox + dropdown + slider) together as a single row
on demand - that specific feature is still an open request from custom node
developers to the ComfyUI team. Every widget in this node is a plain,
built-in ComfyUI widget, which is what keeps it safe from breaking every
time ComfyUI's frontend changes.
