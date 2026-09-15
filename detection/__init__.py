"""
Detection node
--------------
Detects / segments objects on the context image (or a directly connected
image, which overrides it) and writes the result back into the context.
The context's model and clip are never touched.

Modes:
- bbox: the ultralytics bbox detector (models/ultralytics/bbox, same list
  as the Impact Subpack's UltralyticsDetectorProvider /bbox entries);
  prompt inputs are ignored.
- segm: a SAM model (models/sams, SAM files only). Prompted by the bboxes,
  falling back to the bbox detector's findings.
- sam3: the SAM3 / SAM3.1 checkpoint selected in the node (checkpoints
  folder), loaded with the native checkpoint loader together with its own
  SAM3 CLIP - text detection from the sam3_prompt (encoded like CLIP Text
  Encode) and segmentation from the bboxes / point prompts.

The context's model and clip (the image/video generation model) pass
through untouched - the detection models are loaded by this node itself.

With preview on, the node displays a preview of the detection image with
the mask tinted in the mask color blended over it (KJNodes
ImageAndMaskPreview style); the image output and the context keep the
plain image. A batch of frames (video) previews as a video at
preview_fps, encoded the way the native Create Video / Save Video nodes
do it.

The prompts (bboxes, pos/neg coords) use the same formats as SAM3 Detect,
and also accept KJNodes BBOX (startX/startY/endX/endY) boxes. When a
prompt socket is unconnected, the value carried by the context is used
(a previous Detection node stores its bboxes and mask there).
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import nodes
import comfy.model_management
import comfy.sd
import comfy.utils
from PIL import Image
from comfy_api.latest import io, ui, Types
from comfy_extras.color_util import hex_to_rgb
from comfy_extras.nodes_video import CreateVideo

from ..context import _CONTEXT_TYPE

# Fixed confidence for the bbox detector and SAM prompts (the sam3 modes
# have their own sam3_threshold widget)
_DETECT_THRESHOLD = 0.5


def _bbox_detector_models():
    # Same /bbox entries UltralyticsDetectorProvider offers
    try:
        names = folder_paths.get_filename_list("ultralytics_bbox")
    except Exception:
        # Subpack not loaded: scan the default location directly
        names = []
        d = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
        if os.path.isdir(d):
            names = sorted(f for f in os.listdir(d) if f.lower().endswith((".pt", ".onnx")))
    opts = ["bbox/" + n for n in names]
    return opts or ["(no bbox models found)"]


def _segm_models():
    # SAM files from the sams folder (the Impact Pack's SAMLoader list, SAM only)
    try:
        opts = [x for x in folder_paths.get_filename_list("sams")
                if "sam" in x.lower() and x.endswith((".pt", ".pth", ".safetensors"))]
    except Exception:
        opts = []
    return opts or ["(no models found)"]


def _sam3_models():
    try:
        all_models = folder_paths.get_filename_list("checkpoints")
    except Exception:
        all_models = []
    # Checkpoints are not recognizable by name, so prefer the sam3-named ones
    # but fall back to the full list so a differently named ckpt stays reachable
    opts = [m for m in all_models if "sam3" in m.lower()]
    return opts or all_models or ["(no models found)"]


def _resolve_bbox_path(model_name):
    rest = model_name[5:] if model_name.startswith("bbox/") else model_name
    try:
        path = folder_paths.get_full_path("ultralytics_bbox", rest)
    except Exception:
        path = None
    if path is None:
        path = os.path.join(folder_paths.models_dir, "ultralytics", "bbox", rest)
        path = path if os.path.isfile(path) else None
    return path


def _parse_points(coords):
    # '[{"x": int, "y": int}, ...]' (KJNodes PointsEditor / SAM3 Detect format)
    if not coords or not coords.strip():
        return []
    try:
        pts = json.loads(coords)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in point coords: {e}")
    if not isinstance(pts, list):
        raise ValueError("Point coords must be a JSON array of {\"x\": ..., \"y\": ...}")
    out = []
    for p in pts:
        if isinstance(p, dict):
            try:
                out.append([float(p["x"]), float(p["y"])])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _bbox_xyxy(d):
    # x/y/width/height (SAM3 / native BOUNDING_BOX) or startX/startY/endX/endY (KJNodes BBOX)
    try:
        if isinstance(d, dict):
            if "startX" in d:
                x1, y1, x2, y2 = float(d["startX"]), float(d["startY"]), float(d["endX"]), float(d["endY"])
            else:
                x1, y1 = float(d["x"]), float(d["y"])
                x2, y2 = x1 + float(d["width"]), y1 + float(d["height"])
        elif len(d) >= 4 and all(isinstance(v, (int, float)) for v in d[:4]):
            x1, y1, x2, y2 = (float(v) for v in d[:4])  # (x1, y1, x2, y2)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _parse_bboxes(bboxes, B):
    # SAM3 Detect's bboxes input: a dict or a list of boxes applies to all frames,
    # a list of lists is per-frame (missing frames get no boxes).
    # Returns one list of (x1, y1, x2, y2) tuples per frame.
    if bboxes is None:
        return [[] for _ in range(B)]
    if isinstance(bboxes, dict):
        items = [bboxes]
    elif isinstance(bboxes, (list, tuple)):
        if not bboxes:
            return [[] for _ in range(B)]
        if len(bboxes) == 4 and all(isinstance(v, (int, float)) for v in bboxes):
            items = [bboxes]  # a single (x1, y1, x2, y2) box
        elif isinstance(bboxes[0], (list, tuple)):
            per_frame = []
            for frame in bboxes:
                boxes = [_bbox_xyxy(d) for d in frame]
                per_frame.append([b for b in boxes if b is not None])
            while len(per_frame) < B:
                per_frame.append([])
            return per_frame[:B]
        else:
            items = list(bboxes)
    else:
        return [[] for _ in range(B)]
    shared = [_bbox_xyxy(d) for d in items]
    shared = [b for b in shared if b is not None]
    return [shared] * B


def _boxes_to_sam3(per_frame_boxes):
    # (x1, y1, x2, y2) tuples per frame -> SAM3 Detect's bboxes format
    return [[{"x": int(b[0]), "y": int(b[1]), "width": int(b[2] - b[0]), "height": int(b[3] - b[1])} for b in frame]
            for frame in per_frame_boxes]


def _frame2np(frame):
    arr = (frame * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _frame2pil(frame):
    return Image.fromarray(_frame2np(frame), mode="RGB")


def _detect_bbox(image, model_name, threshold):
    from ultralytics import YOLO

    path = _resolve_bbox_path(model_name)
    if path is None:
        raise ValueError(f"bbox detector '{model_name}' not found in models/ultralytics/bbox")
    model = YOLO(path)

    B, H, W, _ = image.shape
    device = str(comfy.model_management.get_torch_device())
    masks = []
    boxes_out = []
    pbar = comfy.utils.ProgressBar(B)
    for b in range(B):
        pred = model(_frame2pil(image[b]), conf=threshold, device=device)[0]
        frame_mask = np.zeros((H, W), dtype=np.float32)
        frame_boxes = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            segm = None
            if pred.masks is not None and pred.masks.data is not None:
                segm = pred.masks.data.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1 = max(0, int(xyxy[i][0])), max(0, int(xyxy[i][1]))
                x2, y2 = min(W, int(xyxy[i][2])), min(H, int(xyxy[i][3]))
                if segm is not None:
                    m = F.interpolate(torch.from_numpy(segm[i]).unsqueeze(0).unsqueeze(0),
                                       size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                    frame_mask = np.maximum(frame_mask, (m > 0.5).astype(np.float32))
                else:
                    frame_mask[y1:y2, x1:x2] = 1.0
                frame_boxes.append({
                    "x": x1, "y": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1),
                    "score": float(confs[i]), "label": model.names[int(boxes.cls[i].item())],
                })
        masks.append(torch.from_numpy(frame_mask))
        boxes_out.append(frame_boxes)
        pbar.update(1)
    return torch.stack(masks), boxes_out


def _detect_sam(image, model_name, pos_pts, neg_pts, per_frame_boxes, threshold):
    loader_cls = nodes.NODE_CLASS_MAPPINGS.get("SAMLoader")
    if loader_cls is None:
        raise ValueError("segm mode with a SAM model needs ComfyUI-Impact-Pack installed (SAMLoader)")
    (sam_model,) = loader_cls().load_model(model_name, "AUTO")
    # SAM/ESAM return the raw model with the predictor attached as sam_wrapper;
    # SAM2 returns the wrapper itself
    sam_obj = getattr(sam_model, "sam_wrapper", sam_model)
    if not hasattr(sam_obj, "predict"):
        raise ValueError(f"Invalid SAM model '{model_name}': connect one from 'SAMLoader (Impact)'")
    try:
        sam_obj.prepare_device()
        B, H, W, _ = image.shape
        points = pos_pts + neg_pts
        plabs = [1] * len(pos_pts) + [0] * len(neg_pts)
        if not points and not any(per_frame_boxes):
            raise ValueError("segm mode needs a prompt: bboxes or pos/neg coords (connected or "
                              "carried by the context), or a bbox detector that finds something")

        masks = []
        pbar = comfy.utils.ProgressBar(B)
        for b in range(B):
            arr = _frame2np(image[b])
            boxes = per_frame_boxes[b]
            frame_mask = np.zeros((H, W), dtype=bool)
            if points:
                for m in sam_obj.predict(arr, points, plabs, boxes[0] if boxes else None, threshold):
                    frame_mask |= np.asarray(m, dtype=bool)
            else:
                for box in boxes:
                    for m in sam_obj.predict(arr, None, None, box, threshold):
                        frame_mask |= np.asarray(m, dtype=bool)
            masks.append(torch.from_numpy(frame_mask).float())
            pbar.update(1)
    finally:
        sam_obj.release_device()
    return torch.stack(masks), _boxes_to_sam3(per_frame_boxes)


_last_sam3 = None  # (model_name, (model, clip))


def _load_sam3(model_name):
    # The native checkpoint loader brings the SAM3 model and its own SAM3 CLIP
    global _last_sam3
    if _last_sam3 is None or _last_sam3[0] != model_name:
        path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
        model, clip, _vae, _clipvision = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        if not hasattr(model.model.diffusion_model, "detector"):
            raise ValueError(f"'{model_name}' is not a SAM3 / SAM3.1 checkpoint")
        _last_sam3 = (model_name, (model, clip))
    return _last_sam3[1]


def _detect_sam3(image, model, conditioning, per_frame_boxes, pos_coords, neg_coords, threshold, refine_iterations):
    from comfy_extras.nodes_sam3 import SAM3_Detect

    bboxes = _boxes_to_sam3(per_frame_boxes)
    if conditioning is None and not any(bboxes) and not pos_coords and not neg_coords:
        raise ValueError("sam3 modes need a prompt: sam3_prompt text, bboxes, or pos/neg coords")
    out = SAM3_Detect.execute(
        model=model, image=image, conditioning=conditioning,
        bboxes=bboxes if any(bboxes) else None,
        positive_coords=pos_coords, negative_coords=neg_coords,
        threshold=threshold, refine_iterations=refine_iterations, individual_masks=False,
    )
    masks, boxes = out.result
    return masks, boxes


def _overlay(image, mask, color):
    # KJNodes ImageAndMaskPreview style: blend the mask color over the image
    # where the mask is set, the hex alpha channel acting as the opacity
    if len(color) not in (7, 9) or color[0] != "#":
        raise ValueError("mask_color must be in format #RRGGBB or #RRGGBBAA")
    r, g, b = hex_to_rgb(color[:7])
    alpha = 1.0 if len(color) == 7 else int(color[7:9], 16) / 255.0
    m = (mask * alpha).unsqueeze(-1)
    tint = torch.zeros_like(image)
    tint[..., 0] = r / 255
    tint[..., 1] = g / 255
    tint[..., 2] = b / 255
    return image * (1 - m) + tint * m


def _preview_video(images, fps):
    # The native Create Video -> Save Video path: encode the batch to a
    # temp mp4 and hand the file to the node preview
    video = CreateVideo.execute(images, fps=fps).result[0]
    prefix = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_temp_directory(), images.shape[3], images.shape[2])
    file = f"{filename}_{counter:05}_.mp4"
    video.save_to(os.path.join(full_output_folder, file),
                  format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
    return ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.temp)])


class GibbyDetection(io.ComfyNode):
    """Detect / segment objects on the context image and write the mask back into the context."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        bbox_models = _bbox_detector_models()
        segm_models = _segm_models()
        sam3_models = _sam3_models()
        return io.Schema(
            node_id="Gibby_Detection",
            display_name="Mask/Segment (Context)",
            category="gibby/detection",
            # output node: runs (and refreshes the preview) even with nothing connected
            is_output_node=True,
            search_aliases=["detect", "segment", "bbox", "yolo", "sam", "sam3"],
            description=(
                "Detects or segments objects on the image (connected image wins over the "
                "context one) and stores the resulting image, mask and bboxes in the context; "
                "the context's model and clip pass through untouched. Modes: bbox (ultralytics "
                "detector), segm (SAM model, prompted by the bboxes or the detector's findings), "
                "sam3 (the selected SAM3 checkpoint, text and prompt detection). With preview "
                "on, the node displays the image with the mask tinted in the mask color over it."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip="Base context; its image is used when no image is connected"),
                io.Image.Input("image", optional=True, tooltip="Overrides the context image"),

                io.BoundingBox.Input("bboxes", optional=True, force_input=True, tooltip="Box prompts, same format as SAM3 Detect"),
                io.String.Input("pos_coords", optional=True, force_input=True, tooltip='Positive point prompts as JSON [{"x": int, "y": int}, ...]'),
                io.String.Input("neg_coords", optional=True, force_input=True, tooltip='Negative point prompts as JSON [{"x": int, "y": int}, ...]'),
                io.Combo.Input("mode", default="sam3", options=["bbox", "segm", "sam3"]),
                io.Combo.Input("bbox_detector", default=bbox_models[0], options=bbox_models,
                               tooltip="Ultralytics bbox detector (bbox mode; also the prompt source when segm gets no bboxes)"),
                io.Combo.Input("segm_model", default=segm_models[0], options=segm_models,
                               tooltip="SAM model (segm / sam3+segm modes)"),
                io.Combo.Input("sam3_model", default=sam3_models[0], options=sam3_models,
                               tooltip="SAM3 / SAM3.1 checkpoint (sam3 modes)"),
                io.Float.Input("sam3_threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Detection score threshold (sam3 modes)"),
                io.Int.Input("sam3_refine_iterations", default=2, min=0, max=5,
                              tooltip="SAM decoder refinement passes (sam3 modes)"),
                io.String.Input("sam3_prompt", default="", multiline=True,
                               tooltip="Text prompt, encoded by the checkpoint's SAM3 clip like CLIP Text Encode (sam3 modes)"),
                io.Boolean.Input("preview", default=True, tooltip="Show the mask tinted in the mask color over the image (video) output"),
                io.Float.Input("preview_fps", default=30.0, min=1.0, max=120.0, step=1.0,
                               tooltip="Frame rate of the preview when the input is a batch of frames (video)"),
                io.Color.Input("mask_color", default="#FF00FF80", tooltip="Mask tint color, #RRGGBB or #RRGGBBAA (the alpha is the opacity)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Image.Output("image_masked", tooltip="The image with the mask tinted in the mask color"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, bboxes=None, pos_coords=None, neg_coords=None,
                mode="sam3", bbox_detector="(no bbox models found)", segm_model="(no models found)",
                sam3_model="(no models found)", sam3_threshold=0.5, sam3_refine_iterations=2,
                sam3_prompt="", preview=True, preview_fps=30.0, mask_color="#FF00FF80") -> io.NodeOutput:
        ctx = dict(context) if isinstance(context, dict) else {}
        det_image = image if image is not None else ctx.get("image")
        if det_image is None:
            raise ValueError("No image to detect on: connect an image or a context carrying one")
        # Prompts: connected inputs win, otherwise fall back to what the context
        # carries (a previous Detection node stores its bboxes and mask there)
        if bboxes is None:
            bboxes = ctx.get("bboxes")

        B = det_image.shape[0]
        per_frame_boxes = _parse_bboxes(bboxes, B)

        if mode == "bbox":
            mask, boxes = _detect_bbox(det_image, bbox_detector, _DETECT_THRESHOLD)
        elif mode == "segm":
            pos_pts = _parse_points(pos_coords)
            neg_pts = _parse_points(neg_coords)
            if not any(per_frame_boxes) and not pos_pts and not neg_pts:
                if _resolve_bbox_path(bbox_detector) is not None:
                    _det_mask, detected = _detect_bbox(det_image, bbox_detector, _DETECT_THRESHOLD)
                    per_frame_boxes = _parse_bboxes(detected, B)
            mask, boxes = _detect_sam(det_image, segm_model, pos_pts, neg_pts,
                                      per_frame_boxes, _DETECT_THRESHOLD)
        elif mode == "sam3":
            model, clip = _load_sam3(sam3_model)
            conditioning = None
            if sam3_prompt and sam3_prompt.strip():
                conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(sam3_prompt))
            mask, boxes = _detect_sam3(det_image, model, conditioning, per_frame_boxes,
                                       pos_coords, neg_coords, sam3_threshold, sam3_refine_iterations)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # The context keeps its model and clip (and the plain image) untouched
        ctx["image"] = det_image
        ctx["mask"] = mask
        ctx["bboxes"] = boxes
        masked = _overlay(det_image, mask, mask_color)
        # The preview is display-only; the image output stays plain
        if preview:
            if det_image.shape[0] > 1:
                return io.NodeOutput(ctx, det_image, mask, masked, ui=_preview_video(masked, preview_fps))
            return io.NodeOutput(ctx, det_image, mask, masked, ui=ui.PreviewImage(masked))
        return io.NodeOutput(ctx, det_image, mask, masked)
