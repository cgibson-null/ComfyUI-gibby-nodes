"""
Detection node
--------------
Detects / segments objects on the context image (or a directly connected
image, which overrides it) and writes the result back into the context.
The context's model and clip are never touched.

A mode selector (like the Context Loader) reveals the widgets of the chosen
approach, following the Impact FaceDetailer's box and SAM options:
- bbox: the ultralytics bbox detector (models/ultralytics/bbox).
- segm: the bbox detector as a prompt source, refined by a SAM model
  (models/sams).
- sam3: the SAM3 / SAM3.1 checkpoint selected in the node (checkpoints
  folder), loaded with the native checkpoint loader together with its own
  SAM3 CLIP - text detection from the sam3_prompt (encoded like CLIP Text
  Encode) and segmentation from the bboxes / point prompts. With tracking on,
  the per-object masks it detects on the first frame - or a mask connected to
  this node, which skips that detection outright - seed the native
  Run SAM3 Video Track tracker, which propagates them through the whole batch
  (the mask output then follows the objects frame by frame, keeping the
  objects object_indices selects).

bbox/segm run through the Impact Subpack's UltralyticsDetectorProvider and
the Impact Pack's SAM/mask helpers (SAMLoader, make_sam_mask); sam3 runs
through the native SAM3 Detect node.

The prompts (bboxes, pos/neg coords) use the same formats as SAM3 Detect,
and also accept KJNodes BBOX (startX/startY/endX/endY) boxes. A connected
prompt beats the detector; with no prompt connected, the detector runs.
The bboxes are used only by this node (to build the mask) and are not
stored in the context - the mask is.

The mask is stored exactly as the detection produced it: growing, inverting,
dropping isolated specks, filling holes, knocking the background out, and the
tinted preview this node used to show all live in Mask Edit (Context), the
node that chains onto this one.
"""

import json
import os

import numpy as np
import torch
import folder_paths
import nodes
import comfy.sd
import comfy.utils
from comfy_api.latest import io

from ..context import _CONTEXT_TYPE, _CROP_INFO_TYPE, ctx_from, parse_index_spec
from ..crop_paste.crop_image_by_mask import _store_crop_info
from ..resolution_latent import _mask_bbox, _resize_mask

def _impact_core():
    # Imported at call time, not module import: the pack puts its modules/ dir
    # on sys.path only once *it* is loaded, which can be after this node, so an
    # import-time probe would miss an installed-but-not-yet-loaded pack.
    try:
        from impact import core
        return core
    except ImportError:
        raise ValueError("bbox/segm detection needs ComfyUI-Impact-Pack installed")


def _impact_available():
    # Load-order independent: the pack's nodes aren't in the registry yet when
    # this node's schema is built, so check the installed folders directly to
    # decide whether to offer the bbox/segm modes.
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pack = os.path.join(base, "comfyui-impact-pack", "modules")
    subpack = os.path.join(base, "comfyui-impact-subpack", "modules")
    return os.path.isdir(pack) and os.path.isdir(subpack)


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


_last_yolo = None  # (model_name, detector) - one entry, like the SAM caches below


def _bbox_detector(model_name):
    global _last_yolo
    provider = nodes.NODE_CLASS_MAPPINGS.get("UltralyticsDetectorProvider")
    if provider is None:
        raise ValueError("bbox/segm detection needs ComfyUI-Impact-Subpack (UltralyticsDetectorProvider)")
    if _last_yolo is None or _last_yolo[0] != model_name:
        _last_yolo = (model_name, provider().doit(model_name)[0])
    return _last_yolo[1]


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


def _boxes_to_mask(boxes, H, W):
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        mask[y1:y2, x1:x2] = 1.0
    return torch.from_numpy(mask)


def _boxes_to_segs(core, boxes, H, W):
    # (x1, y1, x2, y2) boxes -> an Impact SEG list for make_sam_mask (a filled
    # box stands in for the detector's cropped mask)
    items = []
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        cropped_mask = np.ones((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.float32)
        items.append(core.SEG(None, cropped_mask, 1.0, (x1, y1, x2, y2), (x1, y1, x2, y2), None))
    return (H, W), items


def _sam_predict_points(core, sam_model, frame, pos_pts, neg_pts, threshold, dilation):
    # Manual points with no boxes: drive the SAM predictor directly
    sam_obj = getattr(sam_model, "sam_wrapper", sam_model)
    if not hasattr(sam_obj, "predict"):
        raise ValueError("Invalid SAM model: connect one from 'SAMLoader (Impact)'")
    sam_obj.prepare_device()
    try:
        arr = _frame2np(frame)
        H, W = arr.shape[:2]
        points = pos_pts + neg_pts
        plabs = [1] * len(pos_pts) + [0] * len(neg_pts)
        out = np.zeros((H, W), dtype=bool)
        for m in sam_obj.predict(arr, points, plabs, None, threshold):
            out |= np.asarray(m, dtype=bool)
    finally:
        sam_obj.release_device()
    if dilation:
        out = np.asarray(core.utils.dilate_mask(out.astype(np.uint8), dilation)) > 0
    return torch.from_numpy(out).float()


def _sort_segs_by_size(segs):
    # SEGSOrderedFilterDetailerHook (Impact Pack): segs sorted by size (area) descending
    from impact import hooks
    return hooks.SEGSOrderedFilterDetailerHook("area(=w*h)", True, 0, len(segs[1])).post_detection(segs)


def _detect_bbox(image, model_name, threshold, dilation, drop_size, override_boxes=None):
    B, H, W, _ = image.shape
    if override_boxes is not None and any(override_boxes):
        masks = [_boxes_to_mask(override_boxes[b], H, W) for b in range(B)]
        return torch.stack(masks)
    detector = _bbox_detector(model_name)
    core = _impact_core()
    masks = []
    pbar = comfy.utils.ProgressBar(B)
    for b in range(B):
        segs = _sort_segs_by_size(detector.detect(image[b].unsqueeze(0), threshold, dilation, 1.0, drop_size))
        masks.append(core.segs_to_combined_mask(segs))
        pbar.update(1)
    return torch.stack(masks)


def _detect_segm(image, m, pos_pts, neg_pts, per_frame_boxes):
    core = _impact_core()

    B, H, W, _ = image.shape
    # No manual prompt: run the bbox detector as the SAM prompt source
    if not any(per_frame_boxes) and not pos_pts:
        detector = _bbox_detector(m["bbox_detector"])
        thr, dil, drop = m.get("bbox_threshold", 0.5), m.get("bbox_dilation", 0), m.get("drop_size", 1)
        per_frame_boxes = [[tuple(s.bbox) for s in _sort_segs_by_size(detector.detect(image[b].unsqueeze(0), thr, dil, 1.0, drop))[1]] for b in range(B)]
    # Nothing to prompt SAM with: an empty mask, so a paste by it keeps the image untouched
    if not any(per_frame_boxes) and not pos_pts and not neg_pts:
        return torch.zeros(B, H, W, dtype=torch.float32)
    sam_model = _load_sam(m["segm_model"])

    hint = m.get("sam_detection_hint", "center-1")
    sdil = m.get("sam_dilation", 0)
    sthr = m.get("sam_threshold", 0.93)
    sexp = m.get("sam_bbox_expansion", 0)
    mhint = m.get("sam_mask_hint_threshold", 0.7)
    use_neg = m.get("sam_mask_hint_use_negative", "False")

    masks = []
    pbar = comfy.utils.ProgressBar(B)
    for b in range(B):
        fb = per_frame_boxes[b]
        if fb:
            segs = _boxes_to_segs(core, fb, H, W)
            sam_mask = core.make_sam_mask(sam_model, segs, image[b], hint, sdil, sthr, sexp, mhint, use_neg)
            mask = core.segs_to_combined_mask(core.segs_bitwise_and_mask(segs, sam_mask))
        elif pos_pts or neg_pts:
            mask = _sam_predict_points(core, sam_model, image[b], pos_pts, neg_pts, sthr, sdil)
        else:
            mask = torch.zeros(H, W, dtype=torch.float32)
        masks.append(mask)
        pbar.update(1)
    return torch.stack(masks)


_last_sam = None  # (model_name, sam_model)


def _load_sam(model_name):
    # SAMLoader.load_model has no cache of its own (a graph node runs once per
    # prompt); this node calls it internally, so keep the loaded model like _load_sam3
    global _last_sam
    if _last_sam is None or _last_sam[0] != model_name:
        loader = nodes.NODE_CLASS_MAPPINGS.get("SAMLoader")
        if loader is None:
            raise ValueError("segm mode needs ComfyUI-Impact-Pack (SAMLoader)")
        (sam_model,) = loader().load_model(model_name, "AUTO")
        _last_sam = (model_name, sam_model)
    return _last_sam[1]


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


def _detect_sam3(image, model, conditioning, per_frame_boxes, pos_coords, neg_coords, threshold, refine_iterations, individual_masks=False):
    from comfy_extras.nodes_sam3 import SAM3_Detect

    bboxes = _boxes_to_sam3(per_frame_boxes)
    if conditioning is None and not any(bboxes) and not pos_coords and not neg_coords:
        raise ValueError("sam3 modes need a prompt: sam3_prompt text, bboxes, or pos/neg coords")
    out = SAM3_Detect.execute(
        model=model, image=image, conditioning=conditioning,
        bboxes=bboxes if any(bboxes) else None,
        positive_coords=pos_coords, negative_coords=neg_coords,
        threshold=threshold, refine_iterations=refine_iterations, individual_masks=individual_masks,
    )
    return out.result[0]


def _seed_masks(mask, W, H):
    # The tracker takes one binary mask per object for the first frame, so a
    # merged mask (several objects painted together) seeds a single object.
    if mask.shape[1:] != (H, W):
        mask = _resize_mask(mask, W, H, "nearest-exact")
    return mask.round()


def _track_sam3(image, model, conditioning, initial_mask, detection_threshold, max_objects, detect_interval,
                object_indices=""):
    # Run SAM3 Video Track over the batch and flatten the tracks back to one
    # union mask per frame (SAM3 Track to Mask, fed the resolved indices)
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack, SAM3_TrackToMask

    track = SAM3_VideoTrack.execute(
        images=image, model=model, initial_mask=initial_mask, conditioning=conditioning,
        detection_threshold=detection_threshold, max_objects=max_objects,
        detect_interval=detect_interval,
    ).result[0]
    packed = track.get("packed_masks")
    n_obj = 0 if packed is None else packed.shape[1]
    # Track to Mask reads an empty list as "all objects", so a selection that
    # resolved to nothing gets an index it filters out (n_obj is out of range)
    indices = ",".join(str(i) for i in _resolve_indices(object_indices, n_obj)) or str(n_obj)
    return SAM3_TrackToMask.execute(track, object_indices=indices).result[0]


def _resolve_indices(spec, n):
    # SAM3 Track to Mask's "0,2,3" plus single indices, ranges and slices, in
    # track order; an empty spec is every tracked object.
    picked = parse_index_spec(spec, n, field="object_indices")
    return sorted(picked) if picked is not None else list(range(n))


_SAM_HINTS = ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4",
             "mask-area", "mask-points", "mask-point-bbox", "none"]
_SAM_NEGATIVE = ["False", "Small", "Outter"]


def _bbox_inputs(bbox_models):
    # The FaceDetailer box options, minus the crop factor (it only shapes the
    # per-detection crop region, which a mask-only node never produces)
    return [
        io.Combo.Input("bbox_detector", default=bbox_models[0], options=bbox_models,
                       tooltip="Ultralytics bbox detector (models/ultralytics/bbox)"),
        io.Float.Input("bbox_threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                       tooltip="Detection confidence threshold"),
        io.Int.Input("bbox_dilation", default=10, min=-512, max=512, step=1,
                     tooltip="Grow each detected box (negative to shrink)"),
        io.Int.Input("drop_size", default=10, min=1, step=1,
                     tooltip="Drop detections smaller than this many pixels"),
    ]


def _sam3_track_inputs():
    # The Run SAM3 Video Track widgets; detection_tracking.js hides them again
    # while the tracking toggle is off
    return [
        io.Float.Input("detection_threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                       tooltip="Score threshold for text-prompted detection of new objects"),
        io.Int.Input("max_objects", default=4, min=0, max=64,
                     tooltip="Max tracked objects; the initial masks count toward this limit (0 = the internal cap of 64)"),
        io.Int.Input("detect_interval", default=1, min=1,
                     tooltip="Run detection every N frames (1 = every frame, higher saves compute)"),
        io.String.Input("object_indices", default="",
                        tooltip="Tracked objects to keep: '0,2', the last one '-1', a range '0-2', a slice '1:3', ':-2' or '::2' "
                                "(empty = every tracked object)"),
    ]


def _sam_inputs(segm_models):
    return [
        io.Combo.Input("segm_model", default=segm_models[0], options=segm_models, tooltip="SAM model (models/sams)"),
        io.Combo.Input("sam_detection_hint", default="center-1", options=_SAM_HINTS,
                       tooltip="How SAM is prompted from each box"),
        io.Int.Input("sam_dilation", default=0, min=-512, max=512, step=1, tooltip="Grow the SAM mask (negative to shrink)"),
        io.Float.Input("sam_threshold", default=0.93, min=0.0, max=1.0, step=0.01, tooltip="SAM prompt threshold"),
        io.Int.Input("sam_bbox_expansion", default=0, min=0, max=1000, step=1, tooltip="Expand each box before prompting SAM"),
        io.Float.Input("sam_mask_hint_threshold", default=0.7, min=0.0, max=1.0, step=0.01,
                       tooltip="mask-area hint: fraction of the box used for points"),
        io.Combo.Input("sam_mask_hint_use_negative", default="False", options=_SAM_NEGATIVE,
                       tooltip="mask-area hint: add negative (background) points"),
    ]


class GibbyDetection(io.ComfyNode):
    """Detect / segment objects on the context image and write the mask back into the context."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        sam3_models = _sam3_models()
        sam3_option = io.DynamicCombo.Option("sam3", [
            io.Combo.Input("sam3_model", default=sam3_models[0], options=sam3_models, tooltip="SAM3 / SAM3.1 checkpoint"),
            io.Float.Input("sam3_threshold", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Detection score threshold"),
            io.Int.Input("sam3_refine_iterations", default=2, min=0, max=5, tooltip="SAM decoder refinement passes"),
            io.String.Input("sam3_prompt", default="", multiline=True,
                             tooltip="Text prompt, encoded by the checkpoint's SAM3 clip like CLIP Text Encode"),
            # detection_tracking.js hides the tracker rows again while this is off
            io.Boolean.Input("tracking", default=False,
                             tooltip="Track the first frame's detection through the batch (Run SAM3 Video Track)"),
        ] + _sam3_track_inputs())
        # bbox/segm need the Impact pack + subpack; offer them only when installed.
        # sam3 is first so it is the default mode
        options = [sam3_option]
        if _impact_available():
            bbox_models = _bbox_detector_models()
            segm_models = _segm_models()
            options += [
                io.DynamicCombo.Option("bbox", _bbox_inputs(bbox_models)),
                io.DynamicCombo.Option("segm", _bbox_inputs(bbox_models) + _sam_inputs(segm_models)),
            ]

        return io.Schema(
            node_id="Gibby_Detection",
            display_name="Mask/Segment (Context)",
            category="gibby/detection",
            search_aliases=["detect", "segment", "bbox", "yolo", "sam", "sam3"],
            description=(
                "Detects or segments objects on the image (connected image wins over the "
                "context one) and stores the resulting image and mask in the context; "
                "the context's model and clip pass through untouched. A mode selector reveals "
                "the widgets of the chosen approach: bbox (ultralytics detector), segm (SAM "
                "refined by the detector's boxes), sam3 (the selected SAM3 checkpoint, which can "
                "track its first-frame detection - or a connected mask, one per object - through "
                "a batch of frames). The mask is stored "
                "as the detection produced it - grow it, invert it, clean it up or preview it "
                "with Mask Edit (Context). "
                "Also outputs crop info for pasting results back with Image Paste By Mask "
                "(Batch) (Context): the image, the mask and each mask's bbox as the paste rect."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip="Base context; its image is used when no image is connected"),
                io.Image.Input("image", optional=True, tooltip="Overrides the context image"),

                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Box prompts (beat the detector when connected); same format as SAM3 Detect"),
                io.String.Input("pos_coords", optional=True, force_input=True, tooltip='Positive point prompts as JSON [{"x": int, "y": int}, ...]'),
                io.String.Input("neg_coords", optional=True, force_input=True, tooltip='Negative point prompts as JSON [{"x": int, "y": int}, ...]'),
                # Appended last on purpose: the frontend restores links by input
                # index, so inserting it above would re-route the prompt inputs.
                io.Mask.Input("mask", optional=True,
                              tooltip="Seeds the tracker instead of detecting on the first frame (sam3 with tracking on): "
                                      "one mask per object, from the first frame - a merged mask tracks as a single object. "
                                      "Only a connected mask seeds; the context's mask is left alone"),

                io.DynamicCombo.Input("mode", options=options),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                _CROP_INFO_TYPE.Output("crop_info",
                                       tooltip="For Image Paste By Mask (Batch) (Context): the image, the mask and each mask's bbox as the paste rect"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, bboxes=None, pos_coords=None, neg_coords=None,
                mode=None, mask=None) -> io.NodeOutput:
        ctx = ctx_from(context)
        det_image = image if image is not None else ctx.get("image")
        if det_image is None:
            raise ValueError("No image to detect on: connect an image or a context carrying one")
        if not isinstance(mode, dict):
            raise ValueError("Mask/Segment requires a mode.")
        # A connected bboxes prompt overrides the detector; otherwise it runs.
        B = det_image.shape[0]
        per_frame_boxes = _parse_bboxes(bboxes, B)
        pos_pts = _parse_points(pos_coords)
        neg_pts = _parse_points(neg_coords)

        selected = mode.get("mode")
        if selected == "bbox":
            mask = _detect_bbox(det_image, mode["bbox_detector"],
                                mode.get("bbox_threshold", 0.5), mode.get("bbox_dilation", 0),
                                mode.get("drop_size", 1), per_frame_boxes)
        elif selected == "segm":
            mask = _detect_segm(det_image, mode, pos_pts, neg_pts, per_frame_boxes)
        elif selected == "sam3":
            model, clip = _load_sam3(mode["sam3_model"])
            conditioning = None
            if (mode.get("sam3_prompt") or "").strip():
                conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(mode["sam3_prompt"]))
            thr, refine = mode.get("sam3_threshold", 0.5), mode.get("sam3_refine_iterations", 2)
            if mode.get("tracking"):
                # The tracker is seeded either by the connected mask or by the
                # objects SAM3 detects on the first frame (one mask per object),
                # then propagates them over the batch. A seeded run needs no
                # prompt; a sam3_prompt still lets it pick up new objects.
                if mask is not None:
                    init_mask = _seed_masks(mask, det_image.shape[2], det_image.shape[1])
                else:
                    init_mask = _detect_sam3(det_image[0:1], model, conditioning, per_frame_boxes[:1],
                                             pos_coords, neg_coords, thr, refine, individual_masks=True)
                mask = _track_sam3(det_image, model, conditioning,
                                   init_mask if init_mask.shape[0] else None,
                                   mode.get("detection_threshold", 0.5),
                                   mode.get("max_objects", 4),
                                   mode.get("detect_interval", 1),
                                   mode.get("object_indices", ""))
            else:
                mask = _detect_sam3(det_image, model, conditioning, per_frame_boxes,
                                    pos_coords, neg_coords, thr, refine)
        else:
            raise ValueError(f"Unknown mode: {selected}")

        # Crop info for Image Paste By Mask (Batch) (Context): the image, the mask, and each
        # mask's bbox as the paste rect (the full frame when the mask is empty)
        _, H, W, _ = det_image.shape
        rects = []
        for m in mask:
            bbox = _mask_bbox(m)
            rects.append(bbox if bbox is not None else (0, 0, W, H))
        # The context keeps its model and clip untouched. A connected image
        # replaces the context's, so the latent (and cached sample) built from
        # it are stale.
        info = _store_crop_info(ctx, det_image, mask, rects, drop_stale=image is not None)
        return io.NodeOutput(ctx, det_image, mask, info)
