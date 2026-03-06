"""
YOLO batch interface: yolo_infer_batch(frames) -> list[DetectionResult].
Uses existing YOLO model and DecodeBox; coordinates in original image pixels.
Preprocessing uses cv2 (resize + letterbox) for speed instead of PIL.
"""
import numpy as np
import torch
import cv2

from .types import DetectionResult


def _frame_to_input_cv2(frame_bgr: np.ndarray, yolo) -> tuple:
    """
    Letterbox + normalize with cv2 (faster than PIL). Returns (1, 3, H, W) and image_shape (h_orig, w_orig).
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    input_h, input_w = yolo.input_shape[0], yolo.input_shape[1]
    if yolo.letterbox_image:
        scale = min(input_w / w_orig, input_h / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        canvas = np.full((input_h, input_w, 3), 128, dtype=np.uint8)
        top, left = (input_h - nh) // 2, (input_w - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        image_data = canvas
    else:
        image_data = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = image_data.astype(np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    image_shape = np.array([h_orig, w_orig], dtype=np.int32)
    return image_data, image_shape

def yolo_infer_batch(
    yolo,
    frames: list,
    conf_thres: float = None,
    nms_thres: float = None,
) -> list:
    """
    Run YOLO on each frame; return list of DetectionResult (boxes in original image coords).
    frames: list of numpy BGR (H,W,3) from OpenCV. Uses cv2 preprocessing and true batch forward.
    """
    conf_thres = conf_thres if conf_thres is not None else yolo.confidence
    nms_thres = nms_thres if nms_thres is not None else yolo.nms_iou
    letterbox = yolo.letterbox_image
    bbox_util = yolo.bbox_util
    num_classes = yolo.num_classes

    results = []
    if not frames:
        return results

    # --- True batch: cv2 preprocess (no PIL), stack, single forward, then batch NMS ---
    image_shapes = []
    batch_list = []
    for frame in frames:
        image_data, image_shape = _frame_to_input_cv2(frame, yolo)
        image_shapes.append(image_shape)
        batch_list.append(image_data)

    batch_data = np.concatenate(batch_list, axis=0)
    image_tensor = torch.from_numpy(batch_data).float()
    if yolo.cuda:
        image_tensor = image_tensor.cuda()
        if next(yolo.net.parameters()).dtype == torch.float16:
            image_tensor = image_tensor.half()

    with torch.no_grad():
        outputs = yolo.net(image_tensor)
        decoded = bbox_util.decode_box(outputs)
        pred_list = bbox_util.non_max_suppression(
            decoded,
            num_classes,
            yolo.input_shape,
            image_shapes,
            letterbox_image=letterbox,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
        )

    for i in range(len(pred_list)):
        if pred_list[i] is None or len(pred_list[i]) == 0:
            results.append(DetectionResult.empty())
            continue
        pred = pred_list[i]
        # DecodeBox.yolo_correct_boxes outputs [top, left, bottom, right] (y1, x1, y2, x2).
        # Realtime pipeline expects [x1, y1, x2, y2].
        boxes_yxyx = pred[:, :4].astype(np.float32)
        boxes_xyxy = boxes_yxyx[:, [1, 0, 3, 2]]
        scores = pred[:, 4].astype(np.float32)
        class_ids = pred[:, 5].astype(np.int32)
        results.append(DetectionResult(boxes_xyxy=boxes_xyxy, class_id=class_ids, score=scores))

    return results
