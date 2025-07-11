from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_pil_image

import dgenerate.memory
import dgenerate.messages
from dgenerate.extras.asdff.utils import bbox_padding
import dgenerate.textprocessing as _textprocessing

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Please install ultralytics using `pip install ultralytics`")
    raise


def create_mask_from_bbox(
        bboxes: list[list[float]],
        shape: tuple[int, int],
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        mask_shape: str = "rectangle",
        index_filter: set[int] | list[int] | None = None
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            List of [x1, y1, x2, y2] bounding boxes.
        shape: tuple[int, int]
            Shape of the image (width, height).
        padding: int | tuple[int, int] | tuple[int, int, int, int], optional
            Padding to apply to the bounding box (default: 0).
        mask_shape: str, optional
            Shape of the mask ("r", "rect", "rectangle" or "c", "circle", "ellipse").
        index_filter: set[int] | list[int] | None
            Include only these detection indices

    Returns
    -------
        list[Image.Image]
            A list of mask images.
    """
    masks = []
    for idx, bbox in enumerate(bboxes):
        if index_filter is not None:
            if idx not in index_filter:
                continue

        bbox = bbox_padding(tuple(map(int, bbox)), shape, padding)
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)

        m_shape = _textprocessing.parse_basic_mask_shape(mask_shape)

        if m_shape == _textprocessing.BasicMaskShape.RECTANGLE:
            mask_draw.rectangle(bbox, fill=255)
        elif m_shape == _textprocessing.BasicMaskShape.ELLIPSE:
            # Compute center and radius
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            radius = min((bbox[2] - bbox[0]) // 2, (bbox[3] - bbox[1]) // 2)
            mask_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
        else:
            raise ValueError(f"Unsupported mask_shape: {mask_shape}")

        masks.append(mask)

    return masks


def mask_to_pil(
        masks: torch.Tensor,
        shape: tuple[int, int],
        index_filter: set[int] | list[int] | None = None) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    index_filter: set[int] | list[int] | None
        Include only these detection indices

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]

    if index_filter is not None:
        return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n) if i in index_filter]
    else:
        return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


@torch.no_grad()
def yolo_detector(
        image: Image.Image,
        model_path: str | Path | None = None,
        device: str = 'cuda',
        confidence: float = 0.3,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        mask_shape: str = "rectangle",
        boxes_only: bool = False,
        class_filter: set[int | str] | list[int | str] | None = None,
        index_filter: set[int] | list[int] | None = None,
        model_masks: bool = False
) -> list[Image.Image] | list[tuple[int, int, int, int]] | None:
    if not model_path:
        model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")

    dgenerate.messages.debug_log(
        f'running adetailer YOLO detector on device: {device}')

    if class_filter is not None and not isinstance(class_filter, set):
        class_filter = set(class_filter)

    model = None
    try:
        model = YOLO(model_path).to(device)

        pred = model(image, conf=confidence)

        original_bboxes = pred[0].boxes.xyxy.cpu().numpy()
        confidences = pred[0].boxes.conf.cpu().numpy()  # Extract confidence scores
        class_ids = pred[0].boxes.cls
        class_names = [model.names[int(c)] for c in class_ids]

        if original_bboxes.size == 0:
            return None

        # Sort boxes: first by x (left to right),
        # then by y (top to bottom),
        # then by confidence (descending)

        # this orders the boxes the same as
        # words on a page (euro languages)
        # deterministically
        sorted_indices = sorted(
            range(len(original_bboxes)), key=lambda i: (original_bboxes[i][0], original_bboxes[i][1], -confidences[i]))

        filtered_bboxes = []
        filtered_indices = []

        for idx in sorted_indices:
            cls_id = int(class_ids[idx])
            cls_name = class_names[idx]

            if class_filter and not ({cls_id, cls_name} & class_filter):
                continue
            filtered_bboxes.append(original_bboxes[idx])
            filtered_indices.append(idx)

        if boxes_only:
            return filtered_bboxes

        if not model_masks or pred[0].masks is None:
            masks = create_mask_from_bbox(
                bboxes=filtered_bboxes,
                shape=image.size,
                padding=padding,
                mask_shape=mask_shape,
                index_filter=index_filter
            )
        else:
            # use masks from the model itself
            masks = mask_to_pil(
                masks=pred[0].masks.data[filtered_indices],
                shape=image.size,
                index_filter=index_filter
            )
    finally:
        if model is not None and device != 'cpu':
            model.to('cpu')
            del model
            dgenerate.memory.torch_gc()

    return masks

# YOLO DETECTION with output mask in square
# def yolo_detector(
#     image: Image.Image, model_path: str | Path | None = None, confidence: float = 0.5
# ) -> list[Image.Image] | None:
#     if not model_path:
#         model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
#     model = YOLO(model_path)
#     pred = model(image, conf=confidence)

#     bboxes = pred[0].boxes.xyxy.cpu().numpy()
#     if bboxes.size == 0:
#         return None

#     square_bboxes = []
#     for bbox in bboxes:
#         x_min, y_min, x_max, y_max = bbox
#         bbox_width = int(x_max - x_min)
#         bbox_height = int(y_max - y_min)
#         max_dimension = max(bbox_width, bbox_height)

#         # Centralize original bbox
#         center_x = int(x_min) + bbox_width // 2
#         center_y = int(y_min) + bbox_height // 2

#         # New square bbox
#         new_x_min = max(center_x - max_dimension // 2, 0)
#         new_y_min = max(center_y - max_dimension // 2, 0)
#         new_x_max = min(new_x_min + max_dimension, image.size[0])
#         new_y_max = min(new_y_min + max_dimension, image.size[1])

#         # Expanding selection
#         exp = 20

#         new_x_min = new_x_min - exp//2
#         new_y_min = new_y_min - exp//2
#         new_x_max = new_x_max + exp//2
#         new_y_max = new_y_max + exp//2

#         square_bboxes.append((new_x_min, new_y_min, new_x_max, new_y_max))
#     print("dim: ",x_max - x_min, y_max - y_min)
#     print("Normalized to square dim and expanded: ",new_x_max - new_x_min, new_y_max - new_y_min,(new_x_min, new_y_min, new_x_max, new_y_max))

#     if pred[0].masks is None:
#         masks = create_mask_from_bbox(square_bboxes, image.size)
#     else:
#         masks = mask_to_pil(pred[0].masks.data, image.size)

#     return masks
