from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from dgenerate.extras.asdff.utils import bbox_padding

import dgenerate.messages

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Please install ultralytics using `pip install ultralytics`")
    raise


def create_mask_from_bbox(
        bboxes: list[list[float]],
        shape: tuple[int, int],
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        mask_shape: str = "rectangle"
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
            Shape of the mask ("rectangle" or "circle").

    Returns
    -------
        list[Image.Image]
            A list of mask images.
    """
    masks = []
    for bbox in bboxes:
        bbox = bbox_padding(tuple(map(int, bbox)), shape, padding)
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)

        if mask_shape == "rectangle":
            mask_draw.rectangle(bbox, fill=255)
        elif mask_shape == "circle":
            # Compute center and radius
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            radius = min((bbox[2] - bbox[0]) // 2, (bbox[3] - bbox[1]) // 2)
            mask_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
        else:
            raise ValueError(f"Unsupported mask_shape: {mask_shape}")

        masks.append(mask)

    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def yolo_detector(
        image: Image.Image,
        model_path: str | Path | None = None,
        device: str = 'cuda',
        confidence: float = 0.3,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        mask_shape: str = "rectangle"
) -> list[Image.Image] | None:
    if not model_path:
        model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")

    dgenerate.messages.debug_log(f'running adetailer YOLO detector on device: {device}')

    model = None
    try:
        model = YOLO(model_path).to(device)

        pred = model(image, conf=confidence)

        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            return None

        if pred[0].masks is None:
            masks = create_mask_from_bbox(
                bboxes, image.size, padding, mask_shape)
        else:
            masks = mask_to_pil(pred[0].masks.data, image.size)
    finally:
        if model is not None and device != 'cpu':
            model.to('cpu')
            del model
            torch.cuda.empty_cache()

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
