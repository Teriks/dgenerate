# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageStat
import cv2
import huggingface_hub
import numpy
import torch
from torchvision.transforms.functional import to_pil_image as _to_pil_image
from ultralytics import YOLO as _YOLO

import dgenerate.hfhub as _hfhub
import dgenerate.imageprocessors.util as _util
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.webcache as _webcache
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class YOLOProcessor(_imageprocessor.ImageProcessor):
    """
    Process the input image with Ultralytics YOLO object detection.
    
    This processor operates in two distinct modes:
    
    Detection Mode (default, masks=False):

    Returns the original image with bounding boxes or mask outlines drawn around detected objects,
    along with labels showing the detection index, class ID, and class name. The colors of the 
    boxes and text are automatically chosen to contrast with the background for optimal visibility.
    
    Mask Mode (masks=True):

    Returns a single composite mask image containing all detected objects combined together.
    This is useful for inpainting, outpainting, or other mask-based image processing operations.
    
    -----
    
    The "model" argument specifies which YOLO model to use. This can be a path to a local
    model file, a URL to download the model from, or a HuggingFace repository slug / blob link.

    The "weight-name" argument specifies the file name in a HuggingFace repository
    for the model weights, if you have provided a HuggingFace repository slug to the
    model argument.

    The "subfolder" argument specifies the subfolder in a HuggingFace repository
    for the model weights, if you have provided a HuggingFace repository slug to the
    model argument.

    The "revision" argument specifies the revision of a HuggingFace repository
    for the model weights, if you have provided a HuggingFace repository slug to the
    model argument. For example: "main"

    The "token" argument specifies your HuggingFace authentication token explicitly
    if needed for accessing private repositories.

    The "local-files-only" argument specifies that dgenerate should not attempt to
    download any model files, and to only look for them locally in the cache or
    otherwise.

    The "font-size" argument determines the size of the label text. If not specified,
    it will be automatically calculated based on the image dimensions.

    The "line-width" argument controls the thickness of the bounding box lines. If not specified,
    it will be automatically calculated based on the image dimensions.

    The "line-color" argument overrides the color for bounding box lines, mask outlines, 
    and text label backgrounds. This should be specified as a HEX color code, e.g. "#FFFFFF" 
    or "#FFF". If not specified, colors are automatically chosen to contrast with the 
    background. The text color will always be automatically chosen to contrast with the 
    background for optimal readability.
    
    The "class-filter" argument can be used to detect only specific classes. This should be a
    comma-separated list of class IDs or class names, or a single value, for example: "0,2,person,car".
    This filter is applied before "index-filter".

    Example "class-filter" values:

        NOWRAP!
        # Only keep detection class ID 0
        class-filter=0

        NOWRAP!
        # Only keep detection class "hand"
        class-filter=hand

        NOWRAP!
        # keep class ID 2,3
        class-filter=2,3

        NOWRAP!
        # keep class ID 0 & class Name "hand"
        # if entry cannot be parsed as an integer
        # it is interpreted as a name
        class-filter=0,hand

        NOWRAP!
        # "0" is interpreted as a name and not an ID,
        # this is not likely to be useful
        class-filter="0",hand

        NOWRAP!
        # List syntax is supported, you must quote
        # class names
        class-filter=[0, "hand"]

    The "index-filter" argument is a list values or a single value that indicates
    what YOLO detection indices to keep, the index values start at zero. Detections are
    sorted by their top left bounding box coordinate from left to right, top to bottom,
    by (confidence descending). The order of detections in the image is identical to
    the reading order of words on a page (english). Processing will only be
    performed on the specified detection indices, if no indices are specified, then
    processing will be performed on all detections.

    Example "index-filter" values:

        NOWRAP!
        # keep the first, leftmost, topmost detection
        index-filter=0

        NOWRAP!
        # keep detections 1 and 3
        index-filter=[1, 3]

        NOWRAP!
        # CSV syntax is supported (tuple)
        index-filter=1,3

    The "confidence" argument sets the confidence threshold for detections (0.0 to 1.0), defaults to: 0.3

    The "model-masks" argument indicates that masks generated by the model itself should be utilized
    instead of just detection bounding boxes. If this is True, and the model returns mask data (seg models do this),
    mask outlines will be drawn instead of bounding boxes. And in "masks" mode, these masks will be used
    for the composited mask that gets generated. This defaults to False, meaning that bounding boxes
    will be used by default.

    The "masks" argument enables mask generation mode. When True, the processor returns a
    composite mask image instead of the annotated detection image. This defaults to False.

    The "outpaint" argument inverts the generated masks, creating inverted masks suitable
    for outpainting operations. This only has an effect when "masks" is True. This defaults to False.

    The "detector-padding" argument specifies the amount of padding that will be added to the
    detection rectangle for both bounding box drawing and mask generation. The default is 0, you can make
    the bounding box and mask area around the detected feature larger with positive padding
    and smaller with negative padding.

    Padding examples:

        NOWRAP!
        32 (32px Uniform, all sides)

        NOWRAP!
        10x20 (10px Horizontal, 20px Vertical)

        NOWRAP!
        10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)

    The "mask-shape" argument indicates what mask shape should be drawn around a detected feature,
    the default value is "rectangle". You may also specify "circle" to generate an ellipsoid shaped mask.

    Note: When "model-masks" is True and the model returns mask data, the "detector-padding"
    and "mask-shape" arguments will be ignored as the model's own masks are used directly.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['yolo']

    OPTION_ARGS = {
        'mask-shape': ['r', 'rect', 'rectangle', 'c', 'circle', 'ellipse'],
    }

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]},
    }

    @staticmethod
    def _match_hex_color(color):
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        if re.match(pattern, color):
            return True
        else:
            return False

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def __init__(self,
                 model: str,
                 weight_name: str | None = None,
                 subfolder: str | None = None,
                 revision: str | None = None,
                 token: str | None = None,
                 font_size: int | None = None,
                 line_width: int | None = None,
                 line_color: str | None = None,
                 class_filter: int | str | list | tuple | set | None = None,
                 index_filter: int | list | tuple | set | None = None,
                 confidence: float = 0.3,
                 model_masks: bool = False,
                 masks: bool = False,
                 outpaint: bool = False,
                 detector_padding: int | str = _constants.DEFAULT_YOLO_DETECTOR_PADDING,
                 mask_shape: str = _constants.DEFAULT_YOLO_MASK_SHAPE,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: YOLO model to use, can be a local path, a URL, or a HuggingFace repository slug
        :param weight_name: file name in a HuggingFace repository for the model weights,
            if you have provided a HuggingFace repository slug to the model argument
        :param subfolder: subfolder in a HuggingFace repository for the model weights,
            if you have provided a HuggingFace repository slug to the model argument
        :param revision: revision of a HuggingFace repository for the model weights,
            if you have provided a HuggingFace repository slug to the model argument (e.g. "main")
        :param token: HuggingFace authentication token if needed for accessing private repositories
        :param font_size: size of label text, if None will be calculated based on image dimensions
        :param line_width: thickness of bounding box lines, if None will be calculated based on image dimensions
        :param line_color: override color for bounding box lines, mask outlines,
            and text label backgrounds as hex color code (e.g. "#FF0000" or "#F00")
        :param class_filter: list of class IDs or class names to include (e.g. ``[0,2,"person","car"]``)
        :param index_filter: list of detection indices to include (e.g. [0,1,3])
        :param confidence: confidence threshold for detections (0.0 to 1.0)
        :param model_masks: overlay model-generated masks instead of bounding boxes when available, default is ``False``
        :param masks: generate mask images for detected objects, default is ``False``
        :param outpaint: invert generated masks for outpainting, only effective when masks is ``True``, default is ``False``
        :param detector_padding: padding around detection rectangles for both bounding box drawing and mask generation
        :param mask_shape: shape of generated masks ("rectangle" or "circle")
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if confidence < 0.0 or confidence > 1.0:
            raise self.argument_error('Argument "confidence" must be between 0.0 and 1.0.')

        if line_width is not None and line_width < 1:
            raise self.argument_error('Argument "line-width" must be at least 1.')

        if font_size is not None and font_size < 8:
            raise self.argument_error('Argument "font-size" must be at least 8.')

        # Validate color arguments
        if line_color is not None and not self._match_hex_color(line_color):
            raise self.argument_error('Argument "line-color" must be a HEX color code, e.g. #FFFFFF or #FFF')

        if not isinstance(detector_padding, int):
            # Validate and parse padding arguments
            try:
                detector_padding = _textprocessing.parse_dimensions(detector_padding)
                if len(detector_padding) not in {1, 2, 4}:
                    raise ValueError()
            except ValueError:
                raise self.argument_error(
                    'Argument "detector-padding" must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

            if len(detector_padding) == 1:
                detector_padding = detector_padding[0]

        try:
            parsed_shape = _textprocessing.parse_basic_mask_shape(mask_shape)
        except ValueError:
            parsed_shape = None

        if parsed_shape is None or parsed_shape not in {
            _textprocessing.BasicMaskShape.RECTANGLE,
            _textprocessing.BasicMaskShape.ELLIPSE
        }:
            raise self.argument_error(
                'Argument "mask-shape" must be: "r", "rect", "rectangle", or "c", "circle", "ellipse"')

        # HuggingFace parameters
        self._weight_name = weight_name
        self._subfolder = subfolder
        self._revision = revision
        self._token = token

        # Handle model path - support local files, URLs, and HuggingFace repositories
        self._model_path = self._get_model_path(model)

        self._confidence = confidence
        self._line_width = line_width
        self._font_size = font_size
        self._line_color = line_color
        self._model_masks = model_masks
        self._masks = masks
        self._outpaint = outpaint
        self._detector_padding = detector_padding
        self._mask_shape = mask_shape
        self._pre_resize = pre_resize

        # Parse detection filters
        self._class_filter, self._index_filter = _util.yolo_filters_parse(
            class_filter,
            index_filter,
            self.argument_error
        )

        model_size = os.path.getsize(self._model_path)
        self.set_size_estimate(model_size)

        # Load the YOLO model
        try:
            self._model = self.load_object_cached(
                tag=self._model_path,
                estimated_size=self.size_estimate,
                method=lambda: _YOLO(self._model_path)
            )
            self.register_module(self._model.model)
        except Exception as e:
            raise self.argument_error(f'Failed to load YOLO model: {e}') from e

    def _get_model_path(self, model: str) -> str:
        """
        Get the model path, handling local files, URLs, and HuggingFace repositories.
        
        :param model: model specification (local path, URL, or HuggingFace repo slug)
        :return: path to the model file
        """
        try:
            if not _webcache.is_downloadable_url(model):
                _, ext = os.path.splitext(model)
            else:
                ext = ''

            if _hfhub.is_single_file_model_load(model) or ext in {'.pt', '.pth', '.yaml', '.yml'}:
                if os.path.exists(model):
                    return model
                else:
                    # Handle URL downloads
                    return _hfhub.webcache_or_hf_blob_download(model, local_files_only=self.local_files_only)
            else:
                # Handle HuggingFace repository
                return huggingface_hub.hf_hub_download(
                    model,
                    filename=self._weight_name,
                    subfolder=self._subfolder,
                    token=self._token,
                    revision=self._revision,
                    local_files_only=self.local_files_only)
        except Exception as e:
            raise self.argument_error(f'Error loading YOLO model: {e}') from e

    def _apply_padding_to_bbox(self, x1, y1, x2, y2, padding, image_size):
        """
        Apply padding to bounding box coordinates.
        
        :param x1, y1, x2, y2: original bounding box coordinates
        :param padding: padding to apply (int, tuple of 2, or tuple of 4)
        :param image_size: tuple of (width, height) for boundary clipping
        :return: tuple of (x1, y1, x2, y2) with padding applied
        """
        if isinstance(padding, (int, float)):
            # Uniform padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_size[0], x2 + padding)
            y2 = min(image_size[1], y2 + padding)
        elif len(padding) == 2:
            # Horizontal, Vertical padding
            h_pad, v_pad = padding
            x1 = max(0, x1 - h_pad)
            y1 = max(0, y1 - v_pad)
            x2 = min(image_size[0], x2 + h_pad)
            y2 = min(image_size[1], y2 + v_pad)
        elif len(padding) == 4:
            # Left, Top, Right, Bottom padding
            left_pad, top_pad, right_pad, bottom_pad = padding
            x1 = max(0, x1 - left_pad)
            y1 = max(0, y1 - top_pad)
            x2 = min(image_size[0], x2 + right_pad)
            y2 = min(image_size[1], y2 + bottom_pad)

        return x1, y1, x2, y2

    def _create_mask_from_bbox(self, bboxes, shape, padding, mask_shape, index_filter=None):
        """
        Create masks from bounding boxes.
        
        :param bboxes: list of [x1, y1, x2, y2] bounding boxes
        :param shape: tuple of (width, height) for the image
        :param padding: padding to apply to bounding boxes
        :param mask_shape: "rectangle" or "circle"
        :param index_filter: optional set/list of indices to include
        :return: list of PIL Image masks
        """
        masks = []
        for idx, bbox in enumerate(bboxes):
            if index_filter is not None and idx not in index_filter:
                continue

            # Apply padding to bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = self._apply_padding_to_bbox(x1, y1, x2, y2, padding, shape)

            # Create mask
            mask = PIL.Image.new("L", shape, 0)
            mask_draw = PIL.ImageDraw.Draw(mask)

            if mask_shape == "rectangle":
                mask_draw.rectangle([x1, y1, x2, y2], fill=255)
            elif mask_shape == "circle":
                # Compute center and radius
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = min((x2 - x1) // 2, (y2 - y1) // 2)
                mask_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)

            masks.append(mask)

        return masks

    def _get_contrasting_color(self, background_color):
        """
        Calculate the best contrasting color for text based on background color.
        Uses HSV color space to find a high-contrast complementary color.
        
        :param background_color: RGB tuple of the background color
        :return: RGB tuple of the contrasting color
        """
        import colorsys

        # Normalize RGB values to 0-1 range
        r, g, b = [c / 255.0 for c in background_color[:3]]

        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Calculate complementary hue (opposite on color wheel)
        complementary_h = (h + 0.5) % 1.0

        # For high contrast, we want high saturation and appropriate value
        # If background is dark, use bright contrasting color
        # If background is bright, use darker contrasting color
        if v < 0.5:  # Dark background
            contrast_s = min(1.0, s + 0.3)  # Increase saturation
            contrast_v = min(1.0, v + 0.6)  # Increase brightness
        else:  # Bright background  
            contrast_s = min(1.0, s + 0.2)  # Slightly increase saturation
            contrast_v = max(0.2, v - 0.5)  # Decrease brightness

        # Convert back to RGB
        contrast_r, contrast_g, contrast_b = colorsys.hsv_to_rgb(complementary_h, contrast_s, contrast_v)

        # Convert back to 0-255 range and return as integers
        return int(contrast_r * 255), int(contrast_g * 255), int(contrast_b * 255)

    def _sample_bbox_line_area_background_color(self, image, x1, y1, x2, y2, line_width, extra_pixels=3):
        """
        Sample background color from the area where the bounding box lines will be drawn,
        including pixels both inside and outside the line area for better contrast.
        
        :param image: PIL Image to sample from
        :param x1, y1, x2, y2: bounding box coordinates
        :param line_width: width of the line that will be drawn
        :param extra_pixels: additional pixels to sample beyond the line width
        :return: RGB tuple of the average background color around the line area
        """
        image_array = numpy.array(image)
        h, w = image_array.shape[:2]
        
        # Calculate the thickness of the sampling area
        sample_thickness = line_width + extra_pixels * 2
        half_thickness = sample_thickness // 2
        
        # Create lists to collect pixels from all four sides of the box
        pixels = []
        
        # Top edge
        y_start = max(0, y1 - half_thickness)
        y_end = min(h, y1 + half_thickness + 1)
        x_start = max(0, x1 - half_thickness)
        x_end = min(w, x2 + half_thickness + 1)
        if y_end > y_start and x_end > x_start:
            pixels.extend(image_array[y_start:y_end, x_start:x_end].reshape(-1, 3))
        
        # Bottom edge
        y_start = max(0, y2 - half_thickness)
        y_end = min(h, y2 + half_thickness + 1)
        if y_end > y_start and x_end > x_start:
            pixels.extend(image_array[y_start:y_end, x_start:x_end].reshape(-1, 3))
        
        # Left edge (excluding corners to avoid double-counting)
        x_start = max(0, x1 - half_thickness)
        x_end = min(w, x1 + half_thickness + 1)
        y_start = max(0, y1 + half_thickness + 1)
        y_end = min(h, y2 - half_thickness)
        if x_end > x_start and y_end > y_start:
            pixels.extend(image_array[y_start:y_end, x_start:x_end].reshape(-1, 3))
        
        # Right edge (excluding corners to avoid double-counting)
        x_start = max(0, x2 - half_thickness)
        x_end = min(w, x2 + half_thickness + 1)
        if x_end > x_start and y_end > y_start:
            pixels.extend(image_array[y_start:y_end, x_start:x_end].reshape(-1, 3))
        
        if pixels:
            bg_color = numpy.mean(pixels, axis=0)
        else:
            # Fallback to sampling from center of image
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
            bg_sample_area = image.crop((center_x - 25, center_y - 25, center_x + 25, center_y + 25))
            bg_color = PIL.ImageStat.Stat(bg_sample_area).mean
        
        return bg_color

    def _sample_mask_line_area_background_color(self, image, contours, line_width, extra_thickness=3):
        """
        Sample background color from the area where the mask outline will be drawn,
        including pixels both inside and outside the line area for better contrast.
        
        :param image: PIL Image to sample from
        :param contours: list of contours from cv2.findContours
        :param line_width: width of the line that will be drawn
        :param extra_thickness: additional pixels to sample beyond the line width
        :return: RGB tuple of the average background color around the line area
        """
        # Create a mask for the line area
        line_mask = numpy.zeros((image.size[1], image.size[0]), dtype=numpy.uint8)
        
        # Draw the contours with the actual line width plus extra thickness
        # This gives us the area where the line will be plus some surrounding pixels
        sample_width = line_width + extra_thickness * 2
        cv2.drawContours(line_mask, contours, -1, 255, thickness=sample_width)
        
        # Convert image to numpy array
        image_array = numpy.array(image)
        
        # Sample colors from the line area
        line_pixels = image_array[line_mask > 0]
        
        if len(line_pixels) > 0:
            # Calculate mean color from line area pixels
            bg_color = numpy.mean(line_pixels.reshape(-1, 3), axis=0)
        else:
            # Fallback to sampling from center of image if line area is empty
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
            bg_sample_area = image.crop((center_x - 25, center_y - 25, center_x + 25, center_y + 25))
            bg_color = PIL.ImageStat.Stat(bg_sample_area).mean
        
        return bg_color

    def _calculate_line_width_font_size(self, image_size):
        """
        Calculate appropriate line width and font size based on image dimensions.
        
        :param image_size: tuple of (width, height)
        :return: tuple of (line_width, font_size, text_padding)
        """
        # Use the larger dimension to calculate sizes
        max_dim = max(image_size)
        
        # Calculate line width as 0.2% of max dimension, with min of 1
        if self._line_width is None:
            line_width = max(1, int(0.003 * max_dim))
        else:
            line_width = self._line_width
            
        # Calculate font size as 1.5% of max dimension, with min of 10
        if self._font_size is None:
            font_size = max(10, int(0.015 * max_dim))
        else:
            font_size = self._font_size
            
        # Calculate text padding as 0.3% of max dimension, with min of 2
        text_padding = max(2, int(0.003 * max_dim))
            
        return line_width, font_size, text_padding

    @torch.no_grad()
    def _process(self, image):
        # Convert PIL image to numpy array for YOLO
        input_image = numpy.array(image)

        # Calculate dynamic sizes based on image dimensions
        line_width, font_size, text_padding = self._calculate_line_width_font_size(image.size)

        # Run YOLO detection
        results = self._model(input_image, conf=self._confidence)

        # Create a copy of the image to draw on
        output_image = image.copy()
        draw = PIL.ImageDraw.Draw(output_image)

        # Try to load a font, fall back to default if not available
        try:
            font = PIL.ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = PIL.ImageFont.truetype(PIL.ImageFont.load_default().path, font_size)
            except:
                font = PIL.ImageFont.load_default()

        sorted_indices = []
        bboxes = None

        if results and len(results) > 0 and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

            # Sort boxes: first by x (left to right), then by y (top to bottom), then by confidence (descending)
            # This orders the boxes the same as words on a page (euro languages) deterministically
            sorted_indices = sorted(range(len(bboxes)), key=lambda i: (bboxes[i][0], bboxes[i][1], -confidences[i]))

            # Filter by class if class filter is set
            if self._class_filter:
                filtered_indices = []
                for i in sorted_indices:
                    class_id = int(class_ids[i])
                    class_name = results[0].names[class_id]
                    # Include if class ID or class name is in the filter
                    if {class_id, class_name} & self._class_filter:
                        filtered_indices.append(i)
                sorted_indices = filtered_indices

            # Filter by index if index filter is set
            if self._index_filter:
                filtered_indices = []
                for idx, i in enumerate(sorted_indices):
                    if idx in self._index_filter:  # Use idx (position after sorting) not i (original index)
                        filtered_indices.append(i)
                sorted_indices = filtered_indices

            # Filter out very small boxes (likely noise)
            original_count = len(sorted_indices)
            filtered_indices = []
            for i in sorted_indices:
                x1, y1, x2, y2 = bboxes[i].tolist()
                width = x2 - x1
                height = y2 - y1
                if width >= 3 and height >= 3:
                    filtered_indices.append(i)
            sorted_indices = filtered_indices
            filtered_count = original_count - len(sorted_indices)
            if filtered_count > 0:
                _messages.debug_log(f"YOLO detection: Filtered out {filtered_count} tiny boxes (< 3x3 pixels)")
            

            # Check if we should use model masks and they are available
            use_masks = self._model_masks and results[0].masks is not None
            masks = None

            if use_masks:
                # Convert masks to PIL images, applying the same filtering
                mask_data = results[0].masks.data[sorted_indices]  # Apply filtering to mask data
                masks = [_to_pil_image(mask_data[i], mode="L").resize(image.size)
                         for i in range(len(mask_data))]

            # First pass: Draw all bounding boxes and mask outlines
            # Collect text label information for second pass
            text_labels = []

            for idx, i in enumerate(sorted_indices):
                # Get box coordinates
                x1, y1, x2, y2 = bboxes[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Apply detector padding to bounding box coordinates
                x1, y1, x2, y2 = self._apply_padding_to_bbox(x1, y1, x2, y2, self._detector_padding, image.size)

                # Get class information
                class_id = int(class_ids[i])
                class_name = results[0].names[class_id]
                confidence = confidences[i]

                # Sample color for contrast calculation
                contours = None  # Initialize for potential reuse
                if use_masks and idx < len(masks):
                    # For mask outlines, we need to find contours first to sample the line area
                    mask = masks[idx]
                    mask_array = numpy.array(mask)
                    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bg_color = self._sample_mask_line_area_background_color(image, contours, line_width)
                else:
                    # For bounding boxes, sample from where the box lines will be drawn
                    bg_color = self._sample_bbox_line_area_background_color(image, x1, y1, x2, y2, line_width)

                # Determine colors - use overrides if provided, otherwise use contrasting colors
                if self._line_color is not None:
                    line_color = self._hex_to_rgb(self._line_color)
                else:
                    line_color = self._get_contrasting_color(bg_color)

                # Text background uses the same color as lines
                text_bg_color = line_color

                # Text color is always contrasting to the text background for readability
                text_color = self._get_contrasting_color(text_bg_color)

                if use_masks and idx < len(masks):
                    # Draw mask outline instead of bounding box
                    # (contours were already calculated above for color sampling)
                    
                    # Draw mask contours
                    for contour in contours:
                        # Convert contour to the format PIL expects
                        points = []
                        for point in contour:
                            points.extend([int(point[0][0]), int(point[0][1])])

                        if len(points) >= 6:  # Need at least 3 points (6 coordinates) for a polygon
                            draw.polygon(points, outline=line_color, width=line_width)
                else:
                    # Draw the bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=line_color, width=line_width)

                # Store text label information for second pass
                label = f"{idx}: {class_id}-{class_name} ({confidence:.2f})"
                
                # Get proper text bounding box using textbbox
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_offset_y = -bbox[1]  # Baseline offset
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textsize(label, font=font)
                    text_offset_y = 0

                text_labels.append({
                    'label': label,
                    'x': x1,
                    'y': y1,
                    'text_width': text_width,
                    'text_height': text_height,
                    'text_offset_y': text_offset_y,
                    'text_bg_color': text_bg_color,
                    'text_color': text_color
                })

            # Second pass: Draw all text labels on top
            for text_info in text_labels:
                x1, y1 = text_info['x'], text_info['y']
                text_width, text_height = text_info['text_width'], text_info['text_height']
                text_offset_y = text_info['text_offset_y']
                text_bg_color = text_info['text_bg_color']
                text_color = text_info['text_color']
                label = text_info['label']

                # Calculate text background box position and ensure it stays within image bounds
                box_y_top = y1 - text_height - text_padding * 2
                box_y_bottom = y1
                box_x_left = x1
                box_x_right = x1 + text_width + text_padding * 2

                # If box would go above the image, draw it below the top edge of the bbox instead
                if box_y_top < 0:
                    box_y_top = y1
                    box_y_bottom = y1 + text_height + text_padding * 2

                # If box would go off the right edge, shift it left
                if box_x_right > image.size[0]:
                    offset = box_x_right - image.size[0]
                    box_x_left = max(0, box_x_left - offset)
                    box_x_right = image.size[0]

                # Ensure box doesn't go off the left edge
                if box_x_left < 0:
                    box_x_left = 0
                    box_x_right = min(image.size[0], text_width + text_padding * 2)

                # Draw text background box
                draw.rectangle([box_x_left, box_y_top, box_x_right, box_y_bottom], fill=text_bg_color)

                # Draw text centered in the box with proper baseline adjustment
                text_x = box_x_left + text_padding
                text_y = box_y_top + text_padding + text_offset_y
                draw.text((text_x, text_y), label, fill=text_color, font=font)

            if not sorted_indices:
                _messages.debug_log("YOLO detection: No objects matched the filters.")
            elif use_masks and results[0].masks is not None:
                _messages.debug_log(f"YOLO detection: Drew mask outlines for {len(sorted_indices)} detections.")
        else:
            _messages.debug_log("YOLO detection: No objects detected in the image.")

        # If masks mode is enabled, return mask images instead of annotated image
        if self._masks:
            if sorted_indices:
                mask_images = []

                if self._model_masks and results[0].masks is not None:
                    # Use model-generated masks (ignore sizing options as per adetailer behavior)
                    mask_data = results[0].masks.data[sorted_indices]
                    for i in range(len(mask_data)):
                        mask_img = _to_pil_image(mask_data[i], mode="L").resize(image.size)
                        mask_images.append(mask_img)
                else:
                    # Create masks from bounding boxes using our sizing options
                    filtered_bboxes = [bboxes[i] for i in sorted_indices]
                    mask_images = self._create_mask_from_bbox(
                        filtered_bboxes,
                        image.size,
                        self._detector_padding,
                        self._mask_shape,
                        index_filter=set(range(len(filtered_bboxes))) if self._index_filter is None else None
                    )

                # Composite all masks into a single mask image
                composite_mask = PIL.Image.new("L", image.size, 0)

                for mask_img in mask_images:
                    # Use PIL.Image.composite to combine masks (logical OR operation)
                    # Convert to binary masks first
                    mask_array = numpy.array(mask_img)
                    composite_array = numpy.array(composite_mask)

                    # Combine using logical OR (any pixel that's white in either mask becomes white)
                    combined_array = numpy.maximum(mask_array, composite_array)
                    composite_mask = PIL.Image.fromarray(combined_array, mode="L")

                _messages.debug_log(f"YOLO detection: Generated composite mask from {len(mask_images)} detections.")

                if self._outpaint:
                    # Invert the composite mask for outpainting
                    composite_mask = PIL.ImageOps.invert(composite_mask)
                    _messages.debug_log("YOLO detection: Inverted composite mask for outpainting.")
                return composite_mask.convert('RGB')
            else:
                # No detections found - return empty mask
                empty_color = 0 if not self._outpaint else 255
                empty_mask = PIL.Image.new("RGB", image.size, (empty_color, empty_color, empty_color))
                _messages.debug_log("YOLO detection: No objects detected, returning empty mask.")
                return empty_mask

        # Normal detection mode - return annotated image
        return output_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, YOLO detection may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a YOLO detected image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, YOLO detection may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a YOLO detected image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
