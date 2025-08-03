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

import contextlib
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
from ultralytics import SAM as _SAM
from ultralytics.models.sam.build import sam_model_map as _sam_model_map_u

import dgenerate.hfhub as _hfhub
import dgenerate.imageprocessors.constants as _constants
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.webcache as _webcache
from dgenerate.imageprocessors import imageprocessor as _imageprocessor

# sam_h.pt is not actually an available ultralytics asset for whatever reason.
_sam_model_names = [k for k in _sam_model_map_u.keys() if not k == 'sam_h.pt']

_sam_assets_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


@contextlib.contextmanager
def _ultralytics_download_patch(local_files_only: bool):
    import ultralytics.models.sam.build

    og = ultralytics.models.sam.build.attempt_download_asset

    def attempt_download_asset(file) -> str:
        _, path = _webcache.create_web_cache_file(
            f'{_sam_assets_url}{file}',
            local_files_only=local_files_only)
        return path

    ultralytics.models.sam.build.attempt_download_asset = attempt_download_asset

    try:
        yield
    finally:
        ultralytics.models.sam.build.attempt_download_asset = og


class YOLOSAMProcessor(_imageprocessor.ImageProcessor):

    NAMES = ['yolo-sam']

    @staticmethod
    def help(loaded_by_name: str):
        models = ('\n'+' '*12+'* ').join(_sam_model_names)
        # the indentation level of this here string is important
        # to the template, it is at level 8, plus 4 extra (doc indent), star, space
        return \
            f"""
        Process the input image with YOLO object detection followed by SAM (Segment Anything Model) segmentation.
        
        This processor combines the object detection capabilities of YOLO with the precise segmentation of SAM.
        It first runs YOLO to detect objects and get bounding boxes, then uses those boxes as prompts for SAM
        to generate precise segmentation masks.
        
        This processor operates in two distinct modes:
        
        Preview Mode (default, masks=False):

        Returns the original image with generated masks outlined and labeled. The colors of the outlines and 
        text are automatically chosen to contrast with the background for optimal visibility. Labels show the 
        detection index, class information, and confidence score.
        
        Mask Mode (masks=True):

        Returns a single composite mask image containing all generated masks combined together.
        This is useful for inpainting, outpainting, or other mask-based image processing operations.
        
        -----
        
        The "yolo-model" argument specifies the YOLO model to use for object detection. This can be a local path,
        a URL, or a HuggingFace repository slug / blob link.

        The "yolo-weight-name" argument specifies the file name in a HuggingFace repository
        for the YOLO model weights, if you have provided a HuggingFace repository slug to the
        yolo-model argument.

        The "yolo-subfolder" argument specifies the subfolder in a HuggingFace repository
        for the YOLO model weights, if you have provided a HuggingFace repository slug to the
        yolo-model argument.

        The "yolo-revision" argument specifies the revision of a HuggingFace repository
        for the YOLO model weights, if you have provided a HuggingFace repository slug to the
        yolo-model argument. For example: "main"

        The "yolo-token" argument specifies your HuggingFace authentication token explicitly
        if needed for accessing private repositories.

        The "sam-asset" argument specifies which SAM model asset to use. This should be the name
        of an Ultralytics SAM model asset, loading arbitrary checkpoints is not supported.
        This argument may be one of:
        
        NOWRAP!
            * {models}
        
        You may exclude the `.pt` suffix if desired.

        The "confidence" argument sets the confidence threshold for detections (0.0 to 1.0), defaults to: 0.3

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

        The "detector-padding" argument specifies the amount of padding that will be added to the
        YOLO detection rectangles before they are used as SAM prompts. This can expand the detection
        areas to provide more context for segmentation. The default is 0.

        Padding examples:

            NOWRAP!
            32 (32px Uniform, all sides)

            NOWRAP!
            10x20 (10px Horizontal, 20px Vertical)

            NOWRAP!
            10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)

        The "font-size" argument determines the size of the label text. If not specified,
        it will be automatically calculated based on the image dimensions.

        The "line-width" argument controls the thickness of the mask outline lines. If not specified,
        it will be automatically calculated based on the image dimensions.

        The "line-color" argument overrides the color for mask outlines and text label backgrounds.
        This should be specified as a HEX color code, e.g. "#FFFFFF" or "#FFF". If not specified,
        colors are automatically chosen to contrast with the background. The text color will always
        be automatically chosen to contrast with the background for optimal readability.

        The "masks" argument enables mask generation mode. When True, the processor returns a
        composite mask image instead of the annotated preview image. This defaults to False.

        The "outpaint" argument inverts the generated masks, creating inverted masks suitable
        for outpainting operations. This only has an effect when "masks" is True. This defaults to False.

        The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
        This defaults to False, meaning the image is processed after dgenerate is done resizing it.
        """

    OPTION_ARGS = {
        'sam-asset': list(_sam_model_names),
    }

    FILE_ARGS = {
        'yolo-model': {'mode': 'in', 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]},
    }

    @staticmethod
    def _match_hex_color(color):
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        return bool(re.match(pattern, color))

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def __init__(self,
                 yolo_model: str,
                 yolo_weight_name: str | None = None,
                 yolo_subfolder: str | None = None,
                 yolo_revision: str | None = None,
                 yolo_token: str | None = None,
                 sam_asset: str = "sam_b.pt",
                 font_size: int | None = None,
                 line_width: int | None = None,
                 line_color: str | None = None,
                 class_filter: int | str | list | tuple | set | None = None,
                 index_filter: int | list | tuple | set | None = None,
                 confidence: float = 0.3,
                 masks: bool = False,
                 outpaint: bool = False,
                 detector_padding: int | str = 0,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param yolo_model: YOLO model to use for object detection, can be a local path, a URL, or a HuggingFace repository slug
        :param yolo_weight_name: file name in a HuggingFace repository for the YOLO model weights,
            if you have provided a HuggingFace repository slug to the yolo_model argument
        :param yolo_subfolder: subfolder in a HuggingFace repository for the YOLO model weights,
            if you have provided a HuggingFace repository slug to the yolo_model argument
        :param yolo_revision: revision of a HuggingFace repository for the YOLO model weights,
            if you have provided a HuggingFace repository slug to the yolo_model argument (e.g. "main")
        :param yolo_token: HuggingFace authentication token if needed for accessing private repositories
        :param sam_asset: SAM model asset to use, an Ultralytics asset name
        :param font_size: size of label text, if None will be calculated based on image dimensions
        :param line_width: thickness of mask outline lines, if None will be calculated based on image dimensions
        :param line_color: override color for mask outlines and text label backgrounds as hex color code (e.g. "#FF0000" or "#F00")
        :param class_filter: list of class IDs or class names to include (e.g. ``[0,2,"person","car"]``)
        :param index_filter: list of detection indices to include (e.g. [0,1,3])
        :param confidence: confidence threshold for YOLO detections (0.0 to 1.0), default is 0.3
        :param masks: generate mask images instead of preview, default is ``False``
        :param outpaint: invert generated masks for outpainting, only effective when masks is ``True``, default is ``False``
        :param detector_padding: padding around YOLO detection rectangles, default is 0
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if line_width is not None and line_width < 1:
            raise self.argument_error('Argument "line-width" must be at least 1.')

        if font_size is not None and font_size < 8:
            raise self.argument_error('Argument "font-size" must be at least 8.')

        if confidence < 0.0 or confidence > 1.0:
            raise self.argument_error('Argument "confidence" must be between 0.0 and 1.0.')

        # Validate color arguments
        if line_color is not None and not self._match_hex_color(line_color):
            raise self.argument_error('Argument "line-color" must be a HEX color code, e.g. #FFFFFF or #FFF')

        # Validate and parse detector padding
        if not isinstance(detector_padding, int):
            try:
                detector_padding = _textprocessing.parse_dimensions(detector_padding)
                if len(detector_padding) not in {1, 2, 4}:
                    raise ValueError()
            except ValueError:
                raise self.argument_error(
                    'Argument "detector-padding" must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

            if len(detector_padding) == 1:
                detector_padding = detector_padding[0]

        # Store YOLO model parameters
        self._yolo_model = yolo_model
        self._weight_name = yolo_weight_name
        self._subfolder = yolo_subfolder
        self._revision = yolo_revision
        self._token = yolo_token
        self._confidence = confidence
        self._detector_padding = detector_padding

        # Process class filter
        self._class_filter = None
        if class_filter is not None:
            if isinstance(class_filter, (int, str)):
                self._class_filter = {class_filter}
            else:
                self._class_filter = set(class_filter)

        # Process index filter
        self._index_filter = None
        if index_filter is not None:
            if isinstance(index_filter, int):
                self._index_filter = {index_filter}
            else:
                self._index_filter = set(index_filter)

        # Store SAM model parameters
        if not sam_asset.endswith('.pt'):
            sam_asset += '.pt'

        self._sam_asset = sam_asset
        self._line_width = line_width
        self._font_size = font_size
        self._line_color = line_color
        self._masks = masks
        self._outpaint = outpaint
        self._pre_resize = pre_resize

        # Load YOLO model
        try:
            self._yolo_model_path = self._get_yolo_model_path(yolo_model)
            yolo_model_size = os.path.getsize(self._yolo_model_path)
            
            self._yolo_model_obj = self.load_object_cached(
                tag=self._yolo_model_path,
                estimated_size=yolo_model_size,
                method=lambda: _YOLO(self._yolo_model_path)
            )
            self.register_module(self._yolo_model_obj.model)
        except Exception as e:
            raise self.argument_error(f'Failed to load YOLO model: {e}') from e

        # Load SAM model
        try:
            self._sam_model_path = self._get_sam_model_path(sam_asset)
            sam_model_size = os.path.getsize(self._sam_model_path)
            
            with _ultralytics_download_patch(self.local_files_only):
                self._sam_model_obj = self.load_object_cached(
                    tag=self._sam_model_path,
                    estimated_size=sam_model_size,
                    method=lambda: _SAM(sam_asset),
                )
                self.register_module(self._sam_model_obj.model)
                
            # Update total size estimate
            self.set_size_estimate(yolo_model_size + sam_model_size)
        except Exception as e:
            raise self.argument_error(f'Failed to load SAM model: {e}') from e

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

    def _get_yolo_model_path(self, model: str) -> str:
        """
        Get the YOLO model path, handling local files, URLs, and HuggingFace repositories.
        
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

    def _get_sam_model_path(self, asset_name: str) -> str:
        """
        Get the SAM model path from Ultralytics assets.
        
        :param asset_name: SAM asset name
        :return: path to the model file
        """
        if asset_name not in _sam_model_names:
            raise self.argument_error(
                f'Unknown SAM model: {asset_name}, must be one of: '
                f'{_textprocessing.oxford_comma(_sam_model_names, "or")}')

        try:
            _, file = _webcache.create_web_cache_file(
                f'{_sam_assets_url}{asset_name}', local_files_only=self.local_files_only
            )
        except Exception as e:
            raise self.argument_error(f'Error downloading ultralytics asset "sam-asset": {e}')

        return file

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

    def _sample_mask_line_area_background_color(self, image, contours, line_width, extra_thickness=3):
        """
        Sample background color from the area where the outline will be drawn,
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
        
        # Calculate line width as 0.3% of max dimension, with min of 1
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
        # Convert PIL image to numpy array for YOLO/SAM
        input_image = numpy.array(image)
        
        # Calculate dynamic sizes based on image dimensions
        line_width, font_size, text_padding = self._calculate_line_width_font_size(image.size)

        # Run YOLO detection
        yolo_results = self._yolo_model_obj(input_image, conf=self._confidence)

        sorted_indices = []
        bboxes = None
        confidences = None
        class_ids = None
        class_names = []

        if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes

            bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

            # Sort boxes: first by x (left to right), then by y (top to bottom), then by confidence (descending)
            sorted_indices = sorted(range(len(bboxes)), key=lambda i: (bboxes[i][0], bboxes[i][1], -confidences[i]))

            # Filter by class if class filter is set
            if self._class_filter:
                filtered_indices = []
                for i in sorted_indices:
                    class_id = int(class_ids[i])
                    class_name = yolo_results[0].names[class_id]
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
                _messages.debug_log(f"YOLO-SAM: Filtered out {filtered_count} tiny boxes (< 3x3 pixels)")
            
            # Store class names for filtered detections
            for i in sorted_indices:
                class_id = int(class_ids[i])
                class_name = yolo_results[0].names[class_id]
                class_names.append(class_name)

        # If no detections found, return appropriate empty result
        if not sorted_indices:
            _messages.debug_log("YOLO-SAM: No objects detected in the image.")
            if self._masks:
                empty_color = 0 if not self._outpaint else 255
                empty_mask = PIL.Image.new("RGB", image.size, (empty_color, empty_color, empty_color))
                return empty_mask
            else:
                return image.copy()

        # Convert YOLO boxes to SAM box prompts
        sam_boxes = []
        for i in sorted_indices:
            x1, y1, x2, y2 = bboxes[i].tolist()
            # Apply detector padding
            x1, y1, x2, y2 = self._apply_padding_to_bbox(x1, y1, x2, y2, self._detector_padding, image.size)
            sam_boxes.append([x1, y1, x2, y2])

        # Run SAM with box prompts
        sam_results = []
        try:
            sam_result = self._sam_model_obj(input_image, bboxes=sam_boxes)[0]
            if sam_result.masks is not None and len(sam_result.masks) > 0:
                # Extract each mask individually
                for i in range(len(sam_result.masks)):
                    sam_results.append((sam_result, 'yolo-sam', i, i))  # (result, type, prompt_idx, mask_idx)
            else:
                _messages.debug_log(f"YOLO-SAM: No masks generated from {len(sam_boxes)} box prompts")
        except Exception as e:
            _messages.debug_log(f"YOLO-SAM: Error processing SAM prompts: {e}")
            sam_results = []

        if not sam_results:
            _messages.debug_log("YOLO-SAM: No masks were generated from detections.")
            if self._masks:
                empty_color = 0 if not self._outpaint else 255
                empty_mask = PIL.Image.new("RGB", image.size, (empty_color, empty_color, empty_color))
                return empty_mask
            else:
                return image.copy()

        # If masks mode is enabled, return composite mask
        if self._masks:
            composite_mask = PIL.Image.new("L", image.size, 0)

            for result, prompt_type, prompt_idx, mask_idx in sam_results:
                if result.masks is not None and mask_idx < len(result.masks.data):
                    # Get the specific mask data
                    mask_data = result.masks.data[mask_idx]

                    # Convert to PIL Image
                    mask_np = mask_data.cpu().numpy()
                    mask_img = PIL.Image.fromarray((mask_np * 255).astype(numpy.uint8), mode="L")

                    # Resize to match original image size
                    mask_img = mask_img.resize(image.size, PIL.Image.LANCZOS)

                    # Combine with composite mask (logical OR)
                    composite_array = numpy.array(composite_mask)
                    mask_array = numpy.array(mask_img)
                    combined_array = numpy.maximum(composite_array, mask_array)
                    composite_mask = PIL.Image.fromarray(combined_array, mode="L")

            _messages.debug_log(f"YOLO-SAM: Generated composite mask from {len(sam_results)} detections.")

            if self._outpaint:
                # Invert the composite mask for outpainting
                composite_mask = PIL.ImageOps.invert(composite_mask)
                _messages.debug_log("YOLO-SAM: Inverted composite mask for outpainting.")

            return composite_mask.convert('RGB')

        # Preview mode - return annotated image
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

        # Draw mask outlines and labels
        for result, prompt_type, prompt_idx, mask_idx in sam_results:
            if result.masks is not None and mask_idx < len(result.masks.data):
                # Get the specific mask data
                mask_data = result.masks.data[mask_idx]

                # Convert to PIL Image
                mask_np = mask_data.cpu().numpy()
                mask_img = PIL.Image.fromarray((mask_np * 255).astype(numpy.uint8), mode="L")

                # Resize to match original image size
                mask_img = mask_img.resize(image.size, PIL.Image.LANCZOS)
                mask_array = numpy.array(mask_img)

                # Find mask contours first
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Sample background color from the area where the line will be drawn
                bg_color = self._sample_mask_line_area_background_color(image, contours, line_width)

                # Determine colors
                if self._line_color is not None:
                    line_color = self._hex_to_rgb(self._line_color)
                else:
                    line_color = self._get_contrasting_color(bg_color)

                text_bg_color = line_color
                text_color = self._get_contrasting_color(text_bg_color)

                # Draw mask contours
                for contour in contours:
                    # Convert contour to the format PIL expects
                    points = []
                    for point in contour:
                        points.extend([int(point[0][0]), int(point[0][1])])

                    if len(points) >= 6:  # Need at least 3 points (6 coordinates) for a polygon
                        draw.polygon(points, outline=line_color, width=line_width)

                # Create label with detection info
                if prompt_idx < len(sorted_indices):
                    yolo_idx = sorted_indices[prompt_idx]
                    confidence = confidences[yolo_idx]
                    class_name = class_names[prompt_idx]
                    label = f"{prompt_idx}: {class_name} ({confidence:.2f})"
                else:
                    label = f"{prompt_idx}: detection"
                
                # Get proper text bounding box
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_offset_y = -bbox[1]

                # Find a good position for the label (top-left of the mask)
                mask_coords = numpy.where(mask_array > 128)
                if len(mask_coords[0]) > 0:
                    min_y = numpy.min(mask_coords[0])
                    min_x = numpy.min(mask_coords[1])

                    # Calculate text background box position
                    box_x = min_x
                    box_y = max(0, min_y - text_height - text_padding * 2)

                    # If box would go above the image, place it below
                    if box_y < 0:
                        box_y = min_y + text_padding

                    # Ensure box doesn't go off the right edge
                    if box_x + text_width + text_padding * 2 > image.size[0]:
                        box_x = max(0, image.size[0] - text_width - text_padding * 2)

                    # Draw text background box
                    box_right = box_x + text_width + text_padding * 2
                    box_bottom = box_y + text_height + text_padding * 2
                    draw.rectangle([box_x, box_y, box_right, box_bottom], fill=text_bg_color)

                    # Draw text centered in the box with proper baseline adjustment
                    text_x = box_x + text_padding
                    text_y = box_y + text_padding + text_offset_y
                    draw.text((text_x, text_y), label, fill=text_color, font=font)

        _messages.debug_log(f"YOLO-SAM: Drew mask outlines for {len(sam_results)} detections.")
        return output_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, YOLO-SAM processing may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a YOLO-SAM processed image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, YOLO-SAM processing may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a YOLO-SAM processed image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()