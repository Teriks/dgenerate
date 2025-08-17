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
import numpy
import torch
import dgenerate.image as _image
from ultralytics import SAM as _SAM
from ultralytics.models.sam.build import sam_model_map as _sam_model_map_u

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


class USAMProcessor(_imageprocessor.ImageProcessor):

    @staticmethod
    def help(loaded_by_name: str):
        models = ('\n'+' '*12+'* ').join(_sam_model_names)
        # the indentation level of this here string is important
        # to the template, it is at level 8, plus 4 extra (doc indent), star, space
        return \
            f"""
        Process the input image with Ultralytics SAM (Segment Anything Model) using point or bounding box prompts.
        
        This processor operates in two distinct modes:
        
        Preview Mode (default, masks=False):
    
        Returns the original image with generated masks outlined and labeled with prompt indices.
        The colors of the outlines and text are automatically chosen to contrast with the background 
        for optimal visibility.
        
        Mask Mode (masks=True):
    
        Returns a single composite mask image containing all generated masks combined together.
        This is useful for inpainting, outpainting, or other mask-based image processing operations.
        
        -----
        
        The "asset" argument specifies which SAM model asset to use. This should be the name
        of an Ultralytics SAM model asset, loading arbitrary checkpoints is not supported.
        This argument may be one of:
        
        NOWRAP!
            * {models}
        
        You may exclude the `.pt` suffix if desired.
    
        The "local-files-only" argument specifies that dgenerate should not attempt to
        download any model files, and to only look for them locally in the cache or
        otherwise.
    
        The "points" argument specifies point prompts as a list of coordinates. Each point
        can be specified as either:
        
        NOWRAP!
        - Single point: [x,y] or x,y or "x,y" or 50x50 or "50x50"
        - Single point: [x,y,label] or x,y,label or "x,y,label" or 50x50xLabel or "50x50xLabel"
        - Nested list/tuple literal: [[x,y], ...] or [[x,y,label], ...]
        - String format: ["x,y", ...] or ["x,y,label", ...] or "x,y","x,y,label"
        - Token list format: 25x25,50x50xLabel
        
        Where label is 1 for foreground, 0 for background. 
        If no label is provided, it defaults to 1 (foreground).
        
        Note that for string format, comma is interchangeable and mixable with the character "x",
        as the quotes delimit the bounds of the point or box value.
        
        lists / tuple literals may not contain space.
    
    
        NOWRAP!
        Examples:
            points=[100,100]                    # Single point
            points=100,100                      # Single point
            points=100x100                      # Single point
            points=[100,100,1]                  # Single point (label)
            points=100,100,1                    # Single point (label)
            points=100x100x1                    # Single point (label)
            points=[[100,100],[200,200,0]]      # Nested list format
            points=["100,100","200,200,0"]      # String format
            points="100,100","200,200,0"        # String format
            points=["100x100","200x200x0"]      # String format
            points="100x100","200x200x0"        # String format
            points=100x100,200x200x0            # Token format
    
        The "boxes" argument specifies bounding box prompts as a list of coordinates. Each box
        can be specified as either:
    
        NOWRAP!
        - Single box: [x1,y1,x2,y2] or x1,y1,x2,y2 or "x1,y1,x2,y2"
        - Nested list/tuple: [[x1,y1,x2,y2], ...]
        - String format: ["x1,y1,x2,y2", ...]
        - Token list format: 50x50x100x100,200x200x400x400
    
        NOWRAP!
        Examples:
            boxes=[50,50,150,150]                             # Single box
            boxes=50,50,150,150                               # Single box
            boxes=50x50x150x150                               # Single box
            boxes=[[50,50,150,150],[200,200,300,300]]         # Nested list format
            boxes=["50,50,150,150","200,200,300,300"]         # String format
            boxes="50,50,150,150","200,200,300,300"           # String format
            boxes="50x50x150x150","200x200x300x300"           # String format
            boxes=50x50x150x150,200x200x300x300               # Token format
    
        The "boxes-mask" argument specifies a black and white mask image where white areas
        will be automatically converted to bounding box prompts. This is useful for integrating
        with YOLO detection results or other object detection masks. The mask will be resized
        to match the input image dimensions before processing.
    
        The "boxes-mask-processors" argument allows you to pre-process the boxes mask with an
        image processor chain before extracting bounding boxes. This is useful for applying
        filters, transforms, or other modifications to the mask. 
    
        Note: You may use python tuple syntax as well as list syntax, additionally
        something such as: (100,100),(100,100) will be interpreted as a tuple of
        of tuples, and: [100,100],[100,100] a tuple of lists.
    
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

    NAMES = ['u-sam']

    OPTION_ARGS = {
        'asset': list(_sam_model_names),
    }

    FILE_ARGS = {
        'boxes-mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]}
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

    @staticmethod
    def _parse_points(points_input):
        """Parse point coordinates from nested lists/tuples or string format."""
        if not points_input:
            return []

        if isinstance(points_input, str) or any(isinstance(x, int) for x in points_input):
            # singular point
            points_input = [points_input]

        points = []
        for point in points_input:
            if isinstance(point, (list, tuple)):
                # Already parsed nested structure: [x, y] or [x, y, label]
                if len(point) < 2:
                    raise ValueError(f"Point must have at least x,y coordinates: {point}")
                elif len(point) == 2:
                    x, y = map(float, point)
                    points.append([x, y, 1])  # Default to foreground
                elif len(point) == 3:
                    x, y, label = map(float, point)
                    points.append([x, y, int(label)])
                else:
                    raise ValueError(f"Point should have 2 or 3 coordinates: {point}")
            elif isinstance(point, str):
                # String format for backward compatibility: "x,y" or "x,y,label"
                # And: "0x0"
                # And: 0x0,0x0x0
                coords = re.split(r'[x,]', point)
                if len(coords) < 2:
                    raise ValueError(f"Point must have at least x,y coordinates: {point}")
                elif len(coords) == 2:
                    x, y = map(float, coords)
                    points.append([x, y, 1])  # Default to foreground
                elif len(coords) == 3:
                    x, y, label = map(float, coords)
                    points.append([x, y, int(label)])
                else:
                    # Try splitting by comma for multiple points
                    comma_split = point.split(',')
                    if len(comma_split) == 1:
                        # No commas found, this is a malformed single point
                        raise ValueError(f"Invalid point format: {point}")

                    # Multiple points separated by commas
                    for c in comma_split:
                        p = USAMProcessor._parse_points(c)
                        if p:
                            points.extend(p)
                        else:
                            raise ValueError(f'Missing point definition in: "{point}", stray comma?')
            else:
                raise ValueError(f"Point must be a list/tuple or string, got: {type(point).__name__}")
        return points

    @staticmethod
    def _parse_boxes(boxes_input):
        """Parse bounding box coordinates from nested lists/tuples or string format."""
        if not boxes_input:
            return []

        if isinstance(boxes_input, str) or any(isinstance(x, int) for x in boxes_input):
            # singular box
            boxes_input = [boxes_input]

        boxes = []
        for box in boxes_input:
            if isinstance(box, (list, tuple)):
                # Already parsed nested structure: [x1, y1, x2, y2]
                if len(box) != 4:
                    raise ValueError(f"Box must have x1,y1,x2,y2 coordinates: {box}")
                x1, y1, x2, y2 = map(float, box)
                boxes.append([x1, y1, x2, y2])
            elif isinstance(box, str):
                # String format for backward compatibility: "x1,y1,x2,y2"
                # or: 0x0x0x0,0x0x0x0
                coords = re.split(r'[x,]', box)
                if len(coords) > 4:
                    # Try splitting by comma for multiple boxes
                    comma_split = box.split(',')
                    if len(comma_split) == 1:
                        # No commas found, this is a malformed single box
                        raise ValueError(f"Invalid box format - too many coordinates: {box}")

                    # Multiple boxes separated by commas
                    for c in comma_split:
                        b = USAMProcessor._parse_boxes(c)
                        if b:
                            boxes.extend(b)
                        else:
                            raise ValueError(f'Missing box definition in: "{box}", stray comma?')
                elif len(coords) < 4:
                    raise ValueError(f'Box must have x1,y1,x2,y2 coordinates: {box}')
                else:
                    x1, y1, x2, y2 = map(float, coords)
                    boxes.append([x1, y1, x2, y2])
            else:
                raise ValueError(f"Box must be a list/tuple or string, got: {type(box).__name__}")
        return boxes

    def __init__(self,
                 asset: str,
                 points: str | list | tuple | None = None,
                 boxes: str | list | tuple | None = None,
                 boxes_mask: str | None = None,
                 boxes_mask_processors: str | None = None,
                 font_size: int | None = None,
                 line_width: int | None = None,
                 line_color: str | None = None,
                 masks: bool = False,
                 outpaint: bool = False,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param asset: SAM model asset to use, an Ultralytics asset name
        :param points: list of point prompts - can be nested lists [[x,y], [x,y,label]] or strings ["x,y", "x,y,label"]
        :param boxes: list of bounding box prompts - can be nested lists [[x1,y1,x2,y2]] or strings ["x1,y1,x2,y2"]
        :param boxes_mask: path or URL to a black and white mask image where white areas will be converted to bounding boxes
        :param boxes_mask_processors: image processor chain to apply to the boxes mask before extracting bounding boxes
        :param font_size: size of label text, if None will be calculated based on image dimensions
        :param line_width: thickness of mask outline lines, if None will be calculated based on image dimensions
        :param line_color: override color for mask outlines and text label backgrounds as hex color code (e.g. "#FF0000" or "#F00")

        :param masks: generate mask images instead of preview, default is ``False``
        :param outpaint: invert generated masks for outpainting, only effective when masks is ``True``, default is ``False``
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if line_width is not None and line_width < 1:
            raise self.argument_error('Argument "line-width" must be at least 1.')

        if font_size is not None and font_size < 8:
            raise self.argument_error('Argument "font-size" must be at least 8.')

        # Validate color arguments
        if line_color is not None and not self._match_hex_color(line_color):
            raise self.argument_error('line-color must be a HEX color code, e.g. #FFFFFF or #FFF')

        # Validate boxes-mask arguments
        if boxes_mask_processors and not boxes_mask:
            raise self.argument_error(
                'Cannot use "boxes-mask-processors" without specifying "boxes-mask"'
            )

        if not asset.endswith('.pt'):
            asset += '.pt'

        # get model path on disk
        self._model_path = self._get_model_path(asset)

        self._line_width = line_width
        self._font_size = font_size
        self._line_color = line_color
        self._masks = masks
        self._outpaint = outpaint
        self._pre_resize = pre_resize
        self._boxes_mask = boxes_mask
        self._boxes_mask_processors = boxes_mask_processors

        # Parse prompts
        try:
            self._points = self._parse_points(points or [])
            self._boxes = self._parse_boxes(boxes or [])
        except ValueError as e:
            raise self.argument_error(f'Error parsing prompts: {e}') from e

        if not self._points and not self._boxes and not self._boxes_mask:
            raise self.argument_error('At least one point, box, or boxes-mask prompt must be specified.')

        model_size = os.path.getsize(self._model_path)
        self.set_size_estimate(model_size)

        # Load the SAM model
        with _ultralytics_download_patch(self.local_files_only):
            try:
                self._model = self.load_object_cached(
                    tag=self._model_path,
                    estimated_size=self.size_estimate,
                    method=lambda: _SAM(asset),
                )
                self.register_module(self._model.model)
            except Exception as e:
                raise self.argument_error(f'Failed to load SAM model: {e}') from e

    def _get_model_path(self, asset_name: str) -> str:

        if asset_name not in _sam_model_names:
            raise self.argument_error(
                f'Unknown SAM model: {asset_name}, must be one of: '
                f'{_textprocessing.oxford_comma(_sam_model_names, "or")}')

        try:
            _, file = _webcache.create_web_cache_file(
                f'{_sam_assets_url}{asset_name}', local_files_only=self.local_files_only
            )
        except Exception as e:
            raise self.argument_error(f'Error downloading ultralytics asset "model": {e}')

        return file

    def _run_image_processor(
            self,
            uri_chain_string,
            image,
            resize_resolution,
            aspect_correct,
            align,
    ):
        """Run an image processor from a URI chain string."""
        import dgenerate.imageprocessors as _imgp
        
        # Convert image to RGB mode for consistent processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        processor = _imgp.create_image_processor(
            _textprocessing.shell_parse(
                uri_chain_string,
                expand_home=False,
                expand_glob=False,
                expand_vars=False
            ),
            device=self.device,
            model_offload=self.model_offload,
        )
        try:
            return processor.process(
                image,
                resize_resolution=resize_resolution,
                aspect_correct=aspect_correct,
                align=align
            )
        finally:
            processor.to('cpu')

    def _extract_boxes_from_mask(self, target_size: _types.Size) -> list:
        """
        Extract bounding boxes from a black and white mask image.
        
        :param target_size: Size to resize mask to match input image
        :return: List of bounding boxes in format [[x1,y1,x2,y2], ...]
        """
        if not self._boxes_mask:
            return []
            
        try:
            # Handle URL downloads using webcache
            if _webcache.is_downloadable_url(self._boxes_mask):
                # Download and cache the URL
                _, mask_file_path = _webcache.create_web_cache_file(
                    self._boxes_mask,
                    mime_acceptable_desc='image files',
                    mimetype_is_supported=lambda m: m.startswith('image/'),
                    local_files_only=self.local_files_only
                )
                mask_path = mask_file_path
            else:
                # Use local file path directly
                mask_path = self._boxes_mask

            # Load mask image and convert to grayscale
            mask_image = PIL.Image.open(mask_path)

            # Apply processors if specified
            if self._boxes_mask_processors is not None:
                mask_image = self._run_image_processor(
                    self._boxes_mask_processors,
                    mask_image,
                    aspect_correct=False,
                    resize_resolution=None,
                    align=1
                )

            # Convert to grayscale if needed
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')

            # Resize mask to match target image size
            if mask_image.size != target_size:
                old_size = mask_image.size
                mask_image = mask_image.resize(
                    target_size,
                    _image.best_pil_resampling(mask_image.size, target_size)
                )
                _messages.debug_log(f"Boxes mask resized from {old_size} to {target_size}")

            # Convert to numpy array for OpenCV processing
            mask_array = numpy.array(mask_image)
            
            # Threshold to ensure we have a proper binary mask
            # Values > 128 are considered white (areas of interest)
            _, binary_mask = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
            
            # Find contours of white areas
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract bounding boxes from contours
            boxes = []
            for contour in contours:
                # Get bounding rectangle for each contour
                x, y, w, h = cv2.boundingRect(contour)
                # Skip very small contours (likely noise)
                if w < 3 or h < 3:
                    continue
                # Convert to [x1, y1, x2, y2] format
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
            
            _messages.debug_log(f"Extracted {len(boxes)} bounding boxes from boxes-mask")
            return boxes
            
        except Exception as e:
            raise self.argument_error(f'Failed to process argument "boxes-mask" "{self._boxes_mask}": {e}')

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

    def _sample_line_area_background_color(self, image, contours, line_width, extra_thickness=3):
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
        # Convert PIL image to numpy array for SAM
        input_image = numpy.array(image)
        
        # Calculate dynamic sizes based on image dimensions
        line_width, font_size, text_padding = self._calculate_line_width_font_size(image.size)

        # Extract boxes from mask if provided and combine with existing boxes
        extracted_boxes = self._extract_boxes_from_mask(image.size)
        all_boxes = list(self._boxes) + extracted_boxes

        # Prepare prompts for batching
        batch_points = []
        batch_labels = []
        
        # Collect all points for batch processing
        if self._points:
            for point in self._points:
                # point is [x, y, label]
                batch_points.append([point[0], point[1]])
                batch_labels.append(int(point[2]))
        
        # Process based on what prompts we have
        if not self._points and not all_boxes:
            _messages.debug_log("SAM mask: No prompts were specified.")
            # Return empty result based on mode
            if self._masks:
                empty_color = 0 if not self._outpaint else 255
                empty_mask = PIL.Image.new("RGB", image.size, (empty_color, empty_color, empty_color))
                return empty_mask
            else:
                return image.copy()

        # Run SAM with prompts - each call returns a single Results object with multiple masks
        results = []
        try:
            if batch_points and all_boxes:
                # Process points first
                if batch_points:
                    sam_result = self._model(input_image, points=batch_points, labels=batch_labels)[0]
                    if sam_result.masks is not None and len(sam_result.masks) > 0:
                        # Extract each mask individually
                        for i in range(len(sam_result.masks)):
                            results.append((sam_result, 'point', i, i))  # (result, type, prompt_idx, mask_idx)
                
                # Process boxes
                if all_boxes:
                    sam_result = self._model(input_image, bboxes=all_boxes)[0]
                    if sam_result.masks is not None and len(sam_result.masks) > 0:
                        # Extract each mask individually
                        for i in range(len(sam_result.masks)):
                            prompt_idx = len(self._points) + i if self._points else i
                            # Determine box type (original vs mask-extracted)
                            if i < len(self._boxes):
                                box_type = 'box'
                            else:
                                box_type = 'mask-box'
                            results.append((sam_result, box_type, prompt_idx, i))  # (result, type, prompt_idx, mask_idx)
                
            elif batch_points:
                # Only points
                sam_result = self._model(input_image, points=batch_points, labels=batch_labels)[0]
                if sam_result.masks is not None and len(sam_result.masks) > 0:
                    # Extract each mask individually
                    for i in range(len(sam_result.masks)):
                        results.append((sam_result, 'point', i, i))  # (result, type, prompt_idx, mask_idx)
                else:
                    _messages.debug_log(f"SAM mask: No masks generated for point prompts")
                        
            elif all_boxes:
                # Only boxes
                sam_result = self._model(input_image, bboxes=all_boxes)[0]
                if sam_result.masks is not None and len(sam_result.masks) > 0:
                    # Extract each mask individually
                    for i in range(len(sam_result.masks)):
                        # Determine box type (original vs mask-extracted)
                        if i < len(self._boxes):
                            box_type = 'box'
                        else:
                            box_type = 'mask-box'
                        results.append((sam_result, box_type, i, i))  # (result, type, prompt_idx, mask_idx)
                else:
                    _messages.debug_log(f"SAM mask: No masks generated for box prompts")
                        
        except Exception as e:
            _messages.debug_log(f"SAM mask: Error processing prompts: {e}")
            results = []

        if not results:
            _messages.debug_log("SAM mask: No masks were generated from prompts.")
            # Return empty result based on mode
            if self._masks:
                empty_color = 0 if not self._outpaint else 255
                empty_mask = PIL.Image.new("RGB", image.size, (empty_color, empty_color, empty_color))
                return empty_mask
            else:
                return image.copy()

        # If masks mode is enabled, return composite mask
        if self._masks:
            composite_mask = PIL.Image.new("L", image.size, 0)

            for result, prompt_type, prompt_idx, mask_idx in results:
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

            _messages.debug_log(f"SAM mask: Generated composite mask from {len(results)} prompts.")

            if self._outpaint:
                # Invert the composite mask for outpainting
                composite_mask = PIL.ImageOps.invert(composite_mask)
                _messages.debug_log("SAM mask: Inverted composite mask for outpainting.")

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
        for result, prompt_type, prompt_idx, mask_idx in results:
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
                bg_color = self._sample_line_area_background_color(image, contours, line_width)

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

                # Draw label
                label = f"{prompt_idx}: {prompt_type}"
                
                # Get proper text bounding box
                # textbbox returns (left, top, right, bottom) including ascent/descent
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # The text baseline offset (negative value of top coordinate)
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

        _messages.debug_log(f"SAM mask: Drew mask outlines for {len(results)} prompts.")
        return output_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, SAM mask processing may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a SAM mask processed image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, SAM mask processing may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a SAM mask processed image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
