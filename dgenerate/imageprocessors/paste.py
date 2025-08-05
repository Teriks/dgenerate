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

import PIL.Image
import PIL.ImageFilter

import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.webcache as _webcache


class PasteProcessor(_imageprocessor.ImageProcessor):
    """
    Paste an image on top of the incoming image at a specified position.
    
    The "image" argument specifies the path to the image file to paste,
    this may be path on disk or a URL link to an image file.

    The "image-processors" argument allows you to pre-process "image" with an
    arbitrary image processor chain. This arguments value must be quoted
    (single or double string quotes) if you intend to supply arguments to
    the processors in the chain. The pixel alignment of this processor
    chain defaults to 1, meaning no forced alignment will occur, you can
    force alignment using the "resize" image processor if desired.
    
    The "position" argument specifies where to paste the image. It can be:

    NOWRAP!
    - "LEFTxTOP" format (e.g., "100x50") to specify the top-left coordinate
    - "LEFTxTOPxRIGHTxBOTTOM" format (e.g., "100x50x300x200") to specify a bounding
      box where the source image will be resized to fit

    The "feather" argument specifies the feathering radius in pixels for softening edges.
    This creates smooth transitions from opaque to transparent. If not specified, no feathering 
    is applied. Cannot be used together with the "mask" parameter, as this auto generates a
    feather mask for you.

    The "feather-shape" argument controls the shape of the feathering:

    NOWRAP!
    - "r" or "rect" or "rectangle" (default): Rectangular feathering from edges
    - "c" or "circle" or "ellipse": Elliptical feathering from center

    Only used when "feather" is specified.
    
    The "mask" argument allows you to specify a mask image path that will be used to control
    the transparency of the pasted image. This may be a path on disk or a URL link to an image file.
    The mask should be a grayscale image where white areas represent full opacity and black areas
    represent full transparency. Cannot be used together with the "feather" parameter. This
    mask will always be resized to the size of the pasted image, which may be the "image"
    argument, or the processed image depending on the value of "reverse".

    The "mask-processors" argument allows you to pre-process the "mask" argument with an
    arbitrary image processor chain. For example: invert, gaussian-blur, etc. This
    cannot be used in "feather" mode on the auto generated feather mask, only on
    user supplied masks. This arguments value must be quoted (single or double string quotes)
    if you intend to supply arguments to the processors in the chain. The pixel alignment
    of this processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.

    The "position-mask" argument allows you to specify a mask image, which will have
    its white area bounds detected to determine the value of "position" for pasting.
    A bounding box will be determined by looking at the image and finding the extents
    of the white pixels in the mask.  If you specify an image, the "position" argument
    will be ignored. This mask will always be resized to the size of the background
    image, which may be the processed image or the "image" argument depending on
    the value of "reverse".

    The "position-mask-padding" argument allows you to specify a padding value for the
    bounding box detection on "position-mask", this allows you to add positive or negative
    padding the detected mask bounding box. This value should be a single integer (uniform),
    or WIDTHxHEIGHT (horizontal and vertical padding), or (LEFTxTOPxRIGHTxBOTTOM)

    The "position-mask-processors" argument allows you to pre-process the "position-mask" argument
    with an arbitrary image processor chain. For example: invert, gaussian-blur, etc. This
    cannot be used in "feather" mode on the auto generated feather mask, only on
    user supplied masks. This arguments value must be quoted (single or double string quotes)
    if you intend to supply arguments to the processors in the chain. The pixel alignment
    of this processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.

    The "reverse" argument allows you to reverse the paste operation, meaning the "image"
    argument is to be considered the background, and the processed image is to be the pasted
    content.
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate 
    resizes the image. This defaults to False, meaning the image is processed after dgenerate 
    is done resizing it.
    """

    NAMES = ['paste']

    OPTION_ARGS = {
        'feather_shape': ['r', 'rect', 'rectangle', 'c', 'circle', 'ellipse'],
    }

    FILE_ARGS = {
        'image': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]},
        'mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]},
        'position_mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]}
    }

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the image-processors/mask-processors will run on, for example: cpu, cuda, cuda:1.'
            ),
            'model-offload': (
                'The "model-offload" argument can be used to enable '
                'cpu model offloading for the image-processors/mask-processors. If this is disabled, '
                'any torch tensors or modules placed on the GPU will remain there until '
                'the image-processor/mask-processor is done being used, instead of them being moved back to the CPU '
                'after each invocation. Enabling this may help save VRAM when using multiple image/mask processors '
                'that make use of the GPU.'
            )
        }
        return help_messages

    def __init__(self,
                 image: str,
                 image_processors: str | None = None,
                 position: str | None = None,
                 feather: int | None = None,
                 feather_shape: str = "rectangle",
                 mask: str | None = None,
                 mask_processors: str | None = None,
                 position_mask: str | None = None,
                 position_mask_padding: str | int | None = None,
                 position_mask_processors: str | None = None,
                 reverse: bool = False,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param image: path to the image file to paste, or paste on to if ``reverse=True``
        :param image_processors: Pre-process ``image`` with an arbitrary image processor chain
        :param position: position specification in "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM" format
        :param feather: feathering radius in pixels for softening edges (cannot be used with mask)
        :param feather_shape: shape of feathering ("rectangle", "rect", "circle", or "ellipse")
        :param mask: path to a mask image file for controlling transparency (cannot be used with feather)
        :param mask_processors: Pre-process ``mask`` with an arbitrary image processor chain, not compatible with ``feather``.
        :param position_mask: path to a mask image file for determining paste position from white area bounds
        :param position_mask_padding: padding value for the position mask bounding box (default "0")
        :param position_mask_processors: Pre-process ``position_mask`` with an arbitrary image processor chain
        :param reverse: Reverse the paste operation?
        :param pre_resize: process the image before it is resized, or after? default is False (after)
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if feather is not None and mask is not None:
            raise self.argument_error(
                'Cannot use both "feather" and "mask" arguments together. '
                'Choose one method for transparency.'
            )

        if mask is None and mask_processors:
            raise self.argument_error(
                'Cannot use "mask-processors" without specifying "mask"'
            )

        if position_mask is None and position_mask_processors:
            raise self.argument_error(
                'Cannot use "position-mask-processors" without specifying "position-mask"'
            )

        if position_mask is None and position_mask_padding is not None:
            raise self.argument_error(
                'Cannot use "position-mask-padding" without specifying "position-mask"'
            )
        
        if position_mask_padding is None:
            position_mask_padding = 0

        if position is None:
            position = "0x0"

        if feather is not None and feather < 0:
            raise self.argument_error(
                'Feather value must be greater than or equal to 0')

        if feather is not None and feather_shape is not None:

            try:
                parsed_shape = _textprocessing.parse_basic_mask_shape(feather_shape)
            except ValueError:
                parsed_shape = None

            if parsed_shape is None or parsed_shape not in {
                _textprocessing.BasicMaskShape.RECTANGLE,
                _textprocessing.BasicMaskShape.ELLIPSE
            }:
                raise self.argument_error(
                    'Feather shape must be: "r", "rect", "rectangle", or "c", "circle", "ellipse"')

        self._feather = feather
        self._feather_shape = feather_shape
        self._pre_resize = pre_resize
        self._reverse = reverse

        # Load source image upfront
        if not _webcache.is_downloadable_url(image) and not os.path.exists(image):
            raise self.argument_error(f'Argument "image" file does not exist: {image}')

        try:
            self._paste_image = self._load_image(image)
            # Ensure source image is in RGB mode
            if self._paste_image.mode != 'RGB':
                self._paste_image = self._paste_image.convert('RGB')
        except Exception as e:
            raise self.argument_error(f'Failed to load argument "image": {e}')

        if image_processors:
            self._paste_image = self._run_image_processor(
                image_processors,
                self._paste_image,
                resize_resolution=None,
                aspect_correct=False,
                align=1
            )

        # Load mask image upfront if provided
        self._mask_image = None
        if mask is not None:
            if not _webcache.is_downloadable_url(mask) and not os.path.exists(mask):
                raise self.argument_error(f'Argument "mask" file does not exist: {mask}')

            try:
                self._mask_image = self._load_image(mask)
            except Exception as e:
                raise self.argument_error(f'Failed to load argument "mask": {e}')

            if mask_processors:
                self._mask_image = self._run_image_processor(
                    mask_processors,
                    self._mask_image,
                    aspect_correct=False,
                    resize_resolution=None,
                    align=1
                )

            # Convert to grayscale if needed
            if self._mask_image.mode != 'L':
                self._mask_image = self._mask_image.convert('L')

        # Load and process position mask image upfront if provided
        if position_mask is not None:
            self._position_mask_path = position_mask
            self._position_mask_processors = position_mask_processors
            self._position_mask_padding = position_mask_padding
        else:
            # Parse position argument normally if no position mask
            self._position_mask_path = None
            self._position_mask_processors = None
            self._position_mask_padding = None
            self._position = self._parse_position(position)


    def _run_image_processor(
            self,
            uri_chain_string,
            image,
            resize_resolution,
            aspect_correct,
            align,
    ):
        """run an image processor from a URI chain string."""
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

    def _load_image(self, image_path: str) -> PIL.Image.Image:
        """Load an image from a file or URL."""

        # Handle URL downloads using webcache
        if _webcache.is_downloadable_url(image_path):
            # Download and cache the URL
            _, image_path = _webcache.create_web_cache_file(
                image_path,
                mime_acceptable_desc='image files',
                mimetype_is_supported=lambda m: m.startswith('image/'),
                local_files_only=self.local_files_only
            )
        
        return PIL.Image.open(image_path)

    def _parse_position(self, position: str) -> tuple:
        """Parse position string into coordinates"""
        try:
            parts = _textprocessing.parse_dimensions(position)
        except ValueError:
            raise self.argument_error(
                f'Invalid position format: {position}. Expected "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM"'
            )

        if len(parts) == 2:
            # LEFTxTOP format
            left, top = parts
            if left < 0 or top < 0:
                raise self.argument_error('Position coordinates must be non-negative')
            return parts

        elif len(parts) == 4:
            # LEFTxTOPxRIGHTxBOTTOM format
            left, top, right, bottom = parts
            if left < 0 or top < 0 or right < 0 or bottom < 0:
                raise self.argument_error('Position coordinates must be non-negative')
            if left >= right or top >= bottom:
                raise self.argument_error(
                    'Invalid bounding box: left must be less than right and top must be less than bottom')
            return parts

        else:
            raise self.argument_error(
                f'Invalid position format: {position}. Expected "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM"')

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Process the image by pasting the source image onto it"""

        # Make a copy of the base image to avoid modifying the original
        background_image = image.copy() if not self._reverse else self._paste_image.copy()

        # Use the preloaded source image
        paste_image = self._paste_image if not self._reverse else image

        if self._position_mask_path:
            # the position mask needs to be resized to the size of the
            # image that will be pasted on to, which may be the image being
            # processed, or the argument "image" depending on "reverse"

            if not _webcache.is_downloadable_url(self._position_mask_path) and not os.path.exists(self._position_mask_path):
                raise self.argument_error(f'Argument "position-mask" file does not exist: {self._position_mask_path}')

            try:
                position_mask_image = self._load_image(self._position_mask_path)
            except Exception as e:
                raise self.argument_error(f'Failed to load argument "position-mask": {e}')

            if self._position_mask_processors:
                position_mask_image = self._run_image_processor(
                    self._position_mask_processors,
                    position_mask_image,
                    resize_resolution=background_image.size,
                    aspect_correct=False,
                    align=1
                )
            else:
                position_mask_image = position_mask_image.resize(background_image.size)

            # Convert to grayscale if needed
            if position_mask_image.mode != 'L':
                position_mask_image = position_mask_image.convert('L')

            # Calculate position from position mask bounds
            try:
                bounds = _image.find_mask_bounds(position_mask_image, self._position_mask_padding)
            except ValueError as e:
                # could not parse bounds
                raise self.argument_error(f'Error in "position-mask-padding" argument: {e}')
            if bounds is None:
                raise self.argument_error('No white pixels found in position mask image.')

            # overrides "position" argument
            self._position = bounds


        if len(self._position) == 2:
            left, top = self._position
            paste_box = (left, top)  # PIL expects (left, upper)
            paste_size = paste_image.size

        elif len(self._position) == 4:
            left, top, right, bottom = self._position
            paste_box = (left, top, right, bottom)  # PIL expects (left, upper, right, lower)
            paste_size = (right - left, bottom - top)

            # Resize source image to fit the bounding box
            resampling = _image.best_pil_resampling(paste_image.size, paste_size)
            paste_image = paste_image.resize(paste_size, resampling)
        else:
            assert False, f"invalid paste point / box with {len(self._position)} dimensions"

        # Handle transparency - either feather or mask, but not both
        if self._feather is not None:
            # Use paste_with_feather for smooth blending
            background_image = _image.paste_with_feather(
                background=background_image,
                foreground=paste_image,
                feather=self._feather,
                shape=self._feather_shape,
                location=paste_box
            )
        elif self._mask_image is not None:
            # Apply custom mask
            mask_resampling = _image.best_pil_resampling(self._mask_image.size, paste_size)
            mask = self._mask_image.resize(paste_size, mask_resampling)
            background_image.paste(paste_image, paste_box, mask)
        else:
            # Simple paste without transparency
            background_image.paste(paste_image, paste_box)

        return background_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Process the image before resizing if pre_resize is True.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: the processed image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Process the image after resizing if pre_resize is False.

        :param image: image to process
        :return: the processed image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "PasteProcessor":
        """
        Does nothing for this processor since it's PIL-based.

        :param device: the device (ignored)
        :return: this processor
        """
        return self
