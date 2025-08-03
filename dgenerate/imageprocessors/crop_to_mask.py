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

import PIL.Image

import dgenerate.textprocessing as _textprocessing
import dgenerate.webcache as _webcache
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor
import dgenerate.image as _image


class CropToMaskProcessor(_imageprocessor.ImageProcessor):
    """
    Crop an image to the bounds of a mask downloaded from a URL or loaded from a file.
    
    This processor loads a mask image from a file path or URL and automatically crops the input image
    to the bounding box of the white areas in the mask. The mask should have white pixels 
    for the area of interest and black/dark pixels for areas to ignore.
    
    The "mask" argument specifies the path to a mask file or URL to download the mask from.
    If you do not specify "mask", the processed image is assumed to be a mask.
    
    The "mask-processors" argument allows you to pre-process the "mask" argument with an
    arbitrary image processor chain, for example: invert, gaussian-blur, etc. This
    arguments value must be quoted (single or double string quotes) if you intend
    to supply arguments to the processors in the chain. The pixel alignment of this
    processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.
    
    The "padding" argument can be used to add padding around the detected bounds:

    NOWRAP!
    - A single integer (e.g., "10") applies uniform padding on all sides
    - "WIDTHxHEIGHT" format (e.g., "10x20") applies WIDTH padding horizontally and HEIGHT padding vertically  
    - "LEFTxTOPxRIGHTxBOTTOM" format (e.g., "5x10x5x15") applies specific padding to each side

    Padding values may be negative if desired.
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate 
    resizes the image. This defaults to False, meaning the image is processed after dgenerate 
    is done resizing it.
    """

    NAMES = ['crop-to-mask']

    FILE_ARGS = {
        'mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]}
    }

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the mask-processors will run on, for example: cpu, cuda, cuda:1.'
            ),
            'model-offload': (
                'The "model-offload" argument can be used to enable '
                'cpu model offloading for the mask-processors. If this is disabled, '
                'any torch tensors or modules placed on the GPU will remain there until '
                'the mask-processor is done being used, instead of them being moved back to the CPU '
                'after each invocation. Enabling this may help save VRAM when using multiple mask processors '
                'that make use of the GPU.'
            )
        }
        return help_messages

    def __init__(self,
                 mask: str | None = None,
                 mask_processors: str | None = None,
                 padding: str | int | None = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param mask: Path to mask image file or URL. White pixels indicate areas of interest.
            Or ``None`` indicating that the processed image is the mask.
        :param mask_processors: Pre-process ``mask`` with an arbitrary image processor chain.
        :param padding: Padding to apply around the detected bounds. Can be an integer for uniform 
                        padding, ``WIDTHxHEIGHT`` for horizontal/vertical padding, or
                        ``LEFTxTOPxRIGHTxBOTTOM`` for specific side padding.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._mask_path = mask
        self._mask_processors = mask_processors
        self._pre_resize = pre_resize

        if padding is None:
            padding = 0

        try:
            self._padding = _image.normalize_padding_value(padding)
        except ValueError as e:
            raise self.argument_error(f'Error in "padding" argument: {e}') from e

    def _load_mask(self, target_size: _types.Size = None) -> PIL.Image.Image:
        """
        Load the mask image from file path or URL and apply any mask processors.
        
        :param target_size: Optional size to resize mask to match input image
        :return: Processed mask as PIL Image
        """
        try:
            # Handle URL downloads using webcache
            if _webcache.is_downloadable_url(self._mask_path):
                # Download and cache the URL
                _, mask_file_path = _webcache.create_web_cache_file(
                    self._mask_path,
                    mime_acceptable_desc='image files',
                    mimetype_is_supported=lambda m: m.startswith('image/'),
                    local_files_only=self.local_files_only
                )
                mask_path = mask_file_path
            else:
                # Use local file path directly
                mask_path = self._mask_path

            # Load mask image
            mask_image = PIL.Image.open(mask_path)

            # Apply mask processors if specified
            if self._mask_processors is not None:
                mask_image = self._run_image_processor(
                    self._mask_processors,
                    mask_image,
                    resize_resolution=None,
                    aspect_correct=False,
                    align=1
                )

            # Convert to L for consistency
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')

            # Resize mask to match target size if specified
            if target_size is not None and mask_image.size != target_size:
                mask_image = mask_image.resize(target_size, PIL.Image.Resampling.LANCZOS)

            return mask_image

        except Exception as e:
            raise self.argument_error(f'Failed to load argument "mask" from "{self._mask_path}": {e}')

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

    def _process(self, image: PIL.Image.Image):
        """
        Process the image by cropping it to the mask bounds with padding.
        
        :param image: Input image to process
        :return: Cropped image
        """

        if self._mask_path:
            # Load and process the mask image
            mask_image = self._load_mask(target_size=image.size)
        else:
            mask_image = image
            
        # Find the bounds of white pixels in the mask
        bounds = _image.find_mask_bounds(mask_image, self._padding)
        
        if bounds is None:
            raise self.argument_error('No white pixels found in the mask image')

        # Crop the image
        return image.crop(bounds)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, cropping may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a cropped image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, cropping may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a cropped image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "CropToMaskProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all() 