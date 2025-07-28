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
import numpy as np
import cv2

import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.webcache as _webcache
import patchmatch_cython


class PatchMatchProcessor(_imageprocessor.ImageProcessor):
    """
    Inpaint an image with the PatchMatch algorithm (content aware fill).
    
    The PatchMatch algorithm is used in this processor for pyramidical inpainting
    (filling in missing or masked areas) in images. This processor requires a
    mask image to be provided via the "mask" argument.  The mask should be a grayscale
    image where white pixels (255) indicate areas to inpaint and black pixels
    (0) indicate areas to preserve.
    
    The "mask" argument should point to a file path on disk or a URL that can be downloaded.
    Both local files and remote URLs are supported. The mask will be resized to the
    dimension of the incoming image if they are not the same size.

    The "mask-processors" argument allows you to pre-process the "mask" argument with an
    arbitrary image processor chain, for example: invert, gaussian-blur, etc. This
    arguments value must be quoted (single or double string quotes) if you intend
    to supply arguments to the processors in the chain. The pixel alignment of this
    processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.
    
    The "patch-size" argument specifies the patch size for the PatchMatch algorithm.
    Larger patch sizes can provide better coherence but may be slower.
    
    The "seed" argument allows you to specify a random number generator seed for 
    reproducible results.
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate 
    resizes the image. This defaults to False, meaning the image is processed after 
    dgenerate is done resizing it.
    """

    NAMES = ['patchmatch']

    # hide inherited arguments that are device related since PatchMatch runs on CPU
    HIDE_ARGS = ['device', 'model-offload']

    FILE_ARGS = {
        'mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]}
    }

    def __init__(self,
                 mask: str,
                 mask_processors: str | None = None,
                 patch_size: int = 5,
                 seed: int | None = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param mask: Path to mask image file or URL. White pixels indicate areas to inpaint.
        :param mask_processors: Pre-process ``mask`` with an arbitrary image processor chain.
        :param patch_size: Patch size for PatchMatch algorithm. Default is 5.
        :param seed: Random number generator seed for reproducible results. If None, uses random seed.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if not mask:
            raise self.argument_error('Argument "mask" is required and cannot be empty.')

        if patch_size <= 0:
            raise self.argument_error('Argument "patch-size" must be a positive integer.')

        self._mask_path = mask
        self._mask_processors = mask_processors
        self._patch_size = patch_size
        self._seed = seed
        self._pre_resize = pre_resize

    def _load_mask(self, target_size: _types.Size = None) -> np.ndarray:
        """
        Load the mask image from file path or URL and convert to binary mask.
        
        :param target_size: Optional size to resize mask to match input image
        :return: Binary mask as numpy array where True indicates areas to inpaint
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

            # Load mask image and convert to grayscale
            mask_image = PIL.Image.open(mask_path)

            if self._mask_processors is not None:
                mask_image = self._create_image_processor(
                    self._mask_processors
                ).process(mask_image.convert('RGB'))

            # Convert to grayscale if needed
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')

            # Resize mask to match target size if specified using optimal interpolation
            if target_size is not None:
                mask_array_temp = np.array(mask_image)
                old_size = (mask_array_temp.shape[1], mask_array_temp.shape[0])  # (width, height)
                interpolation = _image.best_cv2_resampling(old_size, target_size)
                mask_array_temp = cv2.resize(mask_array_temp, target_size, interpolation=interpolation)
                mask_image = PIL.Image.fromarray(mask_array_temp)

            return np.array(mask_image) > 128

        except Exception as e:
            raise self.argument_error(f'Failed to load mask from "{self._mask_path}": {e}')

    @staticmethod
    def _create_image_processor(uri_chain_string):
        """ Create an image processor from a URI chain string."""
        import dgenerate.imageprocessors as _imgp
        return _imgp.create_image_processor(
            _textprocessing.shell_parse(
                uri_chain_string,
                expand_home=False,
                expand_glob=False,
                expand_vars=False
            )
        )

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Process the image with PatchMatch inpainting.
        
        :param image: Input PIL image
        :return: Processed PIL image
        """
        # Convert PIL image to numpy array
        image_array = np.array(image)

        # Load mask and ensure it matches the image size
        # Note: image.size is (width, height) but numpy arrays are (height, width, channels)
        mask_array = self._load_mask(target_size=image.size)
        
        # Ensure mask dimensions match the image array
        if mask_array.shape != image_array.shape[:2]:
            # Resize mask to match image array dimensions exactly using optimal interpolation
            mask_uint8 = (mask_array.astype(np.uint8) * 255)
            old_size = (mask_array.shape[1], mask_array.shape[0])  # (width, height)
            new_size = (image_array.shape[1], image_array.shape[0])  # (width, height)
            interpolation = _image.best_cv2_resampling(old_size, new_size)
            mask_resized = cv2.resize(mask_uint8, new_size, interpolation=interpolation)
            mask_array = mask_resized > 128

        # Perform PatchMatch inpainting
        try:
            # Use inpaint_pyramid which auto-selects the fastest available solver
            result_array = patchmatch_cython.inpaint_pyramid(
                image_array,
                mask_array,
                patch_size=self._patch_size,
                seed=self._seed
            )

            # Ensure result array has the same shape as input (fallback for edge cases)
            if result_array.shape != image_array.shape:
                # Use optimal interpolation for resizing back to original size
                old_size = (result_array.shape[1], result_array.shape[0])  # (width, height)
                new_size = image.size  # (width, height)
                interpolation = _image.best_cv2_resampling(old_size, new_size)
                result_array = cv2.resize(result_array, new_size, interpolation=interpolation)

            return PIL.Image.fromarray(result_array)

        except Exception as e:
            raise self.argument_error(f'PatchMatch inpainting failed: {e}')

    def impl_pre_resize(self, image: PIL.Image.Image,
                        resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        """
        Implementation called before resize if pre_resize is True.
        
        :param image: Input image
        :param resize_resolution: Target resolution for resize
        :return: Processed image
        """
        if self._pre_resize:
            return self._process(image)
        else:
            return image

    def impl_post_resize(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Implementation called after resize if pre_resize is False.
        
        :param image: Input image (already resized)
        :return: Processed image
        """
        if not self._pre_resize:
            return self._process(image)
        else:
            return image

    def to(self, device) -> "PatchMatchProcessor":
        """
        PatchMatch runs on CPU, so device changes are ignored.
        
        :param device: Target device (ignored)
        :return: Self
        """
        return self
