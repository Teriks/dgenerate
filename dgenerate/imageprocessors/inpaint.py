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
import cv2
import numpy as np
import spandrel
import torch

import dgenerate.hfhub as _hfhub
import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.webcache as _webcache


def _inpaint_pil(image: PIL.Image.Image, mask: PIL.Image.Image, model) -> PIL.Image.Image:
    """
    Inpaint a PIL image using an inpainting model supported by Spandrel.
    """
    with torch.no_grad():
        # Convert PIL to numpy, RGB->BGR for chaiNNer compatibility
        img_array = np.array(image)[:, :, ::-1].copy()
        mask_array = np.array(mask.convert('L'))

        # Convert to tensors using chaiNNer's approach (simplified)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.flip(0).unsqueeze(0)  # BGR->RGB + batch

        mask_tensor = torch.from_numpy(mask_array).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add dims

        # Move to model device/dtype, binarize mask
        img_tensor = img_tensor.to(model.device, model.dtype)
        mask_tensor = mask_tensor.to(model.device, model.dtype)
        mask_tensor = (mask_tensor > 0.5).float()

        # Inference
        result = model(img_tensor, mask_tensor)

        # Convert back to PIL
        result = result.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        result = np.clip(result * 255, 0, 255).round().astype(np.uint8)
        return PIL.Image.fromarray(result)


class InpaintProcessor(_imageprocessor.ImageProcessor):
    """
    Inpaint an image using inpainting model architectures supported by Spandrel (AI-based inpainting).
    
    This processor uses inpainting models such as LaMa, MAT, and other architectures supported
    by Spandrel for advanced AI-based inpainting to fill in missing or masked areas in images.

    This processor requires either a mask or subject image to be provided via the "mask" or 
    "image" arguments. These arguments are mutually exclusive.
    
    When using the "mask" argument, the incoming image is considered the subject
    image to be inpainted, and the "mask" argument provides a grayscale mask where
    white pixels (255) indicate areas to inpaint and black pixels (0) indicate
    areas to preserve.
    
    When using the "image" argument, the incoming image is considered the mask,
    and the "image" argument provides the subject image to be inpainted. The
    incoming mask image should be a grayscale image where white pixels (255)
    indicate areas to inpaint and black pixels (0) indicate areas to preserve.
    
    The "mask" or "image" argument should point to a file path on disk or a URL
    that can be downloaded. Both local files and remote URLs are supported. The
    mask or image will be resized to match the dimensions of the corresponding
    target image if they are not the same size.

    The "mask-processors" argument allows you to pre-process the "mask" argument with an
    arbitrary image processor chain, for example: invert, gaussian-blur, etc. This
    arguments value must be quoted (single or double string quotes) if you intend
    to supply arguments to the processors in the chain. The pixel alignment of this
    processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.
    
    The "image-processors" argument allows you to pre-process the "image" argument with an
    arbitrary image processor chain, for example: invert, gaussian-blur, etc. This
    arguments value must be quoted (single or double string quotes) if you intend
    to supply arguments to the processors in the chain. The pixel alignment of this
    processor chain defaults to 1, meaning no forced alignment will occur, you
    can force alignment using the "resize" image processor if desired.
    
    The "model" argument specifies the path to an inpainting model file supported by Spandrel,
    such as LaMa, MAT, or other compatible architectures. This can be a local file path or a URL 
    that can be downloaded.
    
    The "dtype" argument can be used to specify the datatype to use for the model in memory,
    it can be either "float32" or "float16". Using "float16" will result in a smaller memory
    footprint if supported by the model. If a model doesn't support FP16, it will automatically
    fall back to FP32 with a warning message.
    
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate 
    resizes the image. This defaults to False, meaning the image is processed after 
    dgenerate is done resizing it.
    """

    NAMES = ['inpaint']

    OPTION_ARGS = {
        'dtype': ['float32', 'float16']
    }

    FILE_ARGS = {
        'mask': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]},
        'image': {'mode': 'in', 'filetypes': [('Images', _imageprocessor.ImageProcessor.image_in_filetypes())]},
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.pt', '*.pth', '*.ckpt', '*.safetensors'])]}
    }

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the inpainting model and mask-processors/image-processors will run on, for example: cpu, cuda, cuda:1.'
            ),
            'model-offload': (
                'The "model-offload" argument can be used to enable '
                'cpu model offloading for the inpainting model and mask-processors/image-processors. If this is disabled, '
                'any torch tensors or modules placed on the GPU will remain there until '
                'the processor is done being used, instead of them being moved back to the CPU '
                'after each invocation. Enabling this may help save VRAM when using multiple processors '
                'that make use of the GPU.'
            )
        }
        return help_messages

    def __init__(self,
                 model: str,
                 mask: str | None = None,
                 mask_processors: str | None = None,
                 image: str | None = None,
                 image_processors: str | None = None,
                 dtype: str = 'float32',
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: Path to inpainting model file (LaMa, MAT, etc.) supported by Spandrel, or URL.
        :param mask: Path to mask image file or URL. White pixels indicate areas to inpaint.
        :param mask_processors: Pre-process ``mask`` with an arbitrary image processor chain.
        :param image: Path to subject image file or URL when incoming image is the mask.
        :param image_processors: Pre-process ``image`` with an arbitrary image processor chain.
        :param dtype: The datatype to use for the model in memory, either "float32" or "float16".
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        # Validate mutually exclusive arguments
        if mask and image:
            raise self.argument_error(
                'Arguments "mask" and "image" are mutually exclusive. '
                'Use "mask" when the incoming image is the subject to be inpainted, '
                'or "image" when the incoming image is the mask.'
            )

        if not mask and not image:
            raise self.argument_error(
                'Either "mask" or "image" argument is required. '
                'Use "mask" when the incoming image is the subject to be inpainted, '
                'or "image" when the incoming image is the mask.'
            )

        if mask_processors and not mask:
            raise self.argument_error(
                'Cannot use "mask-processors" without specifying "mask"'
            )

        if image_processors and not image:
            raise self.argument_error(
                'Cannot use "image-processors" without specifying "image"'
            )

        # Validate dtype argument
        dtype = dtype.lower()
        if dtype not in {'float32', 'float16'}:
            raise self.argument_error('Argument "dtype" must be either float32 or float16.')

        self._model_path = model
        self._mask_path = mask
        self._mask_processors = mask_processors
        self._image_path = image
        self._image_processors = image_processors
        self._dtype = torch.float32 if dtype == 'float32' else torch.float16
        self._pre_resize = pre_resize
        self._model = self._load_inpaint_model(model, self._dtype)

        self.register_module(self._model)

    def _load_inpaint_model(self, model_path, dtype):
        """
        Load an inpainting model from a file path or URL using proper caching.

        :param model_path: path to model file
        :param dtype: torch dtype to use
        :return: model
        """
        og_path = model_path

        if _webcache.is_downloadable_url(model_path):
            # Any mimetype
            try:
                model_path = _hfhub.webcache_or_hf_blob_download(
                    model_path, local_files_only=self.local_files_only
                )
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "model": "{model_path}", error: {e}') from e

        # use the model file size as a heuristic
        self.set_size_estimate(os.path.getsize(model_path))

        def load_method():
            try:
                inpaint_model = spandrel.ModelLoader().load_from_file(model_path).eval()

                # Check FP16 support and fallback if needed
                if dtype == torch.float16 and not getattr(inpaint_model, 'supports_half', True):
                    _messages.warning(
                        f'inpaint processor: Model architecture "{_types.fullname(inpaint_model.architecture)}" '
                        f'does not support FP16, falling back to FP32 for model "{og_path}"'
                    )
                    dtype_to_use = torch.float32
                    # Update instance variable so other methods know about the fallback
                    self._dtype = torch.float32
                else:
                    dtype_to_use = dtype

                inpaint_model.to(dtype_to_use)

            except (ValueError,
                    RuntimeError,
                    TypeError,
                    AttributeError) as e:
                raise self.argument_error(
                    f'Unsupported model file format: {e}') from e

            return inpaint_model

        # make sure dtype is part of the cache tag
        model = self.load_object_cached(
            tag=model_path + str(dtype).split('.')[1],
            estimated_size=self.size_estimate,
            method=load_method
        )

        _messages.debug_log(
            f'{_types.fullname(InpaintProcessor._load_inpaint_model)}("{model_path}") -> {model.__class__.__name__}')

        return model

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
                mask_image = self._run_image_processor(
                    self._mask_processors,
                    mask_image,
                    aspect_correct=False,
                    resize_resolution=None,
                    align=1
                )

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
            raise self.argument_error(f'Failed to load argument "mask" from "{self._mask_path}": {e}')

    def _load_subject_image(self, target_size: _types.Size = None) -> PIL.Image.Image:
        """
        Load the subject image from file path or URL.
        
        :param target_size: Optional size to resize image to match input
        :return: Subject image as PIL Image
        """
        try:
            # Handle URL downloads using webcache
            if _webcache.is_downloadable_url(self._image_path):
                # Download and cache the URL
                _, image_file_path = _webcache.create_web_cache_file(
                    self._image_path,
                    mime_acceptable_desc='image files',
                    mimetype_is_supported=lambda m: m.startswith('image/'),
                    local_files_only=self.local_files_only
                )
                image_path = image_file_path
            else:
                # Use local file path directly
                image_path = self._image_path

            # Load subject image
            subject_image = PIL.Image.open(image_path)

            if self._image_processors is not None:
                subject_image = self._run_image_processor(
                    self._image_processors,
                    subject_image,
                    aspect_correct=False,
                    resize_resolution=None,
                    align=1
                )

            # Convert to RGB if needed
            if subject_image.mode != 'RGB':
                subject_image = subject_image.convert('RGB')

            # Resize image to match target size if specified using optimal interpolation
            if target_size is not None:
                old_size = subject_image.size  # (width, height)
                interpolation = _image.best_cv2_resampling(old_size, target_size)
                subject_array = np.array(subject_image)
                subject_array = cv2.resize(subject_array, target_size, interpolation=interpolation)
                subject_image = PIL.Image.fromarray(subject_array)

            return subject_image

        except Exception as e:
            raise self.argument_error(f'Failed to load argument "image" from "{self._image_path}": {e}')

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

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Process the image with Spandrel-based inpainting.
        
        :param image: Input PIL image
        :return: Processed PIL image
        """
        # Use the cached model
        model = self._model

        if self._mask_path:
            # Mask mode: incoming image is the subject, load mask from mask argument
            subject_image = image
            mask_array = self._load_mask(target_size=image.size)
            mask_image = PIL.Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
        else:
            # Image mode: incoming image is the mask, load subject from image argument
            subject_image = self._load_subject_image(target_size=image.size)
            # Convert incoming image to grayscale mask
            incoming_mask_image = image.convert('L') if image.mode != 'L' else image
            # Resize incoming mask to match subject image size if necessary
            if incoming_mask_image.size != subject_image.size:
                old_size = incoming_mask_image.size
                new_size = subject_image.size
                interpolation = _image.best_cv2_resampling(old_size, new_size)
                mask_array_temp = np.array(incoming_mask_image)
                mask_array_temp = cv2.resize(mask_array_temp, new_size, interpolation=interpolation)
                mask_image = PIL.Image.fromarray(mask_array_temp, mode='L')
            else:
                mask_image = incoming_mask_image

        # Ensure subject image is RGB
        if subject_image.mode != 'RGB':
            subject_image = subject_image.convert('RGB')

        # Perform Spandrel inpainting using our helper function
        try:
            result_image = _inpaint_pil(subject_image, mask_image, model)
            return result_image

        except Exception as e:
            raise self.argument_error(f'Spandrel inpainting failed: {e}')

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

    def to(self, device) -> "InpaintProcessor":
        """
        Move the processor and its model to the specified device.
        
        :param device: Target device
        :return: Self
        """
        super().to(device)
        if self._model is not None and hasattr(self._model, 'to'):
            self._model = self._model.to(device)
        return self
