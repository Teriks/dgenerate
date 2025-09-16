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
import PIL.ImageOps
import diffusers
import torch

import dgenerate.extras.asdff.base as _asdff
import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.imageprocessors.util as _util


class AdetailerProcessor(_imageprocessor.ImageProcessor):
    r"""
    adetailer, diffusion based post processor for SD1.5, SDXL, Kolors, SD3, and Flux

    adetailer can detect features of your image and automatically generate an inpaint
    mask for them, such as faces, hands etc. and then re-run diffusion over those portions
    of the image using inpainting to enhance detail.

    This image processor may only be used if a diffusion pipeline has been
    previously executed by dgenerate, that pipeline will be used to process
    the inpainting done by adetailer. For a single command line invocation
    you must use --post-processors to use this image processor correctly. In
    dgenerate config script, you may use it anywhere, and the last executed
    diffusion pipeline will be reused for inpainting.

    Inpainting will occur on the device used by the last executed diffusion
    pipeline unless the "device" argument is specified, the detector model can be run on
    an alternate GPU if desired using the "detector-device" argument, otherwise
    the detector will run on "device".

    Example:

    NOWRAP!
    --post-processors "adetailer;\
                       model=Bingsu/adetailer;\
                       weight-name=face_yolov8n.pt;\
                       prompt=detailed image of a mans face;\
                       negative-prompt=nsfw, blurry, disfigured;\
                       guidance-scale=7;\
                       inference-steps=30;\
                       strength=0.4"

    -----

    The "model" argument specifies which YOLO model to use. This can be a path to a local
    model file, a URL to download the model from, or a HuggingFace repository slug / blob link.

    The "prompt" argument specifies the positive prompt to use for inpainting.

    The "negative-prompt" argument specifies the negative prompt for inpainting.

    The "prompt-weighter" argument specifies a prompt weighter plugin for applying
    prompt weighting to the provided positive and negative prompts. Prompt weighters
    may have arguments, when supplying URI arguments to a prompt weighter you must
    use double quoting around the prompt weighter definition, i.e:
    --post-processors "adetailer;model=...;prompt=test;prompt-weighter='compel;syntax=sdwui'"

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
    if needed.

    The "local-files-only" argument specifies that dgenerate should not attempt to
    download any model files, and to only look for them locally in the cache or
    otherwise.

    The "seed" argument can be used to specify a specific seed for diffusion
    when performing inpainting on the input image.

    The "inference-steps" argument specifies the amount of inference steps
    when performing inpainting on the input image.

    The "guidance-scale" argument specifies the guidance scale for inpainting.

    The "pag-scale" argument indicates the perturbed attention guidance scale,
    this enables a PAG inpaint pipeline if supported. If the previously used
    pipeline was a PAG pipeline, PAG is automatically enabled for inpainting
    if supported and this value defaults to 3.0 if not supplied. The adetailer
    processor supports PAG with --model-type sd and sdxl.

    The "pag-adaptive-scale" argument indicates the perturbed attention guidance
    adaptive scale, this enables a PAG inpaint pipeline if supported.
    If the previously usee pipeline was a PAG pipeline, PAG is automatically
    enabled for inpainting if supported and this value defaults to 0.0 if
    not supplied. The adetailer processor supports PAG with
    --model-type sd and sdxl.

    The "strength" argument is analogous to --image-seed-strengths

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
        index-filter=[0, "hand"]

    The "index-filter" argument is a list values or a single value that indicates
    what YOLO detection indices to keep, the index values start at zero. Detections are
    sorted by their top left bounding box coordinate from left to right, top to bottom,
    by (confidence descending). The order of detections in the image is identical to
    the reading order of words on a page (english). Inpainting will only be
    performed on the specified detection indices, if no indices are specified, then
    inpainting will be performed on all detections.

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

    The "detector-padding" argument specifies the amount of padding
    that will be added to the detection rectangle which is used to
    generate a masked area. The default is 0, you can make the mask
    area around the detected feature larger with positive padding
    and smaller with negative padding.

    Padding examples:

        NOWRAP!
        32 (32px Uniform, all sides)

        NOWRAP!
        10x20 (10px Horizontal, 20px Vertical)

        NOWRAP!
        10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)

    The "mask-padding" argument indicates how much padding to place around
    the masked area when cropping out the image to be inpainted. This value must be
    large enough to accommodate any feathering on the edge of the mask caused
    by "mask-blur" or "mask-dilation" for the best result, the default value is 32.
    The syntax for specifying this value is identical to "detector-padding".

    The "mask-shape" argument indicates what mask shape adetailer should
    attempt to draw around a detected feature, the default value is "rectangle".
    You may also specify "circle" to generate an ellipsoid shaped mask, which
    might be helpful for achieving better blending.

    The "mask-blur" argument indicates the level of gaussian blur to apply
    to the generated inpaint mask, which can help with smooth blending in
    of the inpainted feature

    The "mask-dilation" argument indicates the amount of dilation applied
    to the inpaint mask, see: cv2.dilate

    The "model-masks" argument indicates that masks generated by the
    model itself should be preferred over masks generated from the
    detection bounding box. If this is True, and the model itself
    returns mask data, "mask-shape", "mask-padding",
    and "detector-padding" will all be ignored.

    The "confidence" argument can be used to adjust the confidence
    value for the YOLO detector model. Defaults to: 0.3

    The "detector-device" argument can be used to specify a device
    override for the YOLO detector, i.e. the GPU / Accelerate device
    the model will run on. Example: cuda:0, cuda:1, cpu

    The "size" argument specifies the target size for processing detected areas.
    When specified, detected areas will always be scaled to this target size (with aspect ratio preserved)
    for processing, then scaled back to the original size for compositing.
    This can significantly improve detail quality for small detected features like faces or hands,
    or reduce processing time for overly large detected areas.
    
    The scaling is based on the larger dimension (width or height) of the detected area.
    If the detected area's larger dimension is smaller than the target size, it will be upscaled.
    If the detected area's larger dimension is larger than the target size, it will be downscaled.
    Scaling is always performed when this argument is specified.
    
    The value must be an integer greater than 1. The optimal resampling method is automatically
    selected based on whether upscaling or downscaling is needed.

    Example: size=1024 (always process detected areas at 1024px for the larger dimension)

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['adetailer']

    HIDE_ARGS = ['pipe', 'model-offload']

    OPTION_ARGS = {
        'mask-shape': ['r', 'rect', 'rectangle', 'c', 'circle', 'ellipse'],
    }

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]},
    }

    def __init__(self,
                 model: str,
                 prompt: str,
                 negative_prompt: str | None = None,
                 prompt_weighter: str | None = None,
                 weight_name: str | None = None,
                 subfolder: str | None = None,
                 revision: str | None = None,
                 token: str | None = None,
                 seed: int | None = None,
                 inference_steps: int = 30,
                 guidance_scale: float = 5,
                 pag_scale: float | None = None,
                 pag_adaptive_scale: float | None = None,
                 strength: float = 0.4,
                 detector_padding: int | str = _constants.DEFAULT_ADETAILER_DETECTOR_PADDING,
                 mask_shape: str = _constants.DEFAULT_ADETAILER_MASK_SHAPE,
                 class_filter: int | str | list | tuple | set | None = None,
                 index_filter: int | list | tuple | set | None = None,
                 mask_padding: int | str = _constants.DEFAULT_ADETAILER_MASK_PADDING,
                 mask_blur: int = _constants.DEFAULT_ADETAILER_MASK_BLUR,
                 mask_dilation: int = _constants.DEFAULT_ADETAILER_MASK_DILATION,
                 model_masks: bool = False,
                 confidence: float = _constants.DEFAULT_ADETAILER_DETECTOR_CONFIDENCE,
                 detector_device: _types.OptionalName = None,
                 size: int | None = None,
                 pipe: diffusers.DiffusionPipeline = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if not isinstance(mask_padding, int):
            try:
                mask_padding = _textprocessing.parse_dimensions(mask_padding)

                if len(mask_padding) not in {1, 2, 4}:
                    raise ValueError()

            except ValueError:
                raise self.argument_error(
                    'mask-padding must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

            if len(mask_padding) == 1:
                mask_padding = mask_padding[0]

        if not isinstance(detector_padding, int):
            try:
                detector_padding = _textprocessing.parse_dimensions(detector_padding)

                if len(detector_padding) not in {1, 2, 4}:
                    raise ValueError()

            except ValueError as e:
                raise self.argument_error(
                    'detector-padding must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM') from e

            if len(detector_padding) == 1:
                detector_padding = detector_padding[0]

        mask_shape = mask_shape.lower()

        # Parse detection filters
        self._class_filter, self._index_filter = _util.yolo_filters_parse(
            class_filter,
            index_filter,
            self.argument_error
        )

        if mask_shape not in {'rectangle', 'circle'}:
            raise self.argument_error('mask-shape must be either "rectangle" or "circle".')

        if mask_blur < 0:
            raise self.argument_error('mask-blur may not be less than zero.')

        if mask_dilation < 0:
            raise self.argument_error('mask-dilation may not be less than zero.')

        if inference_steps <= 0:
            raise self.argument_error('inference-steps must be greater than zero.')

        if guidance_scale < 0:
            raise self.argument_error('guidance-scale may not be less than zero.')

        if pag_scale is not None and pag_scale < 0:
            raise self.argument_error('pag-scale may not be less than zero.')

        if pag_adaptive_scale is not None and pag_adaptive_scale < 0:
            raise self.argument_error('pag-adaptive-scale may not be less than zero.')

        if strength < 0:
            raise self.argument_error('strength may not be less than zero.')

        if strength > 1:
            raise self.argument_error('strength may not be greater than 1.')

        if confidence < 0.0:
            raise self.argument_error('confidence may not be less than 0.')

        if size is not None and size <= 1:
            raise self.argument_error('size must be an integer greater than 1.')

        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._detector_padding = detector_padding
        self._mask_shape = mask_shape
        self._mask_padding = mask_padding
        self._mask_blur = mask_blur
        self._mask_dilation = mask_dilation
        self._model_masks = model_masks
        self._inference_steps = inference_steps
        self._guidance_scale = guidance_scale
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale
        self._strength = strength
        self._seed = seed
        self._prompt_weighter = prompt_weighter
        self._detector_device = detector_device
        self._confidence = confidence
        self._size = size

        self._pre_resize = pre_resize
        self._pipe = pipe

        try:
            self._model_path = _uris.AdetailerDetectorUri(
                model=model,
                revision=revision,
                subfolder=subfolder,
                weight_name=weight_name,
                class_filter=self._class_filter,
                index_filter=self._index_filter,
                mask_shape=self._mask_shape,
                detector_padding=self._detector_padding,
                mask_padding=self._mask_padding,
                mask_blur=self._mask_blur,
                mask_dilation=self._mask_dilation,
                model_masks=self._model_masks,
                confidence=self._confidence,
                prompt=self._prompt,
                negative_prompt=self._negative_prompt,
                device=self._detector_device
            ).get_model_path(
                use_auth_token=token,
                local_files_only=self.local_files_only)
        except Exception as e:
            raise self.argument_error(str(e)) from e

    def _adetailer(self, image):
        i_filename = _image.get_filename(image)

        if self._pipe:
            last_pipe = self._pipe
        else:
            last_pipe = _pipelinewrapper.DiffusionPipelineWrapper.recall_last_used_main_pipeline()
            if last_pipe is not None:
                # we only want the primary pipe, not the sdxl refiner for instance
                last_pipe = last_pipe.pipeline

        if last_pipe is None:
            raise self.argument_error(
                'adetailer could not find the last image generation pipeline that was used '
                'for image generation, please perform an image generation operation before attempting to '
                'use this processor. This processor is best used with the --post-processors option '
                'of dgenerate. It is possible however, to use this processor elsewhere in a config '
                'script if image generation has occurred previously. It will re-use the last '
                'image generation pipelines components for inpainting.')

        is_flux = last_pipe.__class__.__name__.startswith('Flux') and \
                  not isinstance(last_pipe, diffusers.FluxFillPipeline)

        is_sdxl = last_pipe.__class__.__name__.startswith('StableDiffusionXL')
        is_sd3 = last_pipe.__class__.__name__.startswith('StableDiffusion3')
        is_sd = last_pipe.__class__.__name__.startswith('StableDiffusion') and not is_sd3 and not is_sdxl
        is_kolors = last_pipe.__class__.__name__.startswith('Kolors')

        ad_pipe = _asdff.AdPipelineBase(last_pipe)

        pipeline_args = {
            "num_inference_steps": self._inference_steps,
            "guidance_scale": self._guidance_scale,
            "prompt": self._prompt
        }

        if self._pag_scale is not None or \
                self._pag_adaptive_scale is not None:
            if not (is_sd or is_sdxl):
                raise self.argument_error(
                    'adetailer arguments "pag-scale" and "pag-adaptive-scale" may not '
                    'be used with anything other than --model-type sd and sdxl')

            ad_pipe.force_pag = True

            if self._pag_scale is not None:
                pipeline_args['pag_scale'] = self._pag_scale
            if self._pag_adaptive_scale is not None:
                pipeline_args['pag_adaptive_scale'] = self._pag_adaptive_scale

        if is_sdxl:
            pipeline_args['target_size'] = image.size

        if not is_flux:
            pipeline_args['negative_prompt'] = self._negative_prompt
        elif self._negative_prompt:
            dgenerate.messages.log(
                'adetailer is ignoring negative prompt, as Flux does not support negative prompting.')

        prompt_weighter = None

        if self._prompt_weighter:
            loader = _promptweighters.PromptWeighterLoader()

            if is_flux:
                model_type = _enums.ModelType.FLUX
            elif is_sdxl:
                model_type = _enums.ModelType.SDXL
            elif is_kolors:
                model_type = _enums.ModelType.KOLORS
            elif is_sd3:
                model_type = _enums.ModelType.SD3
            elif is_sd:
                model_type = _enums.ModelType.SD
            else:
                raise self.argument_error(
                    f'Pipeline: "{last_pipe.__class__.__name__}" does not support adetailer use.')

            if last_pipe.text_encoder is not None:
                encoder_dtype = next(last_pipe.text_encoder.parameters()).dtype
            elif hasattr(last_pipe, 'text_encoder2') and last_pipe.text_encoder2 is not None:
                encoder_dtype = next(last_pipe.text_encoder2.parameters()).dtype
            elif hasattr(last_pipe, 'text_encoder3') and last_pipe.text_encoder3 is not None:
                encoder_dtype = next(last_pipe.text_encoder3.parameters()).dtype
            else:
                raise self.argument_error(
                    'adetailer processor could not determine text encoder dtype for prompt weighting.')

            encoder_dtype = _enums.get_data_type_enum(str(encoder_dtype).lstrip('torch.'))

            try:
                prompt_weighter = loader.load(
                    self._prompt_weighter,
                    model_type=model_type,
                    dtype=encoder_dtype,
                    local_files_only=self.local_files_only,
                    device=self.device
                )
            except Exception as e:
                raise self.argument_error(str(e)) from e

        if self._seed is not None:
            generator = torch.Generator(
                device=self.device).manual_seed(self._seed)
            pipeline_args['generator'] = generator

        pipeline_args['strength'] = self._strength

        result = ad_pipe(
            pipeline_args=pipeline_args,
            images=[image],
            mask_shape=self._mask_shape,
            mask_dilation=self._mask_dilation,
            mask_blur=self._mask_blur,
            mask_padding=self._mask_padding,
            model_masks=self._model_masks,
            detector_padding=self._detector_padding,
            model_path=self._model_path,
            device=self.device,
            detector_device=_types.default(self._detector_device, self.device),
            confidence=self._confidence,
            prompt_weighter=prompt_weighter,
            class_filter=self._class_filter,
            index_filter=self._index_filter,
            processing_size=self._size
        )

        if len(result.images) > 0:
            output_image = result.images[0]
            output_image.filename = i_filename
        else:
            output_image = image

        return output_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        if self._pre_resize:
            return self._adetailer(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._adetailer(image)
        return image

    def to(self, device) -> "AdetailerProcessor":
        """
        Does nothing for this processor.

        :param device: the device
        :return: this processor
        """
        return self
