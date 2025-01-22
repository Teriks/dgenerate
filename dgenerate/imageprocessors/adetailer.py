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

import os.path
import typing

import PIL.Image
import PIL.ImageOps
import diffusers
import torch

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.messages
import dgenerate.types as _types
import dgenerate.extras.asdff.base as _asdff
import dgenerate.pipelinewrapper.pipelines as _pipelines
from huggingface_hub import hf_hub_download
import dgenerate.image as _image
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.promptweighters as _promptweighters
import dgenerate.pipelinewrapper.enums as _enums


class AdetailerProcessor(_imageprocessor.ImageProcessor):
    r"""
    adetailer, diffusion based post processor for SD1.5, SDXL, SD3, and Flux

    adetailer can detect features of your image and automatically generate an inpaint
    mask for them, such as faces, hands etc. and then re-run diffusion over those portions
    of the image using inpainting to enhance detail.

    This image processor may only be used if a diffusion pipeline has been
    previously executed by dgenerate, that pipeline will be used to process
    the inpainting done by adetailer.  For a single command line invocation
    you must use --post-processors to use this image processor correctly. In
    dgenerate config script, you may use it anywhere, and the last executed
    diffusion pipeline will be reused for inpainting.

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

    The "model" argument specifies the YOLO detector model used to detect a feature
    of the image.

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
    when preforming inpainting on the input image.

    The "inference-steps" argument specifies the amount of inference steps
    when preforming inpainting on the input image.

    The "guidance-scale" argument specifies the guidance scale for inpainting.

    The "pag-scale" argument indicates the perturbed attention guidance scale,
    this enables a PAG inpaint pipeline if supported. If the previously used
    pipeline was a PAG pipeline, PAG is automatically enabled for inpainting
    if supported and this value defaults to 3.0 if not supplied. The adetailer
    processor supports PAG with --model-type torch and torch-sdxl.

    The "pag-adaptive-scale" argument indicates the perturbed attention guidance
    adaptive scale, this enables a PAG inpaint pipeline if supported.
    If the previously usee pipeline was a PAG pipeline, PAG is automatically
    enabled for inpainting if supported and this value defaults to 0.0 if
    not supplied. The adetailer processor supports PAG with
    --model-type torch and torch-sdxl.

    The "strength" argument is analogous to --image-seed-strengths

    The "mask-padding" argument indicates how much padding exists between the
    feature and the boundary of the mask area

    The "mask-blur" argument indicates the level of gaussian blur to apply
    to the generated inpaint mask, which can help with smooth blending in
    of the inpainted feature

    The "mask-dilation" argument indicates the amount of dilation applied
    to the inpaint mask, see: cv2.dilate

    """

    NAMES = ['adetailer']

    HIDE_ARGS = ['pipe', 'device', 'model-offload']

    def __init__(self,
                 model: str,
                 prompt: str,
                 negative_prompt: typing.Optional[str] = None,
                 prompt_weighter: typing.Optional[str] = None,
                 weight_name: typing.Optional[str] = None,
                 subfolder: typing.Optional[str] = None,
                 revision: typing.Optional[str] = None,
                 token: typing.Optional[str] = None,
                 local_files_only: bool = False,
                 seed: typing.Optional[int] = None,
                 inference_steps: int = 30,
                 guidance_scale: float = 5,
                 pag_scale: typing.Optional[float] = None,
                 pag_adaptive_scale: typing.Optional[float] = None,
                 strength: float = 0.4,
                 mask_padding: int = 32,
                 mask_blur: int = 4,
                 mask_dilation: int = 4,
                 pipe: diffusers.DiffusionPipeline = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._mask_padding = mask_padding
        self._mask_blur = mask_blur
        self._mask_dilation = mask_dilation
        self._inference_steps = inference_steps
        self._guidance_scale = guidance_scale
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale
        self._strength = strength
        self._seed = seed
        self._prompt_weighter = prompt_weighter

        self._pre_resize = pre_resize
        self._pipe = pipe

        if _hfutil.is_single_file_model_load(model):
            if os.path.exists(model):
                self._model_path = model
            else:
                if local_files_only:
                    raise self.argument_error(f'Could not find adetailer model: {model}')
                self._model_path = _hfutil.download_non_hf_model(model)
        else:
            try:
                self._model_path = hf_hub_download(
                    model,
                    filename=weight_name,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only)
            except Exception as e:
                raise self.argument_error(f'Error loading adetailer model: {e}')

    def _adetailer(self, image):
        i_filename = _image.get_filename(image)

        if self._pipe:
            last_pipe = self._pipe
        else:
            last_pipe = _pipelines.get_last_called_pipeline()

        if last_pipe is None:
            raise self.argument_error(
                'adetailer did not find an active previously used diffusion pipeline, '
                'please preform diffusion before attempting to use this processor.')

        is_flux = last_pipe.__class__.__name__.startswith('Flux') and \
                  not isinstance(last_pipe, diffusers.FluxFillPipeline)

        is_sdxl = last_pipe.__class__.__name__.startswith('StableDiffusionXL')
        is_sd3 = last_pipe.__class__.__name__.startswith('StableDiffusion3')
        is_sd = last_pipe.__class__.__name__.startswith('StableDiffusion') and not is_sd3 and not is_sdxl

        ad_pipe = _asdff.AdPipelineBase(last_pipe)

        common = {
            "num_inference_steps": self._inference_steps,
            "guidance_scale": self._guidance_scale,
            "prompt": self._prompt
        }

        if self._pag_scale is not None or \
                self._pag_adaptive_scale is not None:
            if not (is_sd or is_sdxl):
                raise self.argument_error(
                    'adetailer arguments "pag-scale" and "pag-adaptive-scale" may not '
                    'be used with anything other than --model-type torch and torch-sdxl')

            ad_pipe.force_pag = True

        if is_sdxl:
            common['target_size'] = image.size

        if not is_flux:
            common['negative_prompt'] = self._negative_prompt
        elif self._negative_prompt:
            dgenerate.messages.log(
                'adetailer is ignoring negative prompt, as Flux does not support negative prompting.')

        if self._prompt_weighter:
            loader = _promptweighters.PromptWeighterLoader()

            if is_flux:
                model_type = _enums.ModelType.TORCH_FLUX
            elif is_sdxl:
                model_type = _enums.ModelType.TORCH_SDXL
            elif is_sd3:
                model_type = _enums.ModelType.TORCH_SD3
            elif is_sd:
                model_type = _enums.ModelType.TORCH
            else:
                raise self.argument_error(
                    f'Pipeline: "{last_pipe.__class__.__name__}" does not support adetailer use.')

            if last_pipe.text_encoder is not None:
                encoder_dtype = next(last_pipe.text_encoder.parameters()).dtype
            elif last_pipe.text_encoder2 is not None:
                encoder_dtype = next(last_pipe.text_encoder2.parameters()).dtype
            else:
                raise self.argument_error(
                    'adetailer processor could not determine text encoder dtype for prompt weighting.')

            encoder_dtype = _enums.get_data_type_enum(str(encoder_dtype).lstrip('torch.'))

            weighter = loader.load(self._prompt_weighter,
                                   model_type=model_type,
                                   pipeline_type=_enums.PipelineType.INPAINT,
                                   dtype=encoder_dtype)

            common = weighter.translate_to_embeds(last_pipe, last_pipe._execution_device, common)

        if self._seed is not None:
            generator = torch.Generator(
                device=last_pipe._execution_device).manual_seed(self._seed)
            common['generator'] = generator

        inpaint_only = {'strength': self._strength}

        result = ad_pipe(
            common=common,
            inpaint_only=inpaint_only,
            images=[image],
            mask_dilation=self._mask_dilation,
            mask_blur=self._mask_blur,
            mask_padding=self._mask_padding,
            model_path=self._model_path)

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
