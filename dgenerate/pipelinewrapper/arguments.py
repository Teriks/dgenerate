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
import collections.abc
import typing

import PIL.Image

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class DiffusionArguments(_types.SetFromMixin):
    """
    Represents all possible arguments for a :py:class:`.DiffusionPipelineWrapper` call.
    """
    prompt: _types.OptionalPrompt = None
    """
    Primary prompt
    """

    images: _types.Images | None = None
    """
    Images for img2img operations, or the base for inpainting operations.
    
    All input images involved in a generation except for ``adapter_images`` must match in dimension and be aligned by 8 pixels,
    except in the case of Stable Cascade, which can accept multiple images of any dimension for the purpose of image based
    prompting similar to IP Adapters.
    
    All other pipelines interpret multiple image inputs as a batching request.
    """

    mask_images: _types.Images | None = None
    """
    Mask images for inpainting operations.
    
    The amount of img2img ``images`` must be equal to the amount of ``mask_images`` supplied.
    
    All input images involved in a generation except for ``adapter_images``  must match in dimension and be aligned by 8 pixels,
    except in the case of Stable Cascade, which can accept multiple images of any dimension for the purpose of image based
    prompting similar to IP Adapters.  Stable Cascade cannot perform inpainting, so ``mask_images`` is irrelevant in
    this case. All other pipelines interpret multiple image inputs as a batching request.
    """

    control_images: _types.Images | None = None
    """
    ControlNet guidance images to use if ``controlnet_uris`` were given to the 
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    All input images involved in a generation must match in dimension and be aligned by 8 pixels.
    """

    ip_adapter_images: collections.abc.Sequence[_types.Images] | None = None
    """
    IP Adapter images to use if ``ip_adapter_uris`` were given to the
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    This should be a list of ``Sequence[PIL.Image]``
    
    Each list entry corresponds to an IP adapter URI.
    
    Multiple IP Adapter URIs can be provided, each IP Adapter can get its own set of images.
    """

    floyd_image: PIL.Image.Image | None = None
    """
    The output image of the last stage when preforming img2img or 
    inpainting generation with Deep Floyd. When preforming txt2img 
    generation :py:attr:`DiffusionArguments.image` is used.
    """

    width: _types.OptionalInteger = None
    """
    Output image width.
    
    Ignored when img2img, inpainting, or controlnet guidance images are involved.
    
    Width will be the width of the input image in those cases.
    
    Output image width, must be aligned by 8
    """

    height: _types.OptionalInteger = None
    """
    Output image height.
    
    Ignored when img2img, inpainting, or controlnet guidance images are involved.
    
    Width will be the width of the input image in those cases.
    
    Output image width, must be aligned by 8
    """

    batch_size: _types.OptionalInteger = None
    """
    Number of images to produce in a single generation step on the same GPU.
    """

    s_cascade_decoder_inference_steps: _types.OptionalInteger = None
    """
    Inference steps value for the Stable Cascade decoder, this corresponds 
    to the ``--s-cascade-decoder-inference-steps`` argument of the dgenerate command line tool.
    """

    s_cascade_decoder_guidance_scale: _types.OptionalFloat = None
    """
    Guidance scale value for the Stable Cascade decoder, this 
    corresponds to the ``--s-cascade-decoder-guidance-scales`` argument of the dgenerate
    command line tool.
    """

    s_cascade_decoder_prompt: _types.OptionalPrompt = None
    """
    Primary prompt for the Stable Cascade decoder when a decoder URI is specified in the 
    constructor of :py:class:`.DiffusionPipelineWrapper`. Usually the ``prompt``
    attribute of this object is used, unless you override it by giving this attribute
    a value.
    """

    sd3_max_sequence_length: _types.OptionalInteger = None
    """
    Max number of prompt tokens that the T5EncoderModel (text encoder 3) of Stable Diffusion 3 can handle.
    
    This defaults to 256 when not specified, and the maximum value is 512 and the minimum value is 1.
    
    High values result in more resource usage and processing time.
    """

    flux_max_sequence_length: _types.OptionalInteger = None
    """
    Max number of prompt tokens that the T5EncoderModel (text encoder 2) of Flux can handle.
    
    This defaults to 512 when not specified, and the maximum value is 512 and the minimum value is 1.
    
    High values result in more resource usage and processing time.
    """

    sd3_second_prompt: _types.OptionalPrompt = None
    """
    Secondary prompt for the SD3 main pipeline. Usually the ``prompt``
    attribute of this object is used, unless you override it by giving 
    this attribute a value.
    """

    sd3_third_prompt: _types.OptionalPrompt = None
    """
    Tertiary (T5) prompt for the SD3 main pipeline. Usually the ``prompt``
    attribute of this object is used, unless you override it by giving 
    this attribute a value.
    """

    flux_second_prompt: _types.OptionalPrompt = None
    """
    Secondary prompt for the Flux pipeline. Usually the ``prompt`` attribute
    of this object is used, unless you override it by giving this attribute a
    value.
    """

    sdxl_refiner_edit: _types.OptionalBoolean = None
    """
    Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode.
    """

    sdxl_second_prompt: _types.OptionalPrompt = None
    """
    Secondary prompt for the SDXL main pipeline when a refiner URI is specified in the 
    constructor of :py:class:`.DiffusionPipelineWrapper`. Usually the ``prompt``
    attribute of this object is used, unless you override it by giving this attribute
    a value.
    """

    sdxl_refiner_prompt: _types.OptionalPrompt = None
    """
    Primary prompt for the SDXL refiner when a refiner URI is specified in the 
    constructor of :py:class:`.DiffusionPipelineWrapper`. Usually the ``prompt``
    attribute of this object is used, unless you override it by giving this attribute
    a value.
    """

    sdxl_refiner_second_prompt: _types.OptionalPrompt = None
    """
    Secondary prompt for the SDXL refiner when a refiner URI is specified in the 
    constructor of :py:class:`.DiffusionPipelineWrapper`. Usually the **sdxl_refiner_prompt**
    attribute of this object is used, unless you override it by giving this attribute
    a value.
    """

    seed: _types.OptionalInteger = None
    """
    An integer to serve as an RNG seed.
    """

    image_seed_strength: _types.OptionalFloat = None
    """
    Image seed strength, which relates to how much an img2img source (**image** attribute)
    is used during generation. Between 0.001 (close to zero but not 0) and 1.0, the closer to
    1.0 the less the image is used for generation, IE. the more creative freedom the AI has.
    """

    sdxl_t2i_adapter_factor: _types.OptionalFloat = None
    """
    SDXL specific T2I adapter factor, this controls the amount of time-steps for which a T2I adapter applies
    guidance to an image, this is a value between 0.0 and 1.0. A value of 0.5 for example
    indicates that the T2I adapter is only active for half the amount of time-steps it takes
    to completely render an image. 
    """

    upscaler_noise_level: _types.OptionalInteger = None
    """
    Upscaler noise level for the :py:attr:`dgenerate.pipelinewrapper.ModelType.TORCH_UPSCALER_X4` model type only.
    """

    sdxl_high_noise_fraction: _types.OptionalFloat = None
    """
    SDXL high noise fraction. This proportion of timesteps/inference steps are handled by the primary model,
    while the inverse proportion is handled by the refiner model when an SDXL **model_type** value.
    
    When the refiner is operating in edit mode the number of total inference steps
    for the refiner will be calculated in a different manner, currently the
    refiner operates in edit mode during generations involving ControlNets as 
    well as inpainting. 
    
    In edit mode, the refiner uses img2img with an image seed strength
    to add details to the image instead of cooperative denoising, this image 
    seed strength is calculated as (1.0 - :py:attr:`.DiffusionArguments.sdxl_high_noise_fraction`), and the 
    number of inference steps for the refiner is then calculated as 
    (image_seed_strength * inference_steps).
    """

    sdxl_refiner_inference_steps: _types.OptionalInteger = None
    """
    Override the default amount of inference steps preformed by the SDXL refiner. 
    Which is normally set to the value for the primary model.
    
    The attribute :py:attr:`.DiffusionArguments.sdxl_high_noise_fraction` still 
    factors in to the actual amount of inference steps preformed.
    """

    sdxl_refiner_guidance_scale: _types.OptionalFloat = None
    """
    Override the guidance scale used by the SDXL refiner, which is normally set to the value of
    :py:attr:`.DiffusionArguments.guidance_scale`.
    """

    sdxl_refiner_guidance_rescale: _types.OptionalFloat = None
    """
    Override the guidance rescale value used by the SDXL refiner, which is normally set to the value of
    :py:attr:`.DiffusionArguments.guidance_rescale`.
    """

    sdxl_aesthetic_score: _types.OptionalFloat = None
    """
    Optional, defaults to 6.0. This argument is used for img2img and inpainting operations only
    Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
    Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
    """

    sdxl_original_size: _types.OptionalSize = None
    """
    Optional SDXL conditioning parameter. 
    If :py:attr:`.DiffusionArguments.sdxl_original_size` is not the same as :py:attr:`.DiffusionArguments.sdxl_target_size` 
    the image will appear to be down- or up-sampled. :py:attr:`.DiffusionArguments.sdxl_original_size` defaults to ``(width, height)`` 
    if not specified or the size of any input images provided. Part of SDXL's micro-conditioning as explained in section 2.2 of 
    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
    """

    sdxl_target_size: _types.OptionalSize = None
    """
    Optional SDXL conditioning parameter.
    For most cases, :py:attr:`.DiffusionArguments.sdxl_target_size` should be set to the desired height and width of the generated image. If
    not specified it will default to ``(width, height)`` or the size of any input images provided. Part of SDXL's 
    micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
    """

    sdxl_crops_coords_top_left: _types.OptionalCoordinate = None
    """
    Optional SDXL conditioning parameter.
    :py:attr:`.DiffusionArguments.sdxl_crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
    :py:attr:`.DiffusionArguments.sdxl_crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
    :py:attr:`.DiffusionArguments.sdxl_crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
    """

    sdxl_negative_aesthetic_score: _types.OptionalFloat = None
    """
    Negative influence version of :py:attr:`.DiffusionArguments.sdxl_aesthetic_score`
    """

    sdxl_negative_original_size: _types.OptionalSize = None
    """
    This value is only supported for certain :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
    configurations, an error will be produced when it is unsupported. It is not known to be supported
    by ``pix2pix``.
    
    Optional SDXL conditioning parameter.
    To negatively condition the generation process based on a specific image resolution. Part of SDXL's
    micro-conditioning as explained in section 2.2 of
    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
    """

    sdxl_negative_target_size: _types.OptionalSize = None
    """
    This value is only supported for certain :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
    configurations, an error will be produced when it is unsupported. It is not known to be supported
    by ``pix2pix``.
    
    Optional SDXL conditioning parameter.
    To negatively condition the generation process based on a target image resolution. It should be as same
    as the :py:attr:`.DiffusionArguments.target_size` for most cases. Part of SDXL's micro-conditioning as 
    explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
    For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
    """

    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    """
    Negative influence version of :py:attr:`.DiffusionArguments.sdxl_crops_coords_top_left`
    """

    sdxl_refiner_aesthetic_score: _types.OptionalFloat = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_aesthetic_score`
    """

    sdxl_refiner_original_size: _types.OptionalSize = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_original_size`
    """

    sdxl_refiner_target_size: _types.OptionalSize = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_target_size`
    """

    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinate = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_crops_coords_top_left`
    """

    sdxl_refiner_negative_aesthetic_score: _types.OptionalFloat = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_negative_aesthetic_score`
    """

    sdxl_refiner_negative_original_size: _types.OptionalSize = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_negative_original_size`
    """

    sdxl_refiner_negative_target_size: _types.OptionalSize = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_negative_target_size`
    """

    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    """
    Override the refiner value usually taken from :py:attr:`.DiffusionArguments.sdxl_negative_crops_coords_top_left`
    """

    guidance_scale: _types.OptionalFloat = None
    """
    A higher guidance scale value encourages the model to generate images closely linked to the text
    :py:attr:`.DiffusionArguments.prompt` at the expense of lower image quality. Guidance scale is enabled 
    when :py:attr:`.DiffusionArguments.guidance_scale`  > 1
    """

    pag_scale: _types.OptionalFloat = None
    """
    Perturbed attention guidance scale.
    """

    pag_adaptive_scale: _types.OptionalFloat = None
    """
    Adaptive perturbed attention guidance scale.
    """

    sdxl_refiner_pag_scale: _types.OptionalFloat = None
    """
    Perturbed attention guidance scale for the SDXL refiner.
    """

    sdxl_refiner_pag_adaptive_scale: _types.OptionalFloat = None
    """
    Adaptive perturbed attention guidance scale for the SDXL refiner.
    """

    image_guidance_scale: _types.OptionalFloat = None
    """
    This value is only relevant for ``pix2pix`` :py:class:`dgenerate.pipelinewrapper.ModelType`.
    
    Image guidance scale is to push the generated image towards the initial image :py:attr:`.DiffusionArguments.image`. 
    Image guidance scale is enabled by setting :py:attr:`.DiffusionArguments.image_guidance_scale` > 1. Higher image 
    guidance scale encourages to generate images that are closely linked to the source image :py:attr:`.DiffusionArguments.image`, 
    usually at the expense of lower image quality.
    """

    guidance_rescale: _types.OptionalFloat = None
    """
    This value is only supported for certain :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
    configurations, an error will be produced when it is unsupported.
    
    Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    Guidance rescale factor should fix overexposure when using zero terminal SNR.
    """

    inference_steps: _types.OptionalInteger = None
    """
    The number of denoising steps. More denoising steps usually lead to a higher quality image 
    at the expense of slower inference.
    """

    clip_skip: _types.OptionalInteger = None
    """
    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
    the output of the pre-final layer will be used for computing the prompt embeddings. Only supported for 
    ``model_type`` values ``torch`` and ``torch-sdxl``, including with ``controlnet_uris`` defined.
    """

    sdxl_refiner_clip_skip: _types.OptionalInteger = None
    """
    Clip skip override value for the SDXL refiner, which normally defaults to that of 
    :py:attr:`.DiffusionArguments.clip_skip` when it is defined.
    """

    def get_pipeline_wrapper_kwargs(self):
        """
        Get the arguments dictionary needed to call :py:class:`.DiffusionPipelineWrapper`

        :return: dictionary of argument names with values
        """
        pipeline_args = {}
        for attr, hint in typing.get_type_hints(self).items():
            val = getattr(self, attr)
            if not attr.startswith('_') and not (callable(val) or val is None):
                pipeline_args[attr] = val
        return pipeline_args

    def determine_pipeline_type(self):
        """
        Determine the :py:attr:`dgenerate.pipelinewrapper.PipelineType` needed to utilize these arguments.

        :return: :py:attr:`dgenerate.pipelinewrapper.PipelineType`
        """

        if self.images is not None and self.mask_images is not None:
            # Inpainting is handled by INPAINT type
            return _enums.PipelineType.INPAINT

        if self.images is not None:
            # Image only is handled by IMG2IMG type
            return _enums.PipelineType.IMG2IMG

        # All other situations handled by TXT2IMG type
        return _enums.PipelineType.TXT2IMG

    @staticmethod
    def _describe_prompt(prompt_format, prompt: _prompt.Prompt, pos_title, neg_title):
        if prompt is None:
            return

        prompt_wrap_width = _textprocessing.long_text_wrap_width()
        prompt_val = prompt.positive
        if prompt_val:
            header = f'{pos_title}: '
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
                    subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

        prompt_val = prompt.negative
        if prompt_val:
            header = f'{neg_title}: '
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
                    subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

    def describe_pipeline_wrapper_args(self) -> str:
        """
        Describe the pipeline wrapper arguments in a pretty, human-readable way, with word wrapping
        depending on console size or a maximum length depending on what stdout currently is.

        :return: description string.
        """
        prompt_format = []
        DiffusionArguments._describe_prompt(
            prompt_format, self.prompt,
            "Prompt",
            "Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sd3_second_prompt,
            "SD3 Second Prompt",
            "SD3 Second Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sd3_third_prompt,
            "SD3 Third Prompt",
            "SD3 Third Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.flux_second_prompt,
            "Flux Second Prompt",
            "Flux Second Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.s_cascade_decoder_prompt,
            "Stable Cascade Decoder Prompt",
            "Stable Cascade Decoder Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_second_prompt,
            "SDXL Second Prompt",
            "SDXL Second Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_prompt,
            "SDXL Refiner Prompt",
            "SDXL Refiner Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_second_prompt,
            "SDXL Refiner Second Prompt",
            "SDXL Refiner Second Negative Prompt")

        prompt_format = '\n'.join(prompt_format)
        if prompt_format:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {self.seed}']

        descriptions = [
            (self.clip_skip, "Clip Skip:"),
            (self.sdxl_refiner_clip_skip, "SDXL Refiner Clip Skip:"),
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.s_cascade_decoder_inference_steps, 'Stable Cascade Decoder Inference Steps:'),
            (self.s_cascade_decoder_guidance_scale, 'Stable Cascade Decoder Guidance Scale:'),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_t2i_adapter_factor, "SDXL T2I Adapter Factor:"),
            (self.sdxl_refiner_inference_steps, "SDXL Refiner Inference Steps:"),
            (self.sdxl_refiner_pag_scale, 'SDXL Refiner PAG Scale:'),
            (self.sdxl_refiner_pag_adaptive_scale, 'SDXL Refiner PAG Adaptive Scale:'),
            (self.sdxl_refiner_guidance_scale, "SDXL Refiner Guidance Scale:"),
            (self.sdxl_refiner_guidance_rescale, "SDXL Refiner Guidance Rescale:"),
            (self.sdxl_aesthetic_score, "SDXL Aesthetic Score:"),
            (self.sdxl_original_size, "SDXL Original Size:"),
            (self.sdxl_target_size, "SDXL Target Size:"),
            (self.sdxl_crops_coords_top_left, "SDXL Top Left Crop Coords:"),
            (self.sdxl_negative_aesthetic_score, "SDXL Negative Aesthetic Score:"),
            (self.sdxl_negative_original_size, "SDXL Negative Original Size:"),
            (self.sdxl_negative_target_size, "SDXL Negative Target Size:"),
            (self.sdxl_negative_crops_coords_top_left, "SDXL Negative Top Left Crop Coords:"),
            (self.sdxl_refiner_aesthetic_score, "SDXL Refiner Aesthetic Score:"),
            (self.sdxl_refiner_original_size, "SDXL Refiner Original Size:"),
            (self.sdxl_refiner_target_size, "SDXL Refiner Target Size:"),
            (self.sdxl_refiner_crops_coords_top_left, "SDXL Refiner Top Left Crop Coords:"),
            (self.sdxl_refiner_negative_aesthetic_score, "SDXL Refiner Negative Aesthetic Score:"),
            (self.sdxl_refiner_negative_original_size, "SDXL Refiner Negative Original Size:"),
            (self.sdxl_refiner_negative_target_size, "SDXL Refiner Negative Target Size:"),
            (self.sdxl_refiner_negative_crops_coords_top_left, "SDXL Refiner Negative Top Left Crop Coords:"),
            (self.pag_scale, "PAG Scale:"),
            (self.pag_adaptive_scale, "PAG Adaptive Scale:"),
            (self.guidance_scale, "Guidance Scale:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:")
        ]

        for prompt_val, desc in descriptions:
            if prompt_val is not None:
                inputs.append(desc + ' ' + str(prompt_val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format
