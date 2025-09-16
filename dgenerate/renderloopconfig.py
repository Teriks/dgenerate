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
import random
import typing

import dgenerate.hfhub as _hfhub
import dgenerate.image as _image
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers as _promptupscalers
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types

IMAGE_PROCESSOR_SEP = '+'
"""
The character that is used to separate image processor chains
when specifying processors for individual images in a group
of input images. This is also used for latents processors.
"""


def _iterate_diffusion_args(**kwargs) -> collections.abc.Iterator[_pipelinewrapper.DiffusionArguments]:
    def _list_or_list_of_none(val: typing.Any) -> typing.List[typing.Any]:
        return val if val else [None]

    yield from _types.iterate_attribute_combinations(
        [(arg_name, _list_or_list_of_none(value)) for arg_name, value in kwargs.items()],
        _pipelinewrapper.DiffusionArguments)


def gen_seeds(n: int) -> list[int]:
    """
    Generate a list of N random seed integers

    :param n: number of seeds to generate
    :return: list of integer seeds
    """
    return [random.randint(0, 99999999999999) for _ in range(0, n)]


class RenderLoopConfigError(Exception):
    """
    Raised by :py:meth:`.RenderLoopConfig.check` on configuration errors.
    """
    pass


class RenderLoopConfig(_types.SetFromMixin):
    """
    This object represents configuration for :py:class:`RenderLoop`.

    It nearly directly maps to dgenerate's command line arguments.

    See subclass :py:class:`dgenerate.arguments.DgenerateArguments`
    """

    model_path: _types.OptionalPath = None
    """
    Primary diffusion model path, ``model_path`` argument of dgenerate command line tool.
    """

    subfolder: _types.OptionalPath = None
    """
    Primary model subfolder argument, ``--subfolder`` argument of dgenerate command line tool.
    """

    original_config: _types.OptionalPath = None
    """
    This option can be used to supply an original LDM config .yaml file that was provided with a single file checkpoint.
    """

    second_model_original_config: _types.OptionalPath = None
    """
    This option can be used to supply an original LDM config .yaml file that was provided with a single file checkpoint 
    for the secondary model, i.e. the SDXL Refiner or Stable Cascade Decoder.
    """

    sdxl_refiner_uri: _types.OptionalUri = None
    """
    SDXL Refiner model URI, ``--sdxl-refiner`` argument of dgenerate command line tool.
    """

    sdxl_refiner_edit: _types.OptionalBoolean = None
    """
    Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode.
    """

    batch_size: _types.OptionalInteger = None
    """
    Image generation batch size, ``--batch-size`` argument of dgenerate command line tool.
    """

    batch_grid_size: _types.OptionalSize = None
    """
    Optional image grid size specification for when **batch_size** is greater than one.
    This is the ``--batch-grid-size`` argument of the dgenerate command line tool.
    """

    prompt_weighter_uri: _types.OptionalUri = None
    """
    The URI of a prompt-weighter implementation supported by dgenerate.
    
    This corresponds to the ``--prompt-weighter`` argument of the dgenerate command line tool.
    """

    second_model_prompt_weighter_uri: _types.OptionalUri = None
    """
    The URI of a prompt-weighter implementation supported by dgenerate to 
    use with the SDXL refiner or Stable Cascade decoder.
    
    Defaults to :py:attr:`RenderLoopConfig.prompt_weighter_uri` if not specified.
    
    This corresponds to the ``--second-model-prompt-weighter`` argument of the dgenerate command line tool.
    """

    prompt_upscaler_uri: _types.OptionalUriOrUris = None
    """
    The URI of a prompt-upscaler implementation supported by dgenerate.
    
    This may also be a list of URIs, the prompt upscalers will be chained together sequentially.
    
    This corresponds to the ``--prompt-upscaler`` argument of the dgenerate command line tool.
    """

    second_model_prompt_upscaler_uri: _types.OptionalUriOrUris = None
    """
    The URI of a prompt-upscaler implementation supported by dgenerate to 
    use with the SDXL refiner or Stable Cascade decoder.
    
    Defaults to :py:attr:`RenderLoopConfig.prompt_upscaler_uri` if not specified.
    
    This may also be a list of URIs, the prompt upscalers will be chained together sequentially.
    
    This corresponds to the ``--second-model-prompt-upscaler`` argument of the dgenerate command line tool.
    """

    second_prompt_upscaler_uri: _types.OptionalUriOrUris = None
    """
    The URI of a prompt-upscaler implementation supported by dgenerate
    that applies to :py:attr:`RenderLoopConfig.second_prompts`
    
    Defaults to :py:attr:`RenderLoopConfig.prompt_upscaler_uri` if not specified.
    
    This may also be a list of URIs, the prompt upscalers will be chained together sequentially.
    
    This corresponds to the ``--second-prompt-upscaler`` argument of the dgenerate command line tool.
    """

    second_model_second_prompt_upscaler_uri: _types.OptionalUriOrUris = None
    """
    The URI of a prompt-upscaler implementation supported by dgenerate to 
    use with the SDXL refiner ``--second-prompts`` value. 
    
    Or rather :py:attr:`RenderLoopConfig.second_model_second_prompts`
    
    Defaults to :py:attr:`RenderLoopConfig.prompt_upscaler_uri` if not specified.
    
    This may also be a list of URIs, the prompt upscalers will be chained together sequentially.
    
    This corresponds to the ``--second-model-second-prompt-upscaler`` argument of the 
    dgenerate command line tool.
    """

    third_prompt_upscaler_uri: _types.OptionalUriOrUris = None
    """
    The URI of a prompt-upscaler implementation supported by dgenerate
    that applies to :py:attr:`RenderLoopConfig.third_prompts`
    
    Defaults to :py:attr:`RenderLoopConfig.prompt_upscaler_uri` if not specified.
    
    This may also be a list of URIs, the prompt upscalers will be chained together sequentially.
    
    This corresponds to the ``--third-prompt-upscaler`` argument of the dgenerate command line tool.
    """

    prompts: _prompt.Prompts
    """
    List of prompt objects, this corresponds to the ``--prompts`` argument of the dgenerate
    command line tool.
    """

    max_sequence_length: _types.OptionalInteger = None
    """
    Max number of prompt tokens that the T5EncoderModel (text encoder 3) of Stable Diffusion 3 or Flux can handle.
    
    This defaults to 256 for SD3 when not specified, and 512 for Flux.
    
    The maximum value is 512 and the minimum value is 1.
    
    High values result in more resource usage and processing time.
    """

    second_prompts: _prompt.OptionalPrompts = None
    """
    Optional list of SD3 / Flux secondary prompts, this corresponds to the ``--second-prompts`` argument
    of the dgenerate command line tool.
    """

    third_prompts: _prompt.OptionalPrompts = None
    """
    Optional list of SD3 tertiary prompts, this corresponds to the ``--third-prompts`` argument
    of the dgenerate command line tool. Flux does not support this argument.
    """

    sdxl_t2i_adapter_factors: _types.OptionalFloats = None
    """
    Optional list of SDXL specific T2I adapter factors to try, this controls the amount of time-steps for 
    which a T2I adapter applies guidance to an image, this is a value between 0.0 and 1.0. A value of 0.5 
    for example indicates that the T2I adapter is only active for half the amount of time-steps it takes
    to completely render an image. 
    """

    second_model_prompts: _prompt.OptionalPrompts = None
    """
    Optional list of SDXL refiner or Stable Cascade decoder prompt overrides, 
    this corresponds to the ``--second-model-prompts`` argument of the dgenerate 
    command line tool.
    """

    second_model_second_prompts: _prompt.OptionalPrompts = None
    """
    Optional list of SDXL refiner secondary prompt overrides, this 
    corresponds to the ``--second-model-second-prompts`` argument of the 
    dgenerate command line tool. The Stable Cascade Decoder does not 
    support this argument.
    """

    seeds: _types.Integers
    """
    List of integer seeds, this corresponds to the ``--seeds`` argument of 
    the dgenerate command line tool.
    """

    seeds_to_images: bool = False
    """
    Should :py:attr:`RenderLoopConfig.seeds` be interpreted as seeds for each
    image input instead of combinatorial? this includes control images.
    """

    guidance_scales: _types.Floats
    """
    List of floating point guidance scales, this corresponds to the ``--guidance-scales`` argument 
    of the dgenerate command line tool.
    """

    sigmas: typing.Optional[collections.abc.Sequence[collections.abc.Sequence[float] | str]] = None
    """
    One or more lists of sigma values to try. This is supported
    when using a :py:attr:`RenderLoopConfig.scheduler_uri` that supports 
    setting sigmas.
    
    Or: string expressions involving sigmas from the selected scheduler such as ``sigmas * 0.95``,
    sigmas will be represented as a numpy array, numpy is available through the namespace ``np``, 
    this uses ``asteval``.
    
    Lists of floats and strings representing expressions can be intermixed.
    
    Sigma values control the noise schedule in the diffusion process, allowing for 
    fine-grained control over how noise is added and removed during image generation.
    
    This corresponds to the ``--sigmas`` command line argument, which accepts 
    multiple comma-separated lists of floating point values, or singular values,
    or expressions denoted with: ``expr: ...``.
    
    You do not need to specify ``expr:`` when passing this value in the library,
    simply pass a string instead of a list of floats.
    
    Example: ``[[1.0,2.0,3.0], 'sigmas * 0.95']``
    """

    inference_steps: _types.Integers
    """
    List of inference steps values, this corresponds to the ``--inference-steps`` argument of the
    dgenerate command line tool.
    """

    clip_skips: _types.OptionalIntegers = None
    """
    List of clip skip values. Clip skip is the number of layers to be skipped from CLIP while computing the 
    prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing 
    the prompt embeddings. Only supported for ``model_type`` values ``sd`` and ``sdxl``, including with 
    ``controlnet_uris`` defined.
    """

    sdxl_refiner_clip_skips: _types.OptionalIntegers = None
    """
    Clip skip override values for the SDXL refiner, which normally defaults to the clip skip 
    value for the main model when it is defined.
    """

    image_seeds: _types.OptionalUris = None
    """
    List of ``--image-seeds`` URI strings.
    """

    parsed_image_seeds: _mediainput.OptionalParsedImageSeeds = None
    """
    The results of parsing URIs mentioned in :py:attr:`.RenderLoopConfig.image_seeds`, 
    will only be available if :py:meth:`.RenderLoopConfig.check` has been called.
    """

    image_seed_strengths: _types.OptionalFloats = None
    """
    Optional list of floating point image seed strengths, this corresponds to the ``--image-seed-strengths`` argument
    of the dgenerate command line tool.
    """

    upscaler_noise_levels: _types.OptionalIntegers = None
    """
    Optional list of integer upscaler noise levels, this corresponds to the ``--upscaler-noise-levels`` argument
    of the dgenerate command line tool that is used for the :py:attr:`dgenerate.pipelinewrapper.ModelType.UPSCALER_X4`
    model type only.
    """

    guidance_rescales: _types.OptionalFloats = None
    """
    List of floating point guidance rescale values which are supported by some pipelines, (there will be an
    error if it is unsupported upon running), this corresponds to the ``--guidance-rescales`` argument of 
    the dgenerate command line tool.
    """

    image_guidance_scales: _types.OptionalFloats = None
    """
    Optional list of floating point image guidance scales, used for pix2pix model types, this corresponds
    to the ``--image-guidance-scales`` argument of the dgenerate command line tool.
    """

    s_cascade_decoder_uri: _types.OptionalUri = None
    """
    Stable Cascade model URI, ``--s-cascade-decoder`` argument of dgenerate command line tool.
    """

    sdxl_high_noise_fractions: _types.OptionalFloats = None
    """
    Optional list of SDXL refiner high noise fractions (floats), this value is the fraction of inference steps
    that the base model handles, the inverse proportion of the provided fraction is handled by the refiner model.
    This corresponds to the ``--sdxl-high-noise-fractions`` argument of the dgenerate command line tool.
    """

    second_model_inference_steps: _types.OptionalIntegers = None
    """
    Optional list of inference steps value overrides for the SDXL refiner, this corresponds 
    to the ``--second-model-inference-steps`` argument of the dgenerate command line tool.
    """

    second_model_guidance_scales: _types.OptionalFloats = None
    """
    Optional list of guidance scale value overrides for the SDXL refiner or Stable Cascade decoder, 
    this corresponds to the ``--second-model-guidance-scales`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_sigmas: typing.Optional[collections.abc.Sequence[collections.abc.Sequence[float] | str]] = None
    """
    One or more lists of sigma values to try with the SDXL refiner. This is supported
    when using a :py:attr:`RenderLoopConfig.second_model_scheduler_uri` that supports 
    setting sigmas.
    
    Or: string expressions involving sigmas from the selected scheduler such as ``sigmas * 0.95``,
    sigmas will be represented as a numpy array, numpy is available through the namespace ``np``, 
    this uses ``asteval``.
    
    Lists of floats and strings representing expressions can be intermixed.
    
    Sigma values control the noise schedule in the diffusion process, allowing for 
    fine-grained control over how noise is added and removed during image generation.
    
    This corresponds to the ``--sdxl-refiner-sigmas`` command line argument, which accepts 
    multiple comma-separated lists of floating point values, or singular values,
    or expressions denoted with: ``expr: ...``.
    
    You do not need to specify ``expr:`` when passing this value in the library,
    simply pass a string instead of a list of floats.
    
    Example: ``[[1.0,2.0,3.0], 'sigmas * 0.95']``
    """

    sdxl_refiner_guidance_rescales: _types.OptionalFloats = None
    """
    Optional list of guidance rescale value overrides for the SDXL refiner or Stable Cascade decoder, 
    this corresponds to the ``--sdxl-refiner-guidance-rescales`` argument of the dgenerate command line tool.
    """

    sdxl_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of SDXL aesthetic-score conditioning values, this corresponds 
    to the ``--sdxl-aesthetic-scores`` argument of the dgenerate command line tool.
    """

    sdxl_original_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL original-size micro-conditioning parameters, this corresponds 
    to the ``--sdxl-original-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_target_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL target-size micro-conditioning parameters, this corresponds 
    to the ``--sdxl-target-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_crops_coords_top_left: _types.OptionalCoordinates = None
    """
    Optional list of SDXL top-left-crop-coords micro-conditioning parameters, this corresponds 
    to the ``--sdxl-crops-coords-top-left`` argument of the dgenerate command line tool.
    """

    sdxl_negative_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of negative influence SDXL aesthetic-score conditioning values, 
    this corresponds to the ``--sdxl-negative-aesthetic-scores`` argument of the dgenerate
    command line tool.
    """

    sdxl_negative_original_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL original-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-original-sizes`` argument of the dgenerate command 
    line tool.
    """

    sdxl_negative_target_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL target-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-target-sizes`` argument of the dgenerate 
    command line tool.
    """

    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinates = None
    """
    Optional list of negative influence SDXL top-left crop coords micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-crops-coords-top-left`` argument of the dgenerate 
    command line tool.
    """

    sdxl_refiner_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of SDXL-refiner override aesthetic-score conditioning values, this 
    corresponds to the ``--sdxl-refiner-aesthetic-scores`` argument of the dgenerate command 
    line tool.
    """

    sdxl_refiner_original_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL-refiner override original-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-refiner-original-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_target_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL-refiner override target-size micro-conditioning parameters, this 
    corresponds to the ``--sdxl-refiner-target-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinates = None
    """
    Optional list of SDXL-refiner override top-left-crop-coords micro-conditioning parameters, this 
    corresponds to the ``--sdxl-refiner-crops-coords-top-left`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_negative_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of negative influence SDXL-refiner override aesthetic-score conditioning values, 
    this corresponds to the ``--sdxl-refiner-negative-aesthetic-scores`` argument of the dgenerate
    command line tool.
    """

    sdxl_refiner_negative_original_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL-refiner override original-size micro-conditioning 
    parameters, this corresponds to the ``--sdxl-refiner-negative-original-sizes`` argument of 
    the dgenerate command line tool.
    """

    sdxl_refiner_negative_target_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL-refiner override target-size micro-conditioning 
    parameters, this corresponds to the ``--sdxl-refiner-negative-target-sizes`` argument of 
    the dgenerate command line tool.
    """

    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinates = None
    """
    Optional list of negative influence SDXL-refiner top-left crop coords micro-conditioning parameters, 
    this corresponds to the ``--sdxl-refiner-negative-crops-coords-top-left`` argument of the dgenerate 
    command line tool.
    """

    unet_uri: _types.OptionalUri = None
    """
    Optional user specified UNet URI, this corresponds to the ``--unet`` argument of the dgenerate command line tool.
    """

    second_model_unet_uri: _types.OptionalUri = None
    """
    Optional user specified second UNet URI, this corresponds to the ``--second-model-unet`` argument of the dgenerate 
    command line tool. This UNet uri will be used for the SDXL refiner or Stable Cascade decoder model.
    """

    transformer_uri: _types.OptionalUri = None
    """
    Optional user specified Transformer URI, this corresponds to the ``--transformer`` argument of the 
    dgenerate command line tool.
    
    This is currently only supported for Stable Diffusion 3 and Flux models.
    """

    vae_uri: _types.OptionalUri = None
    """
    Optional user specified VAE URI, this corresponds to the ``--vae`` argument of the dgenerate command line tool.
    """

    vae_tiling: bool = False
    """
    Enable VAE tiling? ``--vae-tiling``
    """

    vae_slicing: bool = False
    """
    Enable VAE slicing? ``--vae-slicing``
    """

    lora_uris: _types.OptionalUris = None
    """
    Optional user specified LoRA URIs, this corresponds to the ``--loras`` argument 
    of the dgenerate command line tool.
    """

    lora_fuse_scale: _types.OptionalFloat = None
    """
    Optional global LoRA fuse scale, this corresponds to the ``--lora-fuse-scale`` argument
    of the dgenerate command line tool.
    
    LoRA weights are merged into the main model at this scale. 
     
    When specifying multiple LoRA models, they are fused together into one set of 
    weights using their individual scale values, after which they are fused into the 
    main model at this scale value. 
    
    The default value when ``None`` is specified is ``1.0``.
    """

    image_encoder_uri: _types.OptionalUri = None
    """
    Optional user specified Image Encoder URI when using IP Adapter models or Stable Cascade.
    This corresponds to the ``--image-encoder`` argument of the dgenerate command line tool.
    
    If none of your specified ``--ip-adapters`` URIs point to a model which contains an Image Encoder
    model, you will need to specify one manually using this argument.
    """

    ip_adapter_uris: _types.OptionalUris = None
    """
    Optional user specified IP Adapter URIs, this corresponds to the ``--ip-adapters`` argument 
    of the dgenerate command line tool.
    """

    textual_inversion_uris: _types.OptionalUris = None
    """
    Optional user specified Textual Inversion URIs, this corresponds to the ``--textual-inversions``
    argument of the dgenerate command line tool.
    """

    text_encoder_uris: _types.OptionalUris = None
    """
    Optional user specified Text Encoder URIs, this corresponds to the ``--text-encoders``
    argument of the dgenerate command line tool.
    """

    second_model_text_encoder_uris: _types.OptionalUris = None
    """
    Optional user specified Text Encoder URIs, this corresponds to the ``--second-model-text-encoders``
    argument of the dgenerate command line tool. This specifies text encoders for the SDXL
    refiner or Stable Cascade decoder.
    """

    controlnet_uris: _types.OptionalUris = None
    """
    Optional user specified ControlNet URIs, this corresponds to the ``--control-nets`` argument
    of the dgenerate command line tool.
    """

    t2i_adapter_uris: _types.OptionalUris = None
    """
    Optional user specified T2IAdapter URIs, this corresponds to the ``--t2i-adapters`` argument
    of the dgenerate command line tool.
    """

    quantizer_uri: _types.OptionalUri = None
    """
    Global quantizer URI for main pipline, this corresponds to the ``--quantizer`` argument 
    of the dgenerate command line tool.
    
    The quantization backend and settings specified by this URI will be used globally 
    on the the most appropriate models associated with the main diffusion pipeline.
    """

    quantizer_map: _types.OptionalStrings = None
    """
    Collection of pipeline submodule names to which quantization should be applied when
    :py:attr`RenderLoopConfig.quantizer_uri` is provided. Valid values 
    include: ``unet``, ``transformer``, ``text_encoder``, ``text_encoder_2``, ``text_encoder_3``. 
    If ``None``, all supported modules will be quantized.
    """

    second_model_quantizer_uri: _types.OptionalUri = None
    """
    Global quantizer URI for secondary pipeline (SDXL Refiner or Stable Cascade decoder), 
    this corresponds to the ``--second-model-quantizer`` argument of the dgenerate command line tool.
    
    The quantization backend and settings specified by this URI will be used globally 
    on the the most appropriate models associated with the secondary diffusion pipeline 
    (SDXL Refiner, Stable Cascade Decoder).
    """

    second_model_quantizer_map: _types.OptionalStrings = None
    """
    Collection of secondary pipeline submodule names to which quantization should be applied when
    :py:attr`RenderLoopConfig.second_model_quantizer_uri` is provided. Valid values 
    include: ``unet``, ``transformer``, ``text_encoder``, ``text_encoder_2``, ``text_encoder_3``. 
    If ``None``, all supported modules will be quantized.
    """

    scheduler_uri: _types.OptionalUriOrUris = None
    """
    Optional primary model scheduler/sampler class name specification, this corresponds to the ``--scheduler``
    argument of the dgenerate command line tool. Setting this to 'help' will yield a help message to stdout
    describing scheduler names compatible with the current configuration upon running. Passing 'helpargs' will
    yield a help message with a list of overridable arguments for each scheduler and their typical defaults.
    
    This may be a list of schedulers, indicating to try each scheduler in turn.
    """

    freeu_params: typing.Optional[collections.abc.Sequence[tuple[float, float, float, float]]] = None
    """
    FreeU is a technique for improving image quality by re-balancing the contributions from 
    the UNet's skip connections and backbone feature maps.
    
    This can be used with no cost to performance, to potentially improve image quality.
    
    This argument can be used to specify The FreeU parameters: s1, s2, b1, and b2 in that order.
    
    You can specify the FreeU parameters as a list / sequence of tuples that will be
    tried in turn for generation.
    
    This argument only applies to models that utilize a UNet: SD1.5/2, SDXL, and Kolors
    
    See: https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
    
    And: https://github.com/ChenyangSi/FreeU
    """

    sdxl_refiner_freeu_params: typing.Optional[collections.abc.Sequence[tuple[float, float, float, float]]] = None
    """
    List / sequence of FreeU parameters to try for the SDXL refiner.
    
    See: :py:attr:`RenderLoopConfig.freeu_params` for clarification.
    """

    hi_diffusion: bool = False
    """
    Activate HiDiffusion for the primary model? 
            
    This can increase the resolution at which the model can
    output images while retaining quality with no overhead, and 
    possibly improved performance.
    
    See: https://github.com/megvii-research/HiDiffusion
    
    This is supported for: ``--model-type sd, sdxl, and kolors``.
    """

    hi_diffusion_no_win_attn: _types.OptionalBoolean = None
    """
    Disable window attention when using HiDiffusion for the primary model?
    
    This disables the MSW-MSA (Multi-Scale Window Multi-Head Self-Attention) component of HiDiffusion.
    
    See: https://github.com/megvii-research/HiDiffusion
    
    This is supported for: ``--model-type sd, sdxl, and kolors``.
    """

    hi_diffusion_no_raunet: _types.OptionalBoolean = None
    """
    Disable RAU-Net when using HiDiffusion for the primary model?
    
    This disables the Resolution-Aware U-Net component of HiDiffusion.
    
    See: https://github.com/megvii-research/HiDiffusion
    
    This is supported for: ``--model-type sd, sdxl, and kolors``.
    """

    sada: bool = False
    """
    Enable SADA (Stability-guided Adaptive Diffusion Acceleration) with default parameters for the primary model.
    
    Specifying this alone is equivalent to setting all SADA parameters to their model-specific default values:

    - SD/SD2: ``sada_max_downsamples=1``, ``sada_sxs=3``, ``sada_sys=3``, ``sada_lagrange_terms=4``, ``sada_lagrange_ints=4``, ``sada_lagrange_steps=24``, ``sada_max_fixes=5120``
    - SDXL/Kolors: ``sada_max_downsamples=2``, ``sada_sxs=3``, ``sada_sys=3``, ``sada_lagrange_terms=4``, ``sada_lagrange_ints=4``, ``sada_lagrange_steps=24``, ``sada_max_fixes=10240``
    - Flux: ``sada_max_downsamples=0``, ``sada_lagrange_terms=3``, ``sada_lagrange_ints=4``, ``sada_lagrange_steps=20``, ``sada_max_fixes=0``
    
    SADA is not compatible with HiDiffusion, DeepCache, or TeaCache.
    """

    sada_max_downsamples: _types.OptionalIntegers = None
    """
    SADA maximum downsample factor for the primary model.
    
    Controls the maximum downsample factor in the SADA algorithm. 
    Lower values can improve quality but may reduce speedup.
    
    Model-specific defaults:
    
    - SD/SD2: 1
    - SDXL/Kolors: 2
    - Flux: 0
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_sxs: _types.OptionalIntegers = None
    """
    SADA spatial downsample factor X for the primary model.
    
    Controls the spatial downsample factor in the X dimension.
    Higher values can increase speedup but may affect quality.
    
    Model-specific defaults:
    
    - SD/SD2: 3
    - SDXL/Kolors: 3
    - Flux: 0 (not used)
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_sys: _types.OptionalIntegers = None
    """
    SADA spatial downsample factor Y for the primary model.
    
    Controls the spatial downsample factor in the Y dimension.
    Higher values can increase speedup but may affect quality.
    
    Model-specific defaults:
    
    - SD/SD2: 3
    - SDXL/Kolors: 3
    - Flux: 0 (not used)
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_acc_ranges: _types.OptionalRanges = None
    """
    SADA acceleration range start / end steps for the primary model.
    
    Defines the starting / ending step for SADA acceleration. 
    
    Starting step must be at least 3 as SADA leverages third-order dynamics.
    
    Defaults to [[10,47]].
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_lagrange_terms: _types.OptionalIntegers = None
    """
    SADA Lagrangian interpolation terms for the primary model.
    
    Number of terms to use in Lagrangian interpolation. 
    Set to 0 to disable Lagrangian interpolation.
    
    Model-specific defaults:
    
    - SD/SD2: 4
    - SDXL/Kolors: 4
    - Flux: 3
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_lagrange_ints: _types.OptionalIntegers = None
    """
    SADA Lagrangian interpolation interval for the primary model.
    
    Interval for Lagrangian interpolation. Must be compatible with 
    sada_lagrange_step (lagrange_step % lagrange_int == 0).
    
    Model-specific defaults:
    
    - SD/SD2: 4
    - SDXL/Kolors: 4
    - Flux: 4
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_lagrange_steps: _types.OptionalIntegers = None
    """
    SADA Lagrangian interpolation step for the primary model.
    
    Step value for Lagrangian interpolation. Must be compatible with 
    sada_lagrange_int (lagrange_step % lagrange_int == 0).
    
    Model-specific defaults:
    
    - SD/SD2: 24
    - SDXL/Kolors: 24
    - Flux: 20
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_max_fixes: _types.OptionalIntegers = None
    """
    SADA maximum fixed memory for the primary model.
    
    Maximum amount of fixed memory to use in SADA optimization.
    
    Model-specific defaults:
    
    - SD/SD2: 5120 (5 * 1024)
    - SDXL/Kolors: 10240 (10 * 1024)
    - Flux: 0
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """

    sada_max_intervals: _types.OptionalIntegers = None
    """
    SADA maximum interval for optimization for the primary model.
    
    Maximum interval between optimizations in the SADA algorithm.
    
    Defaults to 4.
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    
    Each value supplied will be tried in turn.
    """


    tea_cache: bool = False
    """
    Activate TeaCache for the primary model?
    
    This is supported for Flux, teacache uses a novel caching mechanism 
    in the forward pass of the flux transformer to reduce the amount of
    computation needed to generate an image, this can speed up inference
    with small amounts of quality loss.
    
    See: https://github.com/ali-vilab/TeaCache
    
    Also see: :py:attr:`RenderLoopConfig.tea_cache_rel_l1_thresholds`
    
    This is supported for: ``--model-type flux*``.
    
    """

    tea_cache_rel_l1_thresholds: _types.OptionalFloats = None
    """
    TeaCache relative L1 thresholds to try when :py:attr:`RenderLoopConfig.tea_cache` is enabled.
            
    This should be one or more float values between 0.0 and 1.0, each value will be tried in turn.
    Higher values mean more speedup.

    Defaults to 0.6 (2.0x speedup). 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 
    0.6 for 2.0x speedup, 0.8 for 2.25x speedup

    See: https://github.com/ali-vilab/TeaCache
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.tea_cache` is enabled.

    This is supported for: ``--model-type flux*``.
    """

    deep_cache: bool = False
    """
    Activate DeepCache for the main model?
    
    DeepCache caches intermediate attention layer outputs to speed up
    the diffusion process. This is beneficial for higher inference steps.
                  
    See: https://github.com/horseee/DeepCache
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    """

    deep_cache_intervals: _types.OptionalIntegers = None
    """
    Cache intervals to try for DeepCache for the main model.
    
    Controls how frequently the attention layers are cached during
    the diffusion process. Lower values cache more frequently, potentially
    resulting in more speedup but using more memory.
    
    This value must be greater than zero.
    
    Each value supplied will be tried in turn.
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.deep_cache` is enabled.
    
    (default: 5)
    """

    deep_cache_branch_ids: _types.OptionalIntegers = None
    """
    Branch IDs to try for DeepCache for the main model.
    
    Controls which branches of the UNet attention blocks the caching
    is applied to. Advanced usage only.
    
    This value must be greater than or equal to 0.
    
    Each value supplied will be tried in turn.
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.deep_cache` is enabled.
    
    (default: 1)
    """

    sdxl_refiner_deep_cache: _types.OptionalBoolean = None
    """
    Activate DeepCache for the SDXL Refiner?
    
    See: :py:attr:`RenderLoopConfig.deep_cache`
    
    This is supported for Stable Diffusion XL and Kolors based models.
    """

    sdxl_refiner_deep_cache_intervals: _types.OptionalIntegers = None
    """
    Cache intervals to try for DeepCache for the SDXL Refiner.
    
    Controls how frequently the attention layers are cached during
    the diffusion process. Lower values cache more frequently, potentially
    resulting in more speedup but using more memory.
    
    This value must be greater than zero.
    
    Each value supplied will be tried in turn.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.sdxl_refiner_deep_cache` is enabled.
    
    (default: 5)
    """

    sdxl_refiner_deep_cache_branch_ids: _types.OptionalIntegers = None
    """
    Branch IDs to try for DeepCache for the SDXL Refiner.
    
    Controls which branches of the UNet attention blocks the caching
    is applied to. Advanced usage only.
    
    This value must be greater than or equal to 0.
    
    Each value supplied will be tried in turn.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.sdxl_refiner_deep_cache` is enabled.
    
    (default: 1)
    """

    ras: bool = False
    """
    Activate RAS (Region-Adaptive Sampling) for the primary model? 
            
    This can increase inference speed with SD3.
    
    See: https://github.com/microsoft/ras
    
    This is supported for: ``--model-type sd3``.
    """

    ras_index_fusion: _types.OptionalBoolean = None
    """
    Enable index fusion in RAS (Region-Adaptive Sampling) for the primary model?
    
    This can improve attention computation in RAS for SD3.
    
    See: https://github.com/microsoft/ras
    
    Setting to ``True`` implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``, (but not for SD3.5 models)
    """

    ras_sample_ratios: _types.OptionalFloats = None
    """
    Sample ratios to try for RAS (Region-Adaptive Sampling).
    
    For instance, setting this to 0.5 on a sequence of 4096 tokens will result in the 
    noise of averagely 2048 tokens to be updated during each RAS step. Must be between 0 and 1.
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_high_ratios: _types.OptionalFloats = None
    """
    High ratios to try for RAS (Region-Adaptive Sampling).
    
    Based on the metric selected, the ratio of the high value chosen to be cached.
    Default value is 1.0, but can be set between 0 and 1 to balance the sample ratio 
    between the main subject and the background.
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_starvation_scales: _types.OptionalFloats = None
    """
    Starvation scales to try for RAS (Region-Adaptive Sampling).
    
    RAS tracks how often a token is dropped and incorporates this count as a scaling factor in the
    metric for selecting tokens. This scale factor prevents excessive blurring or noise in the 
    final generated image. Larger scaling factor will result in more uniform sampling.
    Usually set between 0.0 and 1.0.
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_metrics: _types.OptionalStrings = None
    """
    One or more RAS metrics to try.
    
    This controls how RAS measures the importance of tokens for caching.
    Valid values are "std" (standard deviation) or "l2norm" (L2 norm).
    Defaults to "std".
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_error_reset_steps: typing.Optional[collections.abc.Sequence[_types.Integers]] = None
    """
    Error reset step patterns to try for RAS (Region-Adaptive Sampling).
    
    The dense sampling steps inserted between the RAS steps to reset the accumulated error.
    Should be a list of lists of step numbers, e.g. [[12, 22], ...].
    
    Each list will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_start_steps: _types.OptionalIntegers = None
    """
    Starting steps to try for RAS (Region-Adaptive Sampling).
    
    This controls when RAS begins applying its sampling strategy.
    Must be greater than or equal to 1.
    Defaults to 4 if not specified.
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_end_steps: _types.OptionalIntegers = None
    """
    Ending steps to try for RAS (Region-Adaptive Sampling).
    
    This controls when RAS stops applying its sampling strategy.
    Must be greater than or equal to 1.
    Defaults to the number of inference steps if not specified.
    
    Each value will be tried in turn.
    
    Supplying any values implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_skip_num_steps: _types.OptionalIntegers = None
    """
    Skip steps to try for RAS (Region-Adaptive Sampling).
    
    This controls the number of steps to skip between RAS steps.
    
    The actual number of tokens skipped will be rounded down to the nearest multiple of 64.
    This ensures efficient memory access patterns for the attention computation.
    
    When used with :py:attr:`RenderLoopConfig.ras_skip_num_step_lengths` > 0, this value determines 
    how much to increase/decrease the number of skipped tokens over time. A positive value will 
    increase the number of skipped tokens, while a negative value will decrease it.
    
    Each value will be tried in turn.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    
    (default: 0)
    """

    ras_skip_num_step_lengths: _types.OptionalIntegers = None
    """
    Skip step lengths to try for RAS (Region-Adaptive Sampling).
    
    This controls the length of steps to skip between RAS steps.
    Must be greater than or equal to 0.
    
    When set to 0, static dropping is used where the same number of tokens are skipped
    at each step (except for error reset steps and steps before :py:attr:`RenderLoopConfig.ras_start_steps`).
    
    When greater than 0, dynamic dropping is used where the number of skipped tokens
    varies over time based on :py:attr:`RenderLoopConfig.ras_skip_num_steps`. The pattern repeats every 
    :py:attr:`RenderLoopConfig.ras_skip_num_step_lengths` steps.
    
    Each value will be tried in turn.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    
    (default: 0)
    """

    pag: bool = False
    """
    Use perturbed attention guidance?
    """

    pag_scales: _types.OptionalFloats = None
    """
    List of floating point perturbed attention guidance scales, this 
    corresponds to the ``--pag-scales`` argument of the dgenerate command line tool.
    """

    pag_adaptive_scales: _types.OptionalFloats = None
    """
    List of floating point adaptive perturbed attention guidance scales, this
    corresponds to the ``--pag-adaptive-scales`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_pag: _types.OptionalBoolean = None
    """
    Use perturbed attention guidance in the SDXL refiner?
    """

    sdxl_refiner_pag_scales: _types.OptionalFloats = None
    """
    List of floating point perturbed attention guidance scales to try with the SDXL refiner,
    this corresponds to the ``--sdxl-refiner-pag-scales`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_pag_adaptive_scales: _types.OptionalFloats = None
    """
    List of floating point adaptive perturbed attention guidance scales to try with the SDXL refiner, 
    this corresponds to the ``--sdxl-refiner-pag-adaptive-scales`` argument of the dgenerate command line tool.
    """

    second_model_scheduler_uri: _types.OptionalUriOrUris = None
    """
    Optional SDXL Refiner / Stable Cascade Decoder model scheduler/sampler class name specification, 
    this corresponds to the ``--second-model-scheduler`` argument of the dgenerate command line tool. 
    Setting this to 'help' will yield a help message to stdout describing scheduler names compatible 
    with the current configuration upon running. Passing 'helpargs' will yield a help message with a 
    list of overridable arguments for each scheduler and their typical defaults.
    
    This may be a list of schedulers, indicating to try each scheduler in turn.
    """

    safety_checker: bool = False
    """
    Enable safety checker? ``--safety-checker``
    """

    model_type: _pipelinewrapper.ModelType = _pipelinewrapper.ModelType.SD
    """
    Corresponds to the ``--model-type`` argument of the dgenerate command line tool.
    """

    device: _types.Name = _torchutil.default_device()
    """
    Processing device specification, for example "cuda" or "cuda:N" where N is an 
    alternate GPU id as reported by nvidia-smi if you want to specify a specific GPU.
    This corresponds to the ``--device`` argument of the dgenerate command line tool.

    The default device on MacOS is "mps".
    
    "xpu" is an option for intel GPUs, for which device indices are also supported.
    """

    dtype: _pipelinewrapper.DataType = _pipelinewrapper.DataType.AUTO
    """
    Primary model data type specification, IE: integer precision. Default is auto selection.
    Lower precision datatypes result in less GPU memory usage.
    This corresponds to the ``--dtype`` argument of the dgenerate command line tool.
    """

    revision: _types.Name = 'main'
    """
    Repo revision selector for the main model when loading from a huggingface repository.
    This corresponds to the ``--revision`` argument of the dgenerate command line tool.
    """

    variant: _types.OptionalName = None
    """
    Primary model weights variant string.
    This corresponds to the ``--variant`` argument of the dgenerate command line tool.
    """

    output_size: _types.OptionalSize = None
    """
    Desired output size, sizes not aligned by 8 pixels will result in an error message.
    This corresponds to the ``--output-size`` argument of the dgenerate command line tool.
    """

    no_aspect: bool = False
    """
    Should Seed, Mask, and Control guidance images specified in :py:attr:`.RenderLoopConfig.image_seeds`
    definitions (``--image-seeds``) have their aspect ratio ignored when being resized to 
    :py:attr:`.RenderLoopConfig.output_size` (``--output-size``) ?
    """

    output_path: _types.Path = 'output'
    """
    Render loop write folder, where images and animations will be written.
    This corresponds to the ``--output-path`` argument of the dgenerate command line tool.
    """

    output_prefix: _types.OptionalString = None
    """
    Output filename prefix, add an optional prefix string to all written files.
    This corresponds to the ``--output-prefix`` argument of the dgenerate command line tool.
    """

    output_overwrite: bool = False
    """
    Allow overwrites of files? or avoid this with a file suffix in a multiprocess safe manner?
    This corresponds to the ``--output-overwrite`` argument of the dgenerate command line tool.
    """

    output_configs: bool = False
    """
    Output a config text file next to every generated image or animation? this file will contain configuration
    that is pipeable to dgenerate stdin, which will reproduce said file.
    This corresponds to the ``--output-configs`` argument of the dgenerate command line tool.
    """

    output_metadata: bool = False
    """
    Write config text to the metadata of all written images? this data is not written to
    animated files, only PNGs and JPEGs. This corresponds to the ``--output-metadata``
    argument of the dgenerate command line tool.
    """

    output_auto1111_metadata: bool = False
    """
    Write Automatic1111 compatible metadata to the metadata of all written images?
    this data is not written to animated files, only PNGs and JPEGs.
    This corresponds to the ``--output-metadata`` argument of the dgenerate
    command line tool.
    """

    animation_format: _types.Name = 'mp4'
    """
    Format for any rendered animations, see: :py:func:`dgenerate.mediaoutput.supported_animation_writer_formats()`.
    This value may also be set to 'frames' to indicate that only individual frames should be output and no animation file coalesced.
    This corresponds to the ``--animation-format`` argument of the dgenerate command line tool.
    """

    image_format: _types.Name = 'png'
    """
    Format for any images that are written including animation frames.
    
    Anything other than "png", "jpg", or "jpeg" is not compatible with ``output_metadata=True`` 
    and a :py:exc:`.RenderLoopConfigError` will be raised upon running the render 
    loop if  ``output_metadata=True`` and this value is not one of those mentioned formats.
    """

    no_frames: bool = False
    """
    Should individual frames not be output when rendering an animation? defaults to False.
    This corresponds to the ``--no-frames`` argument of the dgenerate command line tool.
    Using this option when :py:attr:`RenderLoopConfig.animation_format` is equal to ``"frames"`` will
    cause a :py:exc:`RenderLoopConfigError` be raised.
    """

    frame_start: _types.Integer = 0
    """
    Start frame inclusive frame slice for any rendered animations.
    This corresponds to the ``--frame-start`` argument of the dgenerate command line tool.
    """

    frame_end: _types.OptionalInteger = None
    """
    Optional end frame inclusive frame slice for any rendered animations.
    This corresponds to the ``--frame-end`` argument of the dgenerate command line tool.
    """

    auth_token: _types.OptionalString = None
    """
    Optional huggingface API token which will allow the download of restricted repositories 
    that your huggingface account has been granted access to.
    This corresponds to the ``--auth-token`` argument of the dgenerate command line tool.
    """

    seed_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--seed-image-processors`` argument of the dgenerate command line tool verbatim.
    """

    mask_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--mask-image-processors`` argument of the dgenerate command line tool verbatim.
    """

    control_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--control-image-processors`` argument of the dgenerate command line tool verbatim,
    including the grouping syntax using the "+" symbol, the plus symbol should be used as its own list element,
    IE: it is a token.
    """

    post_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--post-processors`` argument of the dgenerate command line tool verbatim.
    """

    offline_mode: bool = False
    """
    Avoid ever connecting to the internet to download anything? this can be used if 
    all your models / media are cached or if you are only ever using resources that exist 
    on disk already. Corresponds to the ``--offline-mode`` argument of the dgenerate command line tool.
    """

    model_cpu_offload: bool = False
    """
    Force model cpu offloading for the main pipeline, this may reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be slower. Mutually exclusive with :py:attr:`RenderLoopConfig.model_sequential_offload`
    """

    model_sequential_offload: bool = False
    """
    Force sequential model offloading for the main pipeline, this may drastically reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be much slower. Mutually exclusive with :py:attr:`RenderLoopConfig.model_cpu_offload`
    """

    second_model_cpu_offload: _types.OptionalBoolean = None
    """
    Force model cpu offloading for the SDXL refiner or Stable Cascade decoder pipeline, 
    this may reduce memory consumption and allow large models to run when they would 
    otherwise not fit in your GPUs VRAM. Inference will be slower. Mutually exclusive 
    with :py:attr:`RenderLoopConfig.second_model_sequential_offload`
    """

    second_model_sequential_offload: _types.OptionalBoolean = None
    """
    Force sequential model offloading for the SDXL refiner or Stable Cascade decoder pipeline,
    this may drastically reduce memory consumption and allow large models to run when they 
    would otherwise not fit in your GPUs VRAM. Inference will be much slower. Mutually exclusive
    with :py:attr:`RenderLoopConfig.second_model_cpu_offload`
    """

    adetailer_class_filter: _types.OptionalIntegersAndStringsBag = None
    """
    A collection of class IDs and/or class names that indicates what YOLO detection classes to keep.
    This filter is applied before index-filter. Detections that don't match any of the 
    specified classes will be ignored. Integers are treated as ID's, strings are treated
    as names.
    """

    adetailer_index_filter: _types.OptionalIntegersBag = None
    """
    A list index values that indicates what YOLO detection indices to keep, 
    the index values start at zero. Detections are sorted by their top left bounding box 
    coordinate from left to right, top to bottom, by (confidence descending). The order of 
    detections in the image is identical to the reading order of words on a page (english). 
    Inpainting will only be performed on the specified detection indices, if no indices 
    are specified, then inpainting will be performed on all detections.
    This filter is applied after class-filter.
    """

    adetailer_detector_uris: _types.OptionalUris = None
    """
    One or more adetailer YOLO detector model URIs. Corresponds directly to --adetailer-detectors.
    
    Specification of this argument enables the adetailer inpainting algorithm and requires the
    use of :py:attr:`.RenderLoopConfig.image_seeds`
    """

    adetailer_model_masks: _types.OptionalBoolean = None
    """
    Indicates that masks generated by the model itself should be preferred over 
    masks generated from the detection bounding box. If this is ``True``, and the model itself
    returns mask data, :py:attr:`.RenderLoopConfig.adetailer_mask_shape`, 
    :py:attr:`.RenderLoopConfig.adetailer_mask_padding`, and :py:attr:`.RenderLoopConfig.adetailer_detector_padding` 
    will all be ignored.
    """

    adetailer_mask_shapes: _types.OptionalNames = None
    """
    One or more adetailer mask shapes to try.
    
    This indicates what mask shape adetailer should attempt to draw around a detected feature,
    the default value is "rectangle". You may also specify "circle" to generate an ellipsoid 
    shaped mask, which might be helpful for achieving better blending.
    
    Valid values are: ("r", "rect", "rectangle"), or ("c", "circle", "ellipse")
    """

    adetailer_detector_paddings: _types.OptionalPaddings = None
    """
    One or more adetailer detector padding values.
    
    This value specifies the amount of padding
    that will be added to the detection rectangle which is used to
    generate a masked area. The default is 0, you can make the mask
    area around the detected feature larger with positive padding
    and smaller with negative padding.
    
    Example:

    32 (32px Uniform, all sides)

    (10, 20) (10px Horizontal, 20px Vertical)

    (10, 20, 30, 40) (10px Left, 20px Top, 30px Right, 40px Bottom)

    Defaults to [0].
    """

    adetailer_mask_paddings: _types.OptionalPaddings = None
    """
    One or more adetailer mask padding values.
    
    This value indicates how much padding to place around the masked 
    area when cropping out the image to be inpainted, this value must be large
    enough to accommodate any feathering on the edge of the mask caused
    by :py:attr:`.RenderLoopConfig.adetailer_mask_blurs` or 
    :py:attr:`.RenderLoopConfig.adetailer_mask_dilations` for 
    the best result.

    Example:

    32 (32px Uniform, all sides)

    (10, 20) (10px Horizontal, 20px Vertical)

    (10, 20, 30, 40) (10px Left, 20px Top, 30px Right, 40px Bottom)

    Defaults to [32].
    """

    adetailer_mask_blurs: _types.OptionalIntegers = None
    """
    Indicates the level of gaussian blur to apply
    to the inpaint mask generated by adetailer, which can help with 
    smooth blending of the inpainted feature. Defaults to [4].
    """

    adetailer_mask_dilations: _types.OptionalIntegers = None
    """
    Indicates the amount of dilation applied to the generated adetailer inpaint mask, 
    see: cv2.dilate. Defaults to [4].
    """

    adetailer_sizes: _types.OptionalIntegers = None
    """
    One or more target sizes for processing detected areas.
    When specified, detected areas will always be scaled to this target size (with aspect ratio preserved)
    for processing, then scaled back to the original size for compositing.
    This can significantly improve detail quality for small detected features like faces or hands,
    or reduce processing time for overly large detected areas.
    The scaling is based on the larger dimension (width or height) of the detected area.
    The optimal resampling method is automatically selected for both upscaling and downscaling.
    Each value must be an integer greater than 1. Defaults to none (process at native resolution).
    """

    adetailer_crop_control_image: _types.OptionalBoolean = None
    """
    Should adetailer crop any control image the same way that it crops the mask?
    
    This is only relevant when using adetailer with ControlNet models.
    
    When enabled, control images will be cropped to match the detected region
    before being passed to the inpainting pipeline. This can help ensure that
    the control guidance is properly aligned with the area being inpainted.
    
    When disabled (default), control images will be resized to match the
    cropped region size without cropping.
    
    This corresponds to the ``--adetailer-crop-control-image`` argument of the dgenerate command line tool.
    """

    inpaint_crop: bool = False
    """
    Enable cropping to mask bounds for inpainting. When enabled, input images will be
    automatically cropped to the bounds of their masks (plus any padding) before processing, 
    then the generated result will be pasted back onto the original uncropped image. This 
    allows inpainting at higher effective resolutions for better quality results.
    
    Note: Inpaint crop cannot be used with multiple input images. Each image/mask pair must 
    be processed individually for optimal cropping, as different masks may have different bounds.
    However, ``batch_size`` > 1 is supported for generating multiple variations of a single crop.
    
    This corresponds to the ``--inpaint-crop`` argument of the dgenerate command line tool.
    """

    inpaint_crop_paddings: _types.OptionalPaddings = None
    """
    One or more padding values to use around mask bounds for inpaint cropping. Each value will be 
    tried in turn (combinatorial). Specifying this automatically enables :py:attr:`RenderLoopConfig.inpaint_crop`.
    
    Padding can be specified as:
    - A single integer (e.g., 32) for uniform padding on all sides
    - "WIDTHxHEIGHT" format (e.g., "10x20") for horizontal and vertical padding  
    - "LEFTxTOPxRIGHTxBOTTOM" format (e.g., "5x10x5x15") for specific side padding
    
    This corresponds to the ``--inpaint-crop-paddings`` argument of the dgenerate command line tool.
    """

    inpaint_crop_masked: bool = False
    """
    Use the mask when pasting the generated result back onto the original image for inpaint 
    cropping. Specifying this automatically enables :py:attr:`RenderLoopConfig.inpaint_crop`. 
    This means only the masked areas will be replaced. Cannot be used together with 
    :py:attr:`RenderLoopConfig.inpaint_crop_feathers`.
    
    This corresponds to the ``--inpaint-crop-masked`` argument of the dgenerate command line tool.
    """

    inpaint_crop_feathers: _types.OptionalIntegers = None
    """
    One or more feather values to use when pasting the generated result back onto the 
    original image for inpaint cropping. Specifying this automatically enables 
    :py:attr:`RenderLoopConfig.inpaint_crop`. Each value will be tried in turn (combinatorial). 
    Feathering creates smooth transitions from opaque to transparent. Cannot be used together 
    with :py:attr:`RenderLoopConfig.inpaint_crop_masked`.
    
    This corresponds to the ``--inpaint-crop-feathers`` argument of the dgenerate command line tool.
    """

    latents: _types.OptionalTensors = None
    """
    Optional list of tensors containing noisy latents to use as starting points for diffusion.
    These latents can be generated by using --denoising-end with --image-format pt/pth/safetensors to save 
    intermediate noisy latents from a previous generation. This allows for advanced workflows where you can 
    pass partially denoised latents between different models or generation stages.
    """

    latents_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing raw input latents before pipeline execution.
    
    These processors are applied to latents provided through the :py:attr:`RenderLoopConfig.latents` 
    argument (raw latents used as noise initialization). The processors are applied in sequence
    before the latents are passed to the diffusion pipeline.
    
    This corresponds to the ``--latents-processors`` argument of the dgenerate command line tool.
    """

    latents_post_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing output latents when outputting to latents.
    
    These processors are applied to latents when :py:attr:`RenderLoopConfig.image_format` is set to a 
    tensor format (pt, pth, safetensors). The processors are applied in sequence after the diffusion 
    pipeline generates the latents but before they are returned in the result.
    
    This corresponds to the ``--latents-post-processors`` argument of the dgenerate command line tool.
    """

    img2img_latents_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing img2img latents before pipeline execution.
    
    These processors are applied to latent tensors provided through the :py:attr:`RenderLoopConfig.image_seeds` 
    argument when doing img2img with tensor inputs. The processors are applied in sequence and may occur 
    before VAE decoding (for models that decode img2img latents) or before direct pipeline usage.
    
    This corresponds to the ``--img2img-latents-processors`` argument of the dgenerate command line tool.
    """

    denoising_start: _types.OptionalFloat = None
    """
    Denoising should start at this fraction of total timesteps (0.0 to 1.0).
    
    This is useful continuing denoising on noisy latents generated with :py:attr:`RenderLoopConfig.denoising_end`
    
    Scheduler Compatibility:
    
    * SD 1.5 models: Only stateless schedulers are supported (``EulerDiscreteScheduler``, 
      ``LMSDiscreteScheduler``, ``EDMEulerScheduler``, ``DPMSolverMultistepScheduler``, 
      ``DDIMScheduler``, ``DDPMScheduler``, ``PNDMScheduler``)
    * SDXL models: All schedulers supported via native denoising_start/denoising_end
    * SD3/Flux models: FlowMatchEulerDiscreteScheduler and standard schedulers supported
    
    This corresponds to the ``--denoising-start`` argument of the dgenerate command line tool.
    """

    denoising_end: _types.OptionalFloat = None
    """
    Denoising should end at this fraction of total timesteps (0.0 to 1.0).
    
    This is useful for generating noisy latents that can be saved and passed to other models.
    
    Scheduler Compatibility:
    
    * SD 1.5 models: Only stateless schedulers are supported (``EulerDiscreteScheduler``, 
      ``LMSDiscreteScheduler``, ``EDMEulerScheduler``, ``DPMSolverMultistepScheduler``, 
      ``DDIMScheduler``, ``DDPMScheduler``, ``PNDMScheduler``)
    * SDXL models: All schedulers supported via native denoising_start/denoising_end
    * SD3/Flux models: FlowMatchEulerDiscreteScheduler and standard schedulers supported
    
    This corresponds to the ``--denoising-end`` argument of the dgenerate command line tool.
    """

    def __init__(self):
        self.guidance_scales = [_pipelinewrapper.constants.DEFAULT_GUIDANCE_SCALE]
        self.inference_steps = [_pipelinewrapper.constants.DEFAULT_INFERENCE_STEPS]
        self.prompts = [_prompt.Prompt()]
        self.seeds = gen_seeds(1)

    def _check(self, attribute_namer: typing.Optional[typing.Callable[[str], str]] = None):
        """
        Check the configuration for type and logical usage errors, set
        defaults for certain values when needed and not specified.

        This may modify the configuration.

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """
        # Create attribute namer function for consistent error reporting
        a_namer = self._create_attribute_namer(attribute_namer)

        # Validate basic type constraints
        self._validate_type_constraints(a_namer)

        # Setup help mode and validate help-related arguments
        help_mode = self._check_help_arguments(a_namer)

        # Check model-specific optimization features
        self._check_optimization_features(a_namer)

        # Check second model arguments compatibility
        self._check_second_model_compatibility(a_namer)

        # Check configuration files compatibility
        self._check_configuration_files_compatibility(a_namer)

        # Check adetailer compatibility
        self._check_adetailer_compatibility(a_namer)

        # Check inpaint crop compatibility
        self._check_inpaint_crop_compatibility(a_namer)

        # Check image seeds requirements for certain model types
        self._check_image_seeds_requirements(a_namer, help_mode)

        # Check output-related arguments
        self._check_output_arguments(a_namer)

        # Configure PAG defaults if needed
        self._configure_pag_defaults()

        # Check prompt weighters
        self._check_prompt_weighters(a_namer)

        # Check other general compatibility issues
        self._check_general_compatibility(a_namer)

        # Check model-specific requirements and restrictions
        self._check_model_specific_requirements(a_namer)

        # Process image seeds if present
        self._process_image_seeds(a_namer, help_mode)

    def _create_attribute_namer(self, attribute_namer: typing.Optional[typing.Callable[[str], str]]) -> typing.Callable[
        [str], str]:
        """Create a function to standardize attribute naming in error messages."""

        def a_namer(attr_name: str) -> str:
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'{self.__class__.__name__}.{attr_name}'

        return a_namer

    def _validate_type_constraints(self, a_namer: typing.Callable[[str], str]):
        """Validate type constraints on configuration attributes."""
        try:
            _types.type_check_struct(self, a_namer)
        except ValueError as e:
            raise RenderLoopConfigError(e) from e

    def _check_help_arguments(self, a_namer: typing.Callable[[str], str]) -> bool:
        """Check help-related arguments for validity."""
        schedulers = [self.scheduler_uri] if \
            isinstance(self.scheduler_uri, (_types.Uri, type(None))) else self.scheduler_uri
        scheduler_help = any(_pipelinewrapper.scheduler_is_help(s) for s in schedulers)

        second_model_schedulers = [self.second_model_scheduler_uri] if \
            isinstance(self.second_model_scheduler_uri, (_types.Uri, type(None))) else self.second_model_scheduler_uri
        second_model_scheduler_help = any(_pipelinewrapper.scheduler_is_help(s) for s in second_model_schedulers)

        text_encoder_help = _pipelinewrapper.text_encoder_is_help(self.text_encoder_uris)
        second_model_text_encoder_help = _pipelinewrapper.text_encoder_is_help(self.second_model_text_encoder_uris)

        help_mode = scheduler_help or \
                    second_model_scheduler_help or \
                    text_encoder_help or \
                    second_model_text_encoder_help

        if scheduler_help and len(schedulers) > 1:
            raise RenderLoopConfigError(
                f'You cannot specify "help" or "helpargs" to {a_namer("scheduler")} '
                f'with multiple values involved.'
            )

        if second_model_scheduler_help and len(second_model_schedulers) > 1:
            raise RenderLoopConfigError(
                f'You cannot specify "help" or "helpargs" to {a_namer("second_model_scheduler_uri")} '
                f'with multiple values involved.'
            )

        if text_encoder_help and len(self.text_encoder_uris) > 1:
            raise RenderLoopConfigError(
                f'You cannot specify "help" or "helpargs" to {a_namer("text_encoder_uris")} '
                f'with multiple values involved.'
            )

        if second_model_text_encoder_help and len(self.second_model_text_encoder_uris) > 1:
            raise RenderLoopConfigError(
                f'You cannot specify "help" or "helpargs" to {a_namer("second_model_text_encoder_uris")} '
                f'with multiple values involved.'
            )

        return help_mode

    def _check_optimization_features(self, a_namer: typing.Callable[[str], str]):
        """Check optimization features compatibility with the selected model type."""

        if self.quantizer_map and not self.quantizer_uri:
            raise RenderLoopConfigError(
                f'{a_namer("quantizer_map")} cannot be used without {a_namer("quantizer_uri")}'
            )

        if self.second_model_quantizer_map and not self.second_model_quantizer_uri:
            raise RenderLoopConfigError(
                f'{a_namer("second_model_quantizer_map")} cannot be '
                f'used without {a_namer("second_model_quantizer_uri")}'
            )

        # Check TeaCache compatibility
        tea_cache_enabled = (self.tea_cache or any(self._non_null_attr_that_start_with('tea_cache_')))
        if tea_cache_enabled and not _pipelinewrapper.model_type_is_flux(self.model_type):
            raise RenderLoopConfigError(
                f'{a_namer("tea_cache")} and related arguments are only '
                f'compatible with {a_namer("model_type")} flux*'
            )
        if tea_cache_enabled and self.model_cpu_offload:
            raise RenderLoopConfigError(
                f'{a_namer("model_cpu_offload")} is not compatible '
                f'with {a_namer("tea_cache")} and related arguments.'
            )

        # Check RAS compatibility
        ras_enabled = (self.ras or any(self._non_null_attr_that_start_with('ras_')))
        if ras_enabled and not _pipelinewrapper.model_type_is_sd3(self.model_type):
            raise RenderLoopConfigError(
                f'{a_namer("ras")} and related arguments are only '
                f'compatible with {a_namer("model_type")} sd3'
            )
        if ras_enabled and self.model_cpu_offload:
            raise RenderLoopConfigError(
                f'{a_namer("model_cpu_offload")} is not compatible '
                f'with {a_namer("ras")} and related arguments.'
            )
        if self.ras_index_fusion and self.model_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("ras_index_fusion")} is not compatible '
                f'with {a_namer("model_sequential_offload")}.'
            )

        if self.ras_index_fusion and (
                self.quantizer_uri or (self.unet_uri and _pipelinewrapper.UNetUri.parse(self.unet_uri).quantizer)
        ):
            raise RenderLoopConfigError(
                f'{a_namer("ras_index_fusion")} is not supported for RAS when UNet quantization is enabled, '
                f'quantize the text encoders individually using {a_namer("text_encoder_uris")}')

        if self.hi_diffusion:
            if not (
                    self.model_type == _pipelinewrapper.ModelType.SDXL or
                    self.model_type == _pipelinewrapper.ModelType.KOLORS or
                    self.model_type == _pipelinewrapper.ModelType.SD):
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion")} is only supported for '
                    f'Stable Diffusion, Stable Diffusion XL, and Kolors'
                )

            if self.t2i_adapter_uris:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion")} is not supported with {a_namer("t2i_adapter_uris")}'
                )
        else:
            if self.hi_diffusion_no_win_attn is not None:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion_no_win_attn")} is only supported when {a_namer("hi_diffusion")} is enabled.'
                )
            if self.hi_diffusion_no_raunet is not None:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion_no_raunet")} is only supported when {a_namer("hi_diffusion")} is enabled.'
                )

        # Check DeepCache compatibility first (needed for SADA validation)
        deep_cache_enabled = (self.deep_cache or any(self._non_null_attr_that_start_with('deep_cache_')))

        # Check SADA compatibility
        sada_enabled = (self.sada or any(self._non_null_attr_that_start_with('sada_')))
        if sada_enabled and not (
                self.model_type == _pipelinewrapper.ModelType.SD or
                self.model_type == _pipelinewrapper.ModelType.SDXL or
                self.model_type == _pipelinewrapper.ModelType.KOLORS or
                _pipelinewrapper.model_type_is_flux(self.model_type)):
            raise RenderLoopConfigError(
                f'SADA arguments are only supported for '
                f'--model-type sd, sdxl, kolors, and flux*'
            )

        if sada_enabled and tea_cache_enabled:
            raise RenderLoopConfigError(
                f'SADA cannot be used simultaneously with {a_namer("tea_cache")} and related arguments.'
            )

        if sada_enabled and deep_cache_enabled:
            raise RenderLoopConfigError(
                f'SADA cannot be used simultaneously with {a_namer("deep_cache")} and related arguments.'
            )

        if sada_enabled and self.hi_diffusion:
            raise RenderLoopConfigError(
                f'SADA cannot be used simultaneously with {a_namer("hi_diffusion")}'
            )

        # Validate SADA Lagrangian interpolation parameters
        if sada_enabled and self.sada_lagrange_terms and any(term != 0 for term in self.sada_lagrange_terms):
            if not self.sada_lagrange_ints or not self.sada_lagrange_steps:
                raise RenderLoopConfigError(
                    f'When using SADA Lagrangian interpolation ({a_namer("sada_lagrange_terms")} != 0), '
                    f'both {a_namer("sada_lagrange_ints")} and {a_namer("sada_lagrange_steps")} must be specified'
                )

            # Check compatibility for each combination
            for lagrange_int in self.sada_lagrange_ints or []:
                for lagrange_step in self.sada_lagrange_steps or []:
                    if lagrange_step % lagrange_int != 0:
                        raise RenderLoopConfigError(
                            f'SADA {a_namer("sada_lagrange_steps")} ({lagrange_step}) must be '
                            f'divisible by {a_namer("sada_lagrange_ints")} ({lagrange_int})'
                        )

        # Set up SADA defaults when any SADA argument is used
        if sada_enabled:
            # Get model-specific defaults
            sada_defaults = _pipelinewrapper_util.get_sada_model_defaults(self.model_type)

            if self.sada_max_downsamples is None:
                self.sada_max_downsamples = [sada_defaults['max_downsample']]
            if self.sada_sxs is None:
                self.sada_sxs = [sada_defaults['sx']]
            if self.sada_sys is None:
                self.sada_sys = [sada_defaults['sy']]
            if self.sada_acc_ranges is None:
                self.sada_acc_ranges = [sada_defaults['acc_range']]
            if self.sada_lagrange_terms is None:
                self.sada_lagrange_terms = [sada_defaults['lagrange_term']]
            if self.sada_lagrange_ints is None:
                self.sada_lagrange_ints = [sada_defaults['lagrange_int']]
            if self.sada_lagrange_steps is None:
                self.sada_lagrange_steps = [sada_defaults['lagrange_step']]
            if self.sada_max_fixes is None:
                self.sada_max_fixes = [sada_defaults['max_fix']]
            if self.sada_max_intervals is None:
                self.sada_max_intervals = [sada_defaults['max_interval']]

        if deep_cache_enabled and not (
                self.model_type == _pipelinewrapper.ModelType.SDXL or
                self.model_type == _pipelinewrapper.ModelType.SDXL_PIX2PIX or
                self.model_type == _pipelinewrapper.ModelType.KOLORS or
                self.model_type == _pipelinewrapper.ModelType.SD or
                self.model_type == _pipelinewrapper.ModelType.PIX2PIX or
                self.model_type == _pipelinewrapper.ModelType.UPSCALER_X4):
            raise RenderLoopConfigError(
                f'{a_namer("deep_cache")} and related arguments are only '
                f'supported with Stable Diffusion, Stable Diffusion XL, '
                f'Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.'
            )

        # Check FreeU compatibility
        if self.freeu_params is not None:
            freeu_model_types = {
                _pipelinewrapper.ModelType.SD,
                _pipelinewrapper.ModelType.SDXL,
                _pipelinewrapper.ModelType.KOLORS,
                _pipelinewrapper.ModelType.PIX2PIX,
                _pipelinewrapper.ModelType.SDXL_PIX2PIX,
                _pipelinewrapper.ModelType.UPSCALER_X2,
                _pipelinewrapper.ModelType.UPSCALER_X4
            }

            if self.model_type not in freeu_model_types:
                raise RenderLoopConfigError(
                    f'{a_namer("freeu_params")} not supported with '
                    f'{a_namer("model_type")} {_pipelinewrapper.get_model_type_string(self.model_type)}.'
                )

    def _check_second_model_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check compatibility of second model arguments with the primary model type."""
        # Check if second model arguments are valid for the primary model type
        if not (_pipelinewrapper.model_type_is_sdxl(self.model_type) or
                _pipelinewrapper.model_type_is_kolors(self.model_type) or
                _pipelinewrapper.model_type_is_s_cascade(self.model_type)):
            errors = []
            for arg in self._non_null_second_model_arguments():
                errors.append(
                    f'Cannot use {a_namer(arg)} with '
                    f'{a_namer("model_type")} '
                    f'{_pipelinewrapper.get_model_type_string(self.model_type)}'
                )
            if errors:
                raise RenderLoopConfigError('\n'.join(errors))

        # Check if second model scheduler and text encoder are valid
        if not self.sdxl_refiner_uri and not self.s_cascade_decoder_uri:
            if self.second_model_scheduler_uri:
                raise RenderLoopConfigError(
                    f'Cannot use {a_namer("second_model_scheduler_uri")} if {a_namer("sdxl_refiner_uri")} '
                    f'or {a_namer("s_cascade_decoder_uri")} is not specified.')
            if self.second_model_text_encoder_uris:
                raise RenderLoopConfigError(
                    f'Cannot use {a_namer("second_model_text_encoder_uris")} if {a_namer("sdxl_refiner_uri")} '
                    f'or {a_namer("s_cascade_decoder_uri")} is not specified.')

    def _check_configuration_files_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check compatibility of configuration files with model types."""
        # Check original config compatibility
        if not _hfhub.is_single_file_model_load(self.model_path):
            if self.original_config:
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("original_config")} when the main '
                    f'model is not a a single file checkpoint.'
                )

        # Check second model original config compatibility
        if self.second_model_original_config:
            if not self.sdxl_refiner_uri and not self.s_cascade_decoder_uri:
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("second_model_original_config")} '
                    f'without {a_namer("sdxl_refiner_uri")} or {a_namer("s_cascade_decoder_uri")}.'
                )

            if self.sdxl_refiner_uri and \
                    not _hfhub.is_single_file_model_load(
                        _pipelinewrapper.uris.SDXLRefinerUri.parse(self.sdxl_refiner_uri).model):
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("second_model_original_config")} '
                    f'when the {a_namer("sdxl_refiner_uri")} model is not a '
                    f'single file checkpoint.'
                )
            if self.s_cascade_decoder_uri and \
                    not _hfhub.is_single_file_model_load(
                        _pipelinewrapper.uris.SCascadeDecoderUri.parse(self.s_cascade_decoder_uri).model):
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("second_model_original_config")} '
                    f'when the {a_namer("s_cascade_decoder_uri")} model is not a '
                    f'single file checkpoint.'
                )

    def _check_adetailer_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check compatibility of ADetailer-related arguments."""
        adetailer_args_used = list(self._non_null_attr_that_start_with('adetailer'))

        if not self.adetailer_detector_uris and adetailer_args_used:
            bad_adetailer_args = _textprocessing.oxford_comma(
                [a_namer(a) for a in adetailer_args_used if a != "adetailer_detector_uris"], "or")
            raise RenderLoopConfigError(
                f'May not use {bad_adetailer_args} without {a_namer("adetailer_detector_uris")}.')

        if self.adetailer_detector_uris and self.model_type not in {
            _pipelinewrapper.ModelType.SD,
            _pipelinewrapper.ModelType.SDXL,
            _pipelinewrapper.ModelType.KOLORS,
            _pipelinewrapper.ModelType.SD3,
            _pipelinewrapper.ModelType.FLUX,
            _pipelinewrapper.ModelType.FLUX_FILL
        }:
            raise RenderLoopConfigError(
                f'{a_namer("adetailer_detector_uris")} is only compatible with '
                f'{a_namer("model_type")} sd, sdxl, kolors, sd3, and flux')

        if self.adetailer_detector_uris and self.is_output_latents():
            raise RenderLoopConfigError(
                f'Outputting latents with {a_namer("image_format")} {self.image_format} '
                f'is not supported with {a_namer("adetailer_detector_uris")}'
            )

        if self.adetailer_mask_shapes:
            for shape in self.adetailer_mask_shapes:
                try:
                    parsed_shape = _textprocessing.parse_basic_mask_shape(shape)
                except ValueError:
                    parsed_shape = None

                if parsed_shape is None or parsed_shape not in {
                    _textprocessing.BasicMaskShape.RECTANGLE,
                    _textprocessing.BasicMaskShape.ELLIPSE
                }:
                    raise RenderLoopConfigError(
                        f'Unknown {"adetailer_mask_shapes"} value: {shape}')

    def _check_inpaint_crop_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check compatibility of inpaint crop arguments."""
        # Automatically enable inpaint crop if padding, feathering, or masking is specified
        if not self.inpaint_crop and (
                self.inpaint_crop_paddings or self.inpaint_crop_feathers or self.inpaint_crop_masked):
            self.inpaint_crop = True

        if self.inpaint_crop and self.no_aspect:
            raise RenderLoopConfigError(
                f'{a_namer("inpaint_crop")} is not compatible with {a_namer("no_aspect")}'
            )

        # Check if inpaint crop is used without mask inputs
        if self.inpaint_crop and not self.image_seeds:
            raise RenderLoopConfigError(
                f'{a_namer("inpaint_crop")} requires {a_namer("image_seeds")} to be specified '
                f'with mask images for inpainting.')

        # Check mutual exclusivity of masked and feathered modes
        if self.inpaint_crop_masked and self.inpaint_crop_feathers:
            raise RenderLoopConfigError(
                f'{a_namer("inpaint_crop_masked")} and {a_namer("inpaint_crop_feathers")} '
                f'are mutually exclusive options.')

        # Check compatibility with latent output
        if self.inpaint_crop and self.is_output_latents():
            raise RenderLoopConfigError(
                f'Outputting latents with {a_namer("image_format")} {self.image_format} '
                f'is not supported with {a_namer("inpaint_crop")}')

        # Set default padding when inpaint crop is enabled but no padding/feathering specified
        if self.inpaint_crop and not self.inpaint_crop_paddings and not self.inpaint_crop_feathers:
            self.inpaint_crop_paddings = [_pipelinewrapper.constants.DEFAULT_INPAINT_CROP_PADDING]

    def _check_image_seeds_requirements(self, a_namer: typing.Callable[[str], str], help_mode: bool):
        """Verify requirements for image seeds based on model type."""
        if not self.image_seeds:
            args_help = help_mode

            # Check if model type requires image seeds
            if _pipelinewrapper.model_type_is_floyd_ifs(self.model_type) and not args_help:
                raise RenderLoopConfigError(
                    f'you cannot specify Deep Floyd IF super-resolution '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}'
                )

            if _pipelinewrapper.model_type_is_upscaler(self.model_type) and not args_help:
                raise RenderLoopConfigError(
                    f'you cannot specify an upscaler model '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}.'
                )

            if _pipelinewrapper.model_type_is_pix2pix(self.model_type) and not args_help:
                raise RenderLoopConfigError(
                    f'you cannot specify a pix2pix model '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}.'
                )

            if self.model_type == _pipelinewrapper.ModelType.FLUX_FILL:
                raise RenderLoopConfigError(
                    f'you cannot use {a_namer("model_type")} '
                    f'flux-fill without {a_namer("image_seeds")}.'
                )

            # Check arguments that require image seeds
            if self.adetailer_detector_uris:
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("adetailer_detector_uris")} '
                    f'without {a_namer("image_seeds")}.'
                )

            if self.image_seed_strengths:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("image_seed_strengths")} '
                    f'without {a_namer("image_seeds")}.')

            if self.seeds_to_images:
                raise RenderLoopConfigError(
                    f'{a_namer("seeds_to_images")} cannot be specified '
                    f'without {a_namer("image_seeds")}.')

            if self.controlnet_uris:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("controlnet_uris")} '
                    f'without {a_namer("image_seeds")}.')

            if self.t2i_adapter_uris:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("t2i_adapter_uris")} '
                    f'without {a_namer("image_seeds")}.')

            if self.ip_adapter_uris:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("ip_adapter_uris")} '
                    f'without {a_namer("image_seeds")}.')

            # Check image processor arguments without image seeds
            invalid_self = []
            for processor_self in self._non_null_attr_that_end_with('image_processors'):
                invalid_self.append(
                    f'you cannot specify {a_namer(processor_self)} '
                    f'without {a_namer("image_seeds")}.')

            # check for latents processors without any possible latents input
            for processor_self in self._non_null_attr_that_end_with('latents_processors'):
                invalid_self.append(
                    f'you cannot specify {a_namer(processor_self)} '
                    f'without {a_namer("image_seeds")}.')

            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

            # Check pipeline class compatibility
            try:
                _pipelinewrapper.get_pipeline_class(
                    model_type=self.model_type,
                    pipeline_type=_pipelinewrapper.PipelineType.TXT2IMG,
                    unet_uri=self.unet_uri,
                    transformer_uri=self.transformer_uri,
                    vae_uri=self.vae_uri,
                    lora_uris=self.lora_uris,
                    image_encoder_uri=self.image_encoder_uri,
                    ip_adapter_uris=self.ip_adapter_uris,
                    textual_inversion_uris=self.textual_inversion_uris,
                    controlnet_uris=self.controlnet_uris,
                    t2i_adapter_uris=self.t2i_adapter_uris,
                    pag=self.pag,
                    help_mode=help_mode
                )
            except _pipelinewrapper.UnsupportedPipelineConfigError as e:
                raise RenderLoopConfigError(str(e)) from e

    def _check_output_arguments(self, a_namer: typing.Callable[[str], str]):
        """Check output-related arguments for compatibility."""
        # Check output prefix
        if self.output_prefix:
            if '/' in self.output_prefix or '\\' in self.output_prefix:
                raise RenderLoopConfigError(
                    f'{a_namer("output_prefix")} value may not contain slash characters.')

        # Check frame start/end
        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise RenderLoopConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        # Clean and validate animation and image formats
        self.animation_format = self.animation_format.strip().lower()
        self.image_format = self.image_format.strip().lower()

        if self.animation_format not in _mediaoutput.get_supported_animation_writer_formats() + ['frames']:
            raise RenderLoopConfigError(
                f'Unsupported {a_namer("animation_format")} value "{self.animation_format}". Must be one of '
                f'{_textprocessing.oxford_comma(_mediaoutput.get_supported_animation_writer_formats(), "or")}')

        # Check if it's a supported image format or tensor format
        supported_image_formats = _mediaoutput.get_supported_static_image_formats()
        supported_tensor_formats = _mediaoutput.get_supported_tensor_formats()
        all_supported_formats = supported_image_formats + supported_tensor_formats

        if self.image_format not in all_supported_formats:
            raise RenderLoopConfigError(
                f'Unsupported {a_namer("image_format")} value "{self.image_format}". Must be one of '
                f'{_textprocessing.oxford_comma(all_supported_formats, "or")}')

        if self.output_metadata and self.output_auto1111_metadata:
            raise RenderLoopConfigError(
                f'{a_namer("output_metadata")} and {a_namer("output_auto1111_metadata")} '
                f'are mutually exclusive and cannot be used simultaneously.')

        # Only check metadata compatibility for actual image formats, not tensor formats
        if not self.is_output_latents():
            if self.latents_post_processors:
                raise RenderLoopConfigError(
                    f'Cannot specify {a_namer("latents_post_processors")} when {a_namer("image_format")} is not a '
                    f'tensor format such as: '
                    f'{_textprocessing.oxford_comma(_mediainput.get_supported_tensor_formats(), "or")}'
                )

            if self.image_format not in {"png", "jpg", "jpeg"}:
                if self.output_metadata or self.output_auto1111_metadata:
                    prop_name = 'output_metadata' if self.output_metadata else 'output_auto1111_metadata'
                    raise RenderLoopConfigError(
                        f'{a_namer("image_format")} value "{self.image_format}" is '
                        f'unsupported when {a_namer(prop_name)} is enabled. '
                        f'Only "png", "jpg", and "jpeg" formats are supported with {a_namer(prop_name)}.')

        # Tensor formats don't support metadata
        if self.is_output_latents() and (self.output_metadata or self.output_auto1111_metadata):
            prop_name = 'output_metadata' if self.output_metadata else 'output_auto1111_metadata'
            raise RenderLoopConfigError(
                f'{a_namer(prop_name)} is not supported when outputting latents. '
                f'Tensor formats ({", ".join(_mediaoutput.get_supported_tensor_formats())}) do not support metadata.')

        if self.animation_format == 'frames' and self.no_frames:
            raise RenderLoopConfigError(
                f'Cannot specify {a_namer("no_frames")} when {a_namer("animation_format")} is set to "frames"')

    def _configure_pag_defaults(self):
        """Configure PAG default values if needed."""
        if self.pag:
            if not (self.pag_scales or self.pag_adaptive_scales):
                self.pag_scales = [_pipelinewrapper.constants.DEFAULT_PAG_SCALE]
                self.pag_adaptive_scales = [_pipelinewrapper.constants.DEFAULT_PAG_ADAPTIVE_SCALE]

        if self.sdxl_refiner_pag and self.sdxl_refiner_uri:
            if not (self.sdxl_refiner_pag_scales or self.sdxl_refiner_pag_adaptive_scales):
                self.sdxl_refiner_pag_scales = [_pipelinewrapper.constants.DEFAULT_SDXL_REFINER_PAG_SCALE]
                self.sdxl_refiner_pag_adaptive_scales = [
                    _pipelinewrapper.constants.DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE]

    def _check_prompt_weighters(self, a_namer: typing.Callable[[str], str]):
        """Check prompt weighter compatibility."""
        if self.prompt_weighter_uri is not None \
                and not _promptweighters.prompt_weighter_exists(self.prompt_weighter_uri):
            raise RenderLoopConfigError(
                f'Unknown prompt weighter implementation: {_promptweighters.prompt_weighter_name_from_uri(self.prompt_weighter_uri)}, '
                f'must be one of: {_textprocessing.oxford_comma(_promptweighters.prompt_weighter_names(), "or")}')

        if self.second_model_prompt_weighter_uri is not None \
                and not _promptweighters.prompt_weighter_exists(self.second_model_prompt_weighter_uri):
            raise RenderLoopConfigError(
                f'Unknown prompt weighter implementation for secondary model: '
                f'{_promptweighters.prompt_weighter_name_from_uri(self.prompt_weighter_uri)}, '
                f'must be one of: {_textprocessing.oxford_comma(_promptweighters.prompt_weighter_names(), "or")}')

    def _check_general_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check general configuration compatibility."""
        # Check T2I adapter factors
        if self.sdxl_t2i_adapter_factors and not self.t2i_adapter_uris:
            raise RenderLoopConfigError(
                f'You may not specify {a_namer("sdxl_t2i_adapter_factors")} '
                f'without {a_namer("t2i_adapter_uris")}.')

        # Check data type
        supported_dtypes = _pipelinewrapper.supported_data_type_strings()
        if self.dtype not in _pipelinewrapper.supported_data_type_enums():
            raise RenderLoopConfigError(
                f'{a_namer("dtype")} must be {_textprocessing.oxford_comma(supported_dtypes, "or")}')

        # Check batch size
        if self.batch_size is not None and self.batch_size < 1:
            raise RenderLoopConfigError(
                f'{a_namer("batch_size")} must be greater than or equal to 1.')

        # Check model type
        if self.model_type not in _pipelinewrapper.supported_model_type_enums():
            supported_model_types = _textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), "or")
            raise RenderLoopConfigError(
                f'{a_namer("model_type")} must be one of: {supported_model_types}')

        # Check device
        if not _torchutil.is_valid_device_string(self.device):
            raise RenderLoopConfigError(
                f'{a_namer("device")} {_torchutil.invalid_device_message(self.device, cap=False)}')

        # Check model offload options
        if self.model_cpu_offload and self.model_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("model_cpu_offload")} and {a_namer("model_sequential_offload")} '
                f'may not be enabled simultaneously.')

        if self.second_model_cpu_offload and self.second_model_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("second_model_cpu_offload")} and {a_namer("second_model_sequential_offload")} '
                f'may not be enabled simultaneously.')

        # Check model path is specified
        if self.model_path is None:
            raise RenderLoopConfigError(
                f'{a_namer("model_path")} must be specified')

        # Check clip skip compatibility
        if self.clip_skips:
            if (
                    self.model_type != _pipelinewrapper.ModelType.SD and
                    self.model_type != _pipelinewrapper.ModelType.SDXL and
                    self.model_type != _pipelinewrapper.ModelType.SD3 and
                    self.model_type != _pipelinewrapper.ModelType.SD3_PIX2PIX and
                    self.prompt_weighter_uri is None
            ):
                # prompt weighter may be able to handle clip skips
                # when the pipeline cannot, such is the case for StableCascade
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("clip_skips")} for '
                    f'{a_namer("model_type")} values other than '
                    f'"sd", "sdxl", or "sd3" when a '
                    f'{a_namer("prompt_weighter_uri")} is not specified.')

        # Check tensor output compatibility with batch grid
        if self.batch_grid_size is not None and self.is_output_latents():
            raise RenderLoopConfigError(
                f'{a_namer("batch_grid_size")} cannot be used with latents output formats '
                f'(pt, pth, safetensors). Image grids can only be created from decoded images, '
                f'not raw latents tensors. Use a standard image format such as "png" or "jpg" instead.')

    def _check_model_specific_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check specific requirements for different model types."""
        self._check_stable_cascade_requirements(a_namer)
        self._check_upscaler_requirements(a_namer)
        self._check_pix2pix_requirements(a_namer)
        self._check_transformer_compatibility(a_namer)
        self._check_flux_model_requirements(a_namer)
        self._check_sd3_model_requirements(a_namer)
        self._check_sdxl_model_requirements(a_namer)
        self._check_floyd_requirements(a_namer)

    def _check_floyd_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check Floyd model specific requirements."""
        if _pipelinewrapper.model_type_is_floyd(self.model_type):
            if self.is_output_latents():
                raise RenderLoopConfigError(
                    f'Outputting latents with {a_namer("image_format")} {self.image_format} '
                    f'is not supported with Deep Floyd model types.'
                )

    def _check_stable_cascade_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check Stable Cascade specific requirements."""
        if self.model_type == _pipelinewrapper.ModelType.S_CASCADE_DECODER:
            raise RenderLoopConfigError(
                f'Stable Cascade decoder {a_namer("model_type")} may not be used as the primary model.')

        if self.model_type == _pipelinewrapper.ModelType.S_CASCADE:
            if self.hi_diffusion:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion")} is not supported with Stable Cascade.'
                )

            if not self.second_model_guidance_scales:
                self.second_model_guidance_scales = [
                    _pipelinewrapper.constants.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE]

            if not self.second_model_inference_steps:
                self.second_model_inference_steps = [
                    _pipelinewrapper.constants.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS]

            if self.output_size is not None and not _image.is_aligned(self.output_size, 128):
                raise RenderLoopConfigError(
                    f'Stable Cascade requires {a_namer("output_size")} to be aligned by 128.')

        elif self.s_cascade_decoder_uri:
            raise RenderLoopConfigError(
                f'{a_namer("s_cascade_decoder_uri")} may only be used with "s-cascade"')

    def _check_upscaler_requirements(self, a_namer: typing.Callable[[str], str]) -> bool:
        """
        Check upscaler model specific requirements.
        
        Returns: Whether upscaler noise levels default was set
        """
        upscaler_noise_levels_default_set = False

        if not _pipelinewrapper.model_type_is_upscaler(self.model_type) \
                and not _pipelinewrapper.model_type_is_floyd_ifs(self.model_type):
            if self.upscaler_noise_levels:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("upscaler_noise_levels")} for a '
                    f'non upscaler model type, see: {a_namer("model_type")}.')
        elif self.upscaler_noise_levels is None:
            if self.model_type == _pipelinewrapper.ModelType.UPSCALER_X4:
                upscaler_noise_levels_default_set = True
                self.upscaler_noise_levels = [_pipelinewrapper.constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL]
        elif self.model_type != _pipelinewrapper.ModelType.UPSCALER_X4 and \
                not _pipelinewrapper.model_type_is_floyd_ifs(self.model_type):
            raise RenderLoopConfigError(
                f'you cannot specify {a_namer("upscaler_noise_levels")} for an upscaler '
                f'model type that is not "upscaler-x4" or "ifs-*", '
                f'see: {a_namer("model_type")}.')

        return upscaler_noise_levels_default_set

    def _check_pix2pix_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check pix2pix model specific requirements."""
        if not _pipelinewrapper.model_type_is_pix2pix(self.model_type):
            if self.image_guidance_scales:
                raise RenderLoopConfigError(
                    f'argument {a_namer("image_guidance_scales")} only valid with '
                    f'pix2pix models, see: {a_namer("model_type")}.')
        elif not self.image_guidance_scales:
            self.image_guidance_scales = [_pipelinewrapper.constants.DEFAULT_IMAGE_GUIDANCE_SCALE]

    def _check_transformer_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check transformer compatibility with the model type."""
        if self.transformer_uri:
            if not _pipelinewrapper.model_type_is_sd3(self.model_type) \
                    and not _pipelinewrapper.model_type_is_flux(self.model_type):
                raise _pipelinewrapper.UnsupportedPipelineConfigError(
                    f'{a_namer("transformer_uri")} is only supported for '
                    f'{a_namer("model_type")} sd3 and flux.')

    def _check_flux_model_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check Flux model specific requirements."""
        if not _pipelinewrapper.model_type_is_flux(self.model_type):
            invalid_self = []
            for flux_self in self._non_null_attr_that_start_with('flux'):
                invalid_self.append(f'you cannot specify {a_namer(flux_self)} '
                                    f'for a non Flux model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))
        else:
            if self.hi_diffusion:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion")} is not supported with Flux.'
                )
            if self.max_sequence_length is not None:
                if self.max_sequence_length < 1 or self.max_sequence_length > 512:
                    raise RenderLoopConfigError(
                        f'{a_namer("max_sequence_length")} must be greater than or equal '
                        f'to 1 and less than or equal to 512.'
                    )

    def _check_sd3_model_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check SD3 model specific requirements."""
        if not _pipelinewrapper.model_type_is_sd3(self.model_type):
            invalid_self = []
            for sd3_self in self._non_null_attr_that_start_with('sd3'):
                invalid_self.append(f'you cannot specify {a_namer(sd3_self)} '
                                    f'for a non SD3 model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))
        else:
            if self.hi_diffusion:
                raise RenderLoopConfigError(
                    f'{a_namer("hi_diffusion")} is not supported with Stable Diffusion 3.'
                )

            if self.max_sequence_length is not None:
                if self.controlnet_uris:
                    raise RenderLoopConfigError(
                        f'{a_namer("max_sequence_length")} is not supported when '
                        f'{a_namer("controlnet_uris")} is specified.')

                if self.max_sequence_length < 1 or self.max_sequence_length > 512:
                    raise RenderLoopConfigError(
                        f'{a_namer("max_sequence_length")} must be greater than or equal '
                        f'to 1 and less than or equal to 512.'
                    )

            if self.output_size is not None:
                if not _image.is_aligned(self.output_size, 16):
                    raise RenderLoopConfigError(
                        f'Stable Diffusion 3 requires {a_namer("output_size")} to be aligned to 16.')

            if self.unet_uri or self.second_model_unet_uri:
                raise RenderLoopConfigError(
                    f'Stable Diffusion 3 does not support the '
                    f'use of {a_namer("unet_uri")}/{a_namer("second_model_unet_uri")}.')

    def _check_sdxl_model_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check SDXL model specific requirements."""
        # Check if we're using an SDXL or Kolors model
        is_sdxl_model = (_pipelinewrapper.model_type_is_sdxl(self.model_type) or
                         _pipelinewrapper.model_type_is_kolors(self.model_type))

        if not is_sdxl_model:
            # Not an SDXL model, check for incompatible arguments
            invalid_self = []
            for sdxl_self in self._non_null_attr_that_start_with('sdxl'):
                invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                    f'for a non SDXL model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

            # Clear high noise fractions since it's not applicable
            self.sdxl_high_noise_fractions = None
            return

        # We have an SDXL or Kolors model, check refiner compatibility
        if not self.sdxl_refiner_uri:
            # No refiner specified, check for incompatible refiner-specific arguments
            invalid_self = []
            for sdxl_self in self._non_null_second_model_arguments('sdxl_refiner'):
                invalid_self.append(
                    f'you cannot specify {a_namer(sdxl_self)} '
                    f'without {a_namer("sdxl_refiner_uri")}.')

            # High noise fractions require a refiner
            if self.sdxl_high_noise_fractions is not None:
                invalid_self.append(
                    f'you cannot specify {a_namer("sdxl_high_noise_fractions")} '
                    f'without {a_namer("sdxl_refiner_uri")}.')

            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))
        else:
            # Refiner is specified, set default high noise fraction if needed
            if self.sdxl_high_noise_fractions is None:
                self.sdxl_high_noise_fractions = [_pipelinewrapper.constants.DEFAULT_SDXL_HIGH_NOISE_FRACTION]

    def _process_image_seeds(self, a_namer: typing.Callable[[str], str], help_mode: bool):
        """Process image seeds and verify compatibility."""
        if not self.image_seeds:
            return

        # Check if model type supports image seed strength
        no_seed_strength = (_pipelinewrapper.model_type_is_upscaler(self.model_type) or
                            _pipelinewrapper.model_type_is_pix2pix(self.model_type) or
                            _pipelinewrapper.model_type_is_s_cascade(self.model_type) or
                            self.model_type == _pipelinewrapper.ModelType.FLUX_FILL or
                            self.model_type == _pipelinewrapper.ModelType.FLUX_KONTEXT)

        # Set default image seed strength if needed
        image_seed_strengths_default_set = False
        user_provided_image_seed_strengths = False

        if self.image_seed_strengths is None:
            if not no_seed_strength:
                image_seed_strengths_default_set = True
                self.image_seed_strengths = [_pipelinewrapper.constants.DEFAULT_IMAGE_SEED_STRENGTH]
        else:
            if no_seed_strength:
                raise RenderLoopConfigError(
                    f'{a_namer("image_seed_strengths")} '
                    f'cannot be used with pix2pix, upscaler, stablecascade, flux-fill, or flux-kontext models.')
            user_provided_image_seed_strengths = True

        # Check upscaler noise level default setting
        upscaler_noise_levels_default_set = self._check_upscaler_requirements(a_namer)

        # Parse and validate each image seed URI
        parsed_image_seeds = []
        for uri in self.image_seeds:
            parsed_image_seeds.append(
                self._check_image_seed_uri(
                    uri=uri,
                    attribute_namer=a_namer,
                    image_seed_strengths_default_set=image_seed_strengths_default_set,
                    upscaler_noise_levels_default_set=upscaler_noise_levels_default_set))

        # Verify pipeline compatibility
        self._verify_pipeline_compatibility(
            parsed_image_seeds=parsed_image_seeds,
            help_mode=help_mode,
            a_namer=a_namer
        )

        # Check for additional compatibility issues
        self._check_adetailer_mask_compatibility(
            parsed_image_seeds=parsed_image_seeds,
            a_namer=a_namer
        )

        self._check_image_input_compatibility(
            parsed_image_seeds=parsed_image_seeds,
            image_seed_strengths_default_set=image_seed_strengths_default_set,
            user_provided_image_seed_strengths=user_provided_image_seed_strengths,
            a_namer=a_namer
        )

        # Store parsed image seeds
        self.parsed_image_seeds = parsed_image_seeds

    def _verify_pipeline_compatibility(self,
                                       parsed_image_seeds: typing.List[_mediainput.ImageSeedParseResult],
                                       help_mode: bool,
                                       a_namer: typing.Callable[[str], str]):
        """Verify that a pipeline can be built for the given configuration."""
        try:
            for image_seed in parsed_image_seeds:
                is_control_guidance_spec = \
                    (self.controlnet_uris or self.t2i_adapter_uris) and image_seed.is_single_spec

                if image_seed.images and (self.adetailer_detector_uris or image_seed.mask_images):
                    pipeline_type = _pipelinewrapper.PipelineType.INPAINT
                elif image_seed.images and not is_control_guidance_spec:
                    pipeline_type = _pipelinewrapper.PipelineType.IMG2IMG
                else:
                    pipeline_type = _pipelinewrapper.PipelineType.TXT2IMG

                # Check if a class can handle the operation
                _pipelinewrapper.get_pipeline_class(
                    model_type=self.model_type,
                    pipeline_type=pipeline_type,
                    unet_uri=self.unet_uri,
                    transformer_uri=self.transformer_uri,
                    vae_uri=self.vae_uri,
                    lora_uris=self.lora_uris,
                    image_encoder_uri=self.image_encoder_uri,
                    ip_adapter_uris=self.ip_adapter_uris,
                    textual_inversion_uris=self.textual_inversion_uris,
                    controlnet_uris=self.controlnet_uris,
                    t2i_adapter_uris=self.t2i_adapter_uris,
                    pag=self.pag,
                    help_mode=help_mode
                )
        except _pipelinewrapper.UnsupportedPipelineConfigError as e:
            raise RenderLoopConfigError(str(e)) from e

    def _check_adetailer_mask_compatibility(self,
                                            parsed_image_seeds: typing.List[_mediainput.ImageSeedParseResult],
                                            a_namer: typing.Callable[[str], str]):
        """Check that adetailer is not used with manual inpaint masks."""
        if self.adetailer_detector_uris and any(p.mask_images is not None for p in parsed_image_seeds):
            raise RenderLoopConfigError(
                f'Cannot specify inpaint masks when using {a_namer("adetailer_detector_uris")}, inpaint masks '
                f'are generated automatically with this option.'
            )

    def _check_image_input_compatibility(self,
                                         parsed_image_seeds: typing.List[_mediainput.ImageSeedParseResult],
                                         image_seed_strengths_default_set: bool,
                                         user_provided_image_seed_strengths: bool,
                                         a_namer: typing.Callable[[str], str]):
        """Check compatibility of image inputs with model type and settings."""
        # Check if no images are provided for img2img
        if all(p.images is None for p in parsed_image_seeds):
            if image_seed_strengths_default_set:
                self.image_seed_strengths = None
            elif self.image_seed_strengths:
                raise RenderLoopConfigError(
                    f'{a_namer("image_seed_strengths")} '
                    f'cannot be used unless an img2img operation exists '
                    f'in at least one {a_namer("image_seeds")} definition.')

        # Check flux-fill compatibility
        if not all(p.mask_images is not None for p in parsed_image_seeds):
            if self.model_type == _pipelinewrapper.ModelType.FLUX_FILL \
                    and not self.adetailer_detector_uris:
                raise RenderLoopConfigError(
                    f'Only inpainting {a_namer("image_seeds")} '
                    f'definitions can be used with {a_namer("model_type")} flux-fill.')

        # Check non-inpainting mode compatibility
        if all(p.mask_images is None for p in parsed_image_seeds):

            if self.model_type == _pipelinewrapper.ModelType.IFS:
                self.image_seed_strengths = None
                if user_provided_image_seed_strengths:
                    raise RenderLoopConfigError(
                        f'{a_namer("image_seed_strengths")} '
                        f'cannot be used with {a_namer("model_type")} ifs '
                        f'when no inpainting is taking place in any image seed '
                        f'specification.')

            if self.inpaint_crop:
                raise RenderLoopConfigError(
                    f'All image seeds must be inpainting '
                    f'definitions when {a_namer("inpaint_crop")} or related arguments are used.'
                )

        # Check batching compatibility
        if self.inpaint_crop:
            for image_seed in parsed_image_seeds:
                if ((image_seed.images is not None and len(list(image_seed.images)) > 1) or
                        (image_seed.mask_images is not None and len(list(image_seed.mask_images)) > 1)):
                    raise RenderLoopConfigError(
                        f'Inpaint batching via {a_namer("image_seeds")} batching syntax is '
                        f'not supported with {a_namer("inpaint_crop")}, but you may '
                        f'use {a_namer("batch_size")}')

    def _check_image_seed_uri(self,
                              uri: str,
                              attribute_namer: typing.Callable[[str], str],
                              image_seed_strengths_default_set: bool,
                              upscaler_noise_levels_default_set: bool) -> _mediainput.ImageSeedParseResult:
        """
        Check image seed URI for compatibility with the current configuration.
        
        :param uri: The URI to check
        :param attribute_namer: Function to name attributes in error messages
        :param image_seed_strengths_default_set: Whether check() has set an image_seed_strengths default value
        :param upscaler_noise_levels_default_set: Whether check() has set an upscaler_noise_levels default value
        :return: Parsed image seed result
        """
        a_namer = attribute_namer

        try:
            parsed = _mediainput.parse_image_seed_uri(uri)
        except _mediainput.ImageSeedError as e:
            raise RenderLoopConfigError(e) from e

        mask_part = 'mask=my-mask.png;' if parsed.mask_images else ''
        # ^ Used for nice messages about image seed keyword argument misuse

        # Check model-specific image seed requirements
        self._check_model_specific_image_seed_requirements(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer
        )

        # Check adapter image compatibility
        self._check_adapter_image_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer
        )

        # Check image processor compatibility
        self._check_image_processor_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer
        )

        # Check latents processor compatibility
        self._check_latents_processor_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer
        )

        # Check adapter compatibility
        self._check_adapter_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer
        )

        # Check control guidance compatibility
        self._check_control_guidance_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer,
            image_seed_strengths_default_set=image_seed_strengths_default_set,
            upscaler_noise_levels_default_set=upscaler_noise_levels_default_set
        )

        # Check deep floyd compatibility
        self._check_floyd_compatibility(
            parsed=parsed,
            uri=uri,
            a_namer=a_namer,
            mask_part=mask_part
        )

        return parsed

    def _check_model_specific_image_seed_requirements(self,
                                                      parsed: _mediainput.ImageSeedParseResult,
                                                      uri: str,
                                                      a_namer: typing.Callable[[str], str]):
        """Check model-specific requirements for image seeds."""
        if _pipelinewrapper.model_type_is_s_cascade(self.model_type):
            if not parsed.is_single_spec:
                raise RenderLoopConfigError(
                    f'{a_namer("image_seeds")} configurations other than plain '
                    f'img2img are not supported for Stable Cascade.')

        if _pipelinewrapper.model_type_is_upscaler(self.model_type):
            if not parsed.is_single_spec:
                raise RenderLoopConfigError(
                    f'{a_namer("image_seeds")} configurations other than plain '
                    f'img2img are not supported for Stable Diffusion upscaler models.')

        if _pipelinewrapper.model_type_is_sdxl(self.model_type):
            if self.denoising_start is not None and self.denoising_start != 0.0:
                if not parsed.images:
                    raise RenderLoopConfigError(
                        f'{a_namer("denoising_start")} may not be used with SDXL and an '
                        f'{a_namer("image_seeds")} URI that does not contain an img2img image source, '
                        f'if you are trying to refine latents, pass them in as a normal img2img definition.')

    def _check_adapter_image_compatibility(self,
                                           parsed: _mediainput.ImageSeedParseResult,
                                           uri: str,
                                           a_namer: typing.Callable[[str], str]):
        """Check compatibility of adapter images with IP adapters."""
        if parsed.adapter_images:
            if not self.ip_adapter_uris:
                raise RenderLoopConfigError(
                    f'You may not use IP Adapter images in your {a_namer("image_seeds")} specification '
                    f'without specifying {a_namer("ip_adapter_uris")}.')

            number_of_ip_adapter_image_groups = len(parsed.adapter_images)
            number_of_ip_adapter_uris = len(self.ip_adapter_uris)

            if len(parsed.adapter_images) != number_of_ip_adapter_uris:
                raise RenderLoopConfigError(
                    f'The amount of IP Adapter image groups in your {a_namer("image_seeds")} specification "{uri}" '
                    f'must equal the number of {a_namer("ip_adapter_uris")} supplied. You have supplied '
                    f'{number_of_ip_adapter_image_groups} IP Adapter image groups for {number_of_ip_adapter_uris} IP '
                    f'Adapter models.')

    def _check_latents_processor_compatibility(self,
                                               parsed: _mediainput.ImageSeedParseResult,
                                               uri: str,
                                               a_namer: typing.Callable[[str], str]):
        if self.img2img_latents_processors:
            if not any(_mediainput.is_tensor_file(i) for i in parsed.images):
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" does not '
                    f'contain any tensor file (latents) img2img inputs, therefore '
                    f'{a_namer("img2img_latents_processors")} cannot be used.'
                )

            num_img2img_latents = len(parsed.images) if parsed.images is not None else 0

            latents_processor_chain_count = \
                (sum(1 for p in self.img2img_latents_processors if
                     p == _pipelinewrapper.constants.LATENTS_PROCESSOR_SEP) + 1)

            if latents_processor_chain_count > num_img2img_latents:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_img2img_latents} '
                    f'tensor file (latents) img2img image sources, and you have specified {latents_processor_chain_count} '
                    f'{a_namer("img2img_latents_processors")} actions / action chains. The amount of processors '
                    f'must not exceed the amount of inputs.'
                )

        if self.latents_processors:
            if not parsed.latents:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" does not '
                    f'contain any raw latents tensor file inputs, therefore '
                    f'{a_namer("img2img_latents_processors")} cannot be used.'
                )

            num_latents = len(parsed.latents) if parsed.latents is not None else 0

            latents_processor_chain_count = \
                (sum(1 for p in self.latents_processors if
                     p == _pipelinewrapper.constants.LATENTS_PROCESSOR_SEP) + 1)

            if latents_processor_chain_count > num_latents:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_latents} '
                    f'tensor file (raw latents) inputs, and you have specified {latents_processor_chain_count} '
                    f'{a_namer("latents_processors")} actions / action chains. The amount of processors '
                    f'must not exceed the amount of inputs.'
                )

    def _check_image_processor_compatibility(self,
                                             parsed: _mediainput.ImageSeedParseResult,
                                             uri: str,
                                             a_namer: typing.Callable[[str], str]):
        """Check compatibility of image processors with available images."""
        if self.seed_image_processors:
            num_img2img_images = len(parsed.images) if parsed.images is not None else 0

            seed_processor_chain_count = \
                (sum(1 for p in self.seed_image_processors if p == IMAGE_PROCESSOR_SEP) + 1)

            if seed_processor_chain_count > num_img2img_images:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_img2img_images} '
                    f'img2img image sources, and you have specified {seed_processor_chain_count} '
                    f'{a_namer("seed_image_processors")} actions / action chains. The amount of processors '
                    f'must not exceed the amount of img2img images.'
                )

        if self.mask_image_processors:
            num_mask_images = len(parsed.mask_images) if parsed.mask_images is not None else 0

            mask_processor_chain_count = \
                (sum(1 for p in self.mask_image_processors if p == IMAGE_PROCESSOR_SEP) + 1)

            if mask_processor_chain_count > num_mask_images:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_mask_images} '
                    f'inpaint mask image sources, and you have specified {mask_processor_chain_count} '
                    f'{a_namer("mask_image_processors")} actions / action chains. The amount of processors '
                    f'must not exceed the amount of inpaint mask images.'
                )

    def _check_adapter_compatibility(self,
                                     parsed: _mediainput.ImageSeedParseResult,
                                     uri: str,
                                     a_namer: typing.Callable[[str], str]):
        """Check T2I adapter or ControlNet compatibility with the image seeds."""
        control_image_paths = parsed.get_control_image_paths()
        num_control_images = len(control_image_paths) if control_image_paths is not None else 0

        if self.t2i_adapter_uris:
            if not parsed.is_single_spec:
                raise RenderLoopConfigError(
                    f'You may not use img2img or inpainting in your {a_namer("image_seeds")} specification '
                    f'when using {a_namer("t2i_adapter_uris")} as it is not supported.')

            if num_control_images != len(self.t2i_adapter_uris):
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_control_images} '
                    f'control guidance image sources, and you have specified {len(self.t2i_adapter_uris)} '
                    f'{a_namer("t2i_adapter_uris")} URIs. The amount of guidance image sources and the '
                    f'amount of T2I Adapter models must be equal.'
                )

        elif self.controlnet_uris:
            if not parsed.is_single_spec and parsed.control_images is None:
                images_str = ', '.join(parsed.images)
                raise RenderLoopConfigError(
                    f'You must specify a control image with the control argument '
                    f'IE: "my-seed.png;control=my-control.png" in your '
                    f'{a_namer("image_seeds")} "{uri}" when using {a_namer("controlnet_uris")} '
                    f'in order to use inpainting. If you want to use the control image alone '
                    f'without a mask, use {a_namer("image_seeds")} "{images_str}".')

            if control_image_paths is None:
                raise RenderLoopConfigError(
                    f'You must specify controlnet guidance images in your {a_namer("image_seeds")} '
                    f'specification "{uri}" (for example: "img2img;mask=my-mask.png;control=control1.png, control2.png") '
                    f'when using {a_namer("controlnet_uris")}'
                )

            if num_control_images != len(self.controlnet_uris):
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_control_images} '
                    f'control guidance image sources, and you have specified {len(self.controlnet_uris)} '
                    f'{a_namer("controlnet_uris")} URIs. The amount of guidance image sources and the '
                    f'amount of ControlNet models must be equal.'
                )

        if self.control_image_processors:
            control_processor_chain_count = \
                (sum(1 for p in self.control_image_processors if p == IMAGE_PROCESSOR_SEP) + 1)

            if control_processor_chain_count > num_control_images:
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_control_images} '
                    f'control guidance image sources, and you have specified {control_processor_chain_count} '
                    f'{a_namer("control_image_processors")} actions / action chains. The amount of processors '
                    f'must not exceed the amount of control guidance images.'
                )

    def _check_control_guidance_compatibility(self,
                                              parsed: _mediainput.ImageSeedParseResult,
                                              uri: str,
                                              a_namer: typing.Callable[[str], str],
                                              image_seed_strengths_default_set: bool,
                                              upscaler_noise_levels_default_set: bool):
        """Check control guidance compatibility."""
        is_control_guidance_spec = (self.controlnet_uris or self.t2i_adapter_uris) and parsed.is_single_spec
        has_additional_control = parsed.control_images and not parsed.is_single_spec

        if has_additional_control and not self.controlnet_uris:
            raise RenderLoopConfigError(
                f'Cannot use the "control" argument in an image seed without '
                f'specifying {a_namer("controlnet_uris")}.'
            )

        if is_control_guidance_spec and self.image_seed_strengths:
            if image_seed_strengths_default_set:
                # check() set this default that isn't valid
                # upon further parsing
                self.image_seed_strengths = None
            else:
                # user set this
                raise RenderLoopConfigError(
                    f'Cannot use {a_namer("image_seed_strengths")} with a control guidance image '
                    f'specification "{uri}". IE: when {a_namer("controlnet_uris")} is specified and '
                    f'your {a_namer("image_seeds")} specification has a single source or comma '
                    f'separated list of sources.')

        if is_control_guidance_spec and self.upscaler_noise_levels:
            # upscaler noise level should already be handled but handle it again just incase
            if upscaler_noise_levels_default_set:
                # check() set this default that isn't valid
                # upon further parsing
                self.upscaler_noise_levels = None
            else:
                # user set this
                raise RenderLoopConfigError(
                    f'Cannot use {a_namer("upscaler_noise_levels")} with a control guidance image '
                    f'specification "{uri}". IE: when {a_namer("controlnet_uris")} is specified and '
                    f'your {a_namer("image_seeds")} specification has a single source or comma '
                    f'separated list of sources.')

    def _check_floyd_compatibility(self,
                                   parsed: _mediainput.ImageSeedParseResult,
                                   uri: str,
                                   a_namer: typing.Callable[[str], str],
                                   mask_part: str):
        """Check compatibility with Deep Floyd IF models."""
        if self.model_type == _pipelinewrapper.ModelType.IFS_IMG2IMG or \
                (parsed.mask_images and _pipelinewrapper.model_type_is_floyd_ifs(self.model_type)):

            if not parsed.floyd_image:
                raise RenderLoopConfigError(
                    f'You must specify a floyd image with the floyd argument '
                    f'IE: "my-seed.png;{mask_part}floyd=previous-stage-image.png" '
                    f'in your {a_namer("image_seeds")} "{uri}" to disambiguate this '
                    f'usage of Deep Floyd IF super-resolution.')

    def _normalized_schedulers(self) -> typing.Tuple[
        typing.List[typing.Optional[_types.Uri]], typing.List[typing.Optional[_types.Uri]]]:
        """
        Return normalized lists of schedulers and second model schedulers.
        
        Returns:
            A tuple containing two lists: (schedulers, second_model_schedulers)
        """
        schedulers = self.scheduler_uri
        second_model_schedulers = self.second_model_scheduler_uri

        if isinstance(schedulers, (_types.Uri, type(None))):
            schedulers = [schedulers]
        if isinstance(second_model_schedulers, (_types.Uri, type(None))):
            second_model_schedulers = [second_model_schedulers]
        return schedulers, second_model_schedulers

    def check(self, attribute_namer: typing.Optional[typing.Callable[[str], str]] = None):
        """
        Check the configuration for type and logical usage errors, set
        defaults for certain values when needed and not specified.

        This may modify the configuration.

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """
        try:
            self._check(attribute_namer)
        except (_pipelinewrapper.InvalidModelUriError, _mediainput.ImageSeedParseError) as e:
            raise RenderLoopConfigError(e) from e

    def calculate_generation_steps(self) -> int:
        """
        Calculate the number of generation steps that this configuration results in.

        This factors in diffusion parameter combinations as well as scheduler combinations.

        :return: int
        """
        optional_factors = [
            self.second_model_prompts,
            self.second_model_second_prompts,
            self.second_prompts,
            self.third_prompts,
            self.image_guidance_scales,
            self.image_seeds,
            self.image_seed_strengths,
            self.sdxl_t2i_adapter_factors,
            self.clip_skips,
            self.sdxl_refiner_clip_skips,
            self.upscaler_noise_levels,
            self.guidance_rescales,
            self.freeu_params,
            self.sdxl_refiner_freeu_params,
            self.sdxl_high_noise_fractions,
            self.sdxl_aesthetic_scores,
            self.sdxl_original_sizes,
            self.sdxl_target_sizes,
            self.sdxl_crops_coords_top_left,
            self.sdxl_negative_aesthetic_scores,
            self.sdxl_negative_original_sizes,
            self.sdxl_negative_target_sizes,
            self.sdxl_negative_crops_coords_top_left,
            self.sdxl_refiner_aesthetic_scores,
            self.sdxl_refiner_original_sizes,
            self.sdxl_refiner_target_sizes,
            self.sdxl_refiner_crops_coords_top_left,
            self.sdxl_refiner_negative_aesthetic_scores,
            self.sdxl_refiner_negative_original_sizes,
            self.sdxl_refiner_negative_target_sizes,
            self.sdxl_refiner_negative_crops_coords_top_left,
            self.pag_scales,
            self.pag_adaptive_scales,
            self.sdxl_refiner_pag_scales,
            self.sdxl_refiner_pag_adaptive_scales,
            self.second_model_inference_steps,
            self.second_model_guidance_scales,
            self.sdxl_refiner_guidance_rescales,
            self.adetailer_mask_shapes,
            self.adetailer_detector_paddings,
            self.adetailer_mask_paddings,
            self.adetailer_mask_blurs,
            self.adetailer_mask_dilations,
            self.adetailer_sizes,
            self.tea_cache_rel_l1_thresholds,
            self.ras_error_reset_steps,
            self.ras_high_ratios,
            self.ras_sample_ratios,
            self.ras_starvation_scales,
            self.ras_metrics,
            self.ras_start_steps,
            self.ras_end_steps,
            self.ras_skip_num_steps,
            self.ras_skip_num_step_lengths,
            self.deep_cache_intervals,
            self.deep_cache_branch_ids,
            self.sdxl_refiner_deep_cache_intervals,
            self.sdxl_refiner_deep_cache_branch_ids,
            self.sigmas,
            self.sdxl_refiner_sigmas,
            self.sada_max_downsamples,
            self.sada_sxs,
            self.sada_sys,
            self.sada_acc_ranges,
            self.sada_lagrange_terms,
            self.sada_lagrange_ints,
            self.sada_lagrange_steps,
            self.sada_max_fixes,
            self.sada_max_intervals
        ]

        schedulers, second_model_schedulers = self._normalized_schedulers()

        product = 1
        for lst in optional_factors:
            product *= max(0 if lst is None else len(lst), 1)

        return (product *
                len(self.prompts) *
                (len(self.seeds) if not self.seeds_to_images else 1) *
                len(self.guidance_scales) *
                len(self.inference_steps) *
                len(schedulers) *
                len(second_model_schedulers))

    def copy(self) -> 'RenderLoopConfig':
        """
        Create a deep copy of this :py:class:`RenderLoopConfig` instance.
        
        :return: :py:class:`RenderLoopConfig` instance that is a deep copy of this instance.
        """
        new_config = RenderLoopConfig()

        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (list, tuple, dict, set)):
                new_config.__dict__[attr_name] = _types.partial_deep_copy_container(attr_value)
            elif hasattr(attr_value, 'copy') and callable(getattr(attr_value, 'copy')):
                new_config.__dict__[attr_name] = attr_value.copy()
            else:
                new_config.__dict__[attr_name] = attr_value

        return new_config

    def apply_prompt_upscalers(self):
        """
        Apply requested prompt upscaling operations to all prompts in the configuration.

        This potentially modifies the configuration in place, specifically the prompt arguments.

        :raises dgenerate.promptupscalers.PromptUpscalerNotFoundError:
        :raises dgenerate.promptupscalers.PromptUpscalerArgumentError:
        :raises dgenerate.promptupscalers.PromptUpscalerProcessingError:
        """

        def upscale_prompts(prompts, default_upscaler_uri):
            return _promptupscalers.upscale_prompts(
                prompts=prompts,
                default_upscaler_uri=default_upscaler_uri,
                device=self.device,
                local_files_only=self.offline_mode
            )

        self.prompts = upscale_prompts(self.prompts, self.prompt_upscaler_uri)

        if self.second_prompts:
            self.second_prompts = upscale_prompts(
                self.second_prompts, _types.default(
                    self.second_prompt_upscaler_uri, self.prompt_upscaler_uri))

        if self.third_prompts:
            self.third_prompts = upscale_prompts(
                self.third_prompts, _types.default(
                    self.third_prompt_upscaler_uri, self.prompt_upscaler_uri))

        if self.second_model_prompts:
            self.second_model_prompts = upscale_prompts(
                self.second_model_prompts, _types.default(
                    self.second_model_prompt_upscaler_uri, self.prompt_upscaler_uri))

        if self.second_model_second_prompts:
            self.second_model_second_prompts = upscale_prompts(
                self.second_model_second_prompts, _types.default(
                    self.second_model_second_prompt_upscaler_uri, self.prompt_upscaler_uri))

    def iterate_diffusion_args(self, **overrides) -> collections.abc.Iterator[_pipelinewrapper.DiffusionArguments]:
        """
        Iterate over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` argument objects using
        every combination of argument values provided for that object by this configuration.

        :param overrides: use key word arguments to override specific attributes of this object with a new list value.
        :return: an iterator over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
        """

        def ov(n: str, v: typing.Any) -> typing.Optional[typing.List[typing.Any]]:
            if not hasattr(_pipelinewrapper.DiffusionArguments, n):
                # fat-fingered an argument name too many times :)
                raise AssertionError(
                    f'dgenerate.pipelinewrapper.DiffusionArguments lacks property: {n}')

            if not (
                    _pipelinewrapper.model_type_is_sdxl(self.model_type) or
                    _pipelinewrapper.model_type_is_kolors(self.model_type)):
                if n.startswith('sdxl_'):
                    return None
            else:
                if n.startswith('sdxl_refiner_') and not self.sdxl_refiner_uri:
                    return None

            if not _pipelinewrapper.model_type_is_sd3(self.model_type):
                if n.startswith('sd3_'):
                    return None

            if not _pipelinewrapper.model_type_is_flux(self.model_type):
                if n.startswith('flux_'):
                    return None

            if not self.adetailer_detector_uris:
                if n.startswith('adetailer_'):
                    return None

            if n in overrides:
                return overrides[n]
            return v

        schedulers = [self.scheduler_uri] if \
            isinstance(self.scheduler_uri, (str, type(None))) else \
            self.scheduler_uri

        second_model_schedulers = [self.second_model_scheduler_uri] if \
            isinstance(self.second_model_scheduler_uri, (str, type(None))) else \
            self.second_model_scheduler_uri

        for arg in _iterate_diffusion_args(
                prompt=ov('prompt', self.prompts),
                prompt_weighter_uri=ov('prompt_weighter_uri', [self.prompt_weighter_uri]),
                vae_tiling=ov('vae_tiling', [self.vae_tiling]),
                vae_slicing=ov('vae_slicing', [self.vae_slicing]),
                scheduler_uri=ov('scheduler_uri', schedulers),
                second_model_scheduler_uri=
                ov('second_model_scheduler_uri', second_model_schedulers),
                second_model_prompt_weighter_uri=
                ov('second_model_prompt_weighter_uri', [self.second_model_prompt_weighter_uri]),
                second_prompt=ov('second_prompt', self.second_prompts),
                third_prompt=ov('third_prompt', self.third_prompts),
                second_model_prompt=ov('second_model_prompt', self.second_model_prompts),
                second_model_second_prompt=ov('second_model_second_prompt', self.second_model_second_prompts),
                max_sequence_length=ov('max_sequence_length', [self.max_sequence_length]),
                seed=ov('seed', self.seeds),
                clip_skip=ov('clip_skip', self.clip_skips),
                sdxl_refiner_clip_skip=ov('sdxl_refiner_clip_skip', self.sdxl_refiner_clip_skips),
                sdxl_t2i_adapter_factor=ov('sdxl_t2i_adapter_factor', self.sdxl_t2i_adapter_factors),
                image_seed_strength=ov('image_seed_strength', self.image_seed_strengths),
                guidance_scale=ov('guidance_scale', self.guidance_scales),
                freeu_params=ov('freeu_params', self.freeu_params),
                hi_diffusion=ov('hi_diffusion', [self.hi_diffusion]),
                hi_diffusion_no_win_attn=ov(
                    'hi_diffusion_no_win_attn', [self.hi_diffusion_no_win_attn]),
                hi_diffusion_no_raunet=ov(
                    'hi_diffusion_no_raunet', [self.hi_diffusion_no_raunet]),
                sada=ov('sada', [self.sada]),
                sada_max_downsample=ov('sada_max_downsample', self.sada_max_downsamples),
                sada_sx=ov('sada_sx', self.sada_sxs),
                sada_sy=ov('sada_sy', self.sada_sys),
                sada_acc_range=ov('sada_acc_range', self.sada_acc_ranges),
                sada_lagrange_term=ov('sada_lagrange_term', self.sada_lagrange_terms),
                sada_lagrange_int=ov('sada_lagrange_int', self.sada_lagrange_ints),
                sada_lagrange_step=ov('sada_lagrange_step', self.sada_lagrange_steps),
                sada_max_fix=ov('sada_max_fix', self.sada_max_fixes),
                sada_max_interval=ov('sada_max_interval', self.sada_max_intervals),
                tea_cache=ov('tea_cache', [self.tea_cache]),
                tea_cache_rel_l1_threshold=ov('tea_cache_rel_l1_threshold', self.tea_cache_rel_l1_thresholds),
                ras=ov('ras', [self.ras]),
                ras_index_fusion=ov('ras_index_fusion', [self.ras_index_fusion]),
                ras_sample_ratio=ov('ras_sample_ratio', self.ras_sample_ratios),
                ras_high_ratio=ov('ras_high_ratio', self.ras_high_ratios),
                ras_starvation_scale=ov('ras_starvation_scale', self.ras_starvation_scales),
                ras_error_reset_steps=ov('ras_error_reset_steps', self.ras_error_reset_steps),
                ras_metric=ov('ras_metric', self.ras_metrics),
                ras_start_step=ov('ras_start_step', self.ras_start_steps),
                ras_end_step=ov('ras_end_step', self.ras_end_steps),
                ras_skip_num_step=ov('ras_skip_num_step', self.ras_skip_num_steps),
                ras_skip_num_step_length=ov('ras_skip_num_step_length', self.ras_skip_num_step_lengths),
                deep_cache=ov('deep_cache', [self.deep_cache]),
                deep_cache_interval=ov('deep_cache_interval', self.deep_cache_intervals),
                deep_cache_branch_id=ov('deep_cache_branch_id', self.deep_cache_branch_ids),
                sdxl_refiner_deep_cache=ov('sdxl_refiner_deep_cache', [self.sdxl_refiner_deep_cache]),
                sdxl_refiner_deep_cache_interval=ov('sdxl_refiner_deep_cache_interval',
                                                    self.sdxl_refiner_deep_cache_intervals),
                sdxl_refiner_deep_cache_branch_id=ov('sdxl_refiner_deep_cache_branch_id',
                                                     self.sdxl_refiner_deep_cache_branch_ids),
                pag_scale=ov('pag_scale', self.pag_scales),
                pag_adaptive_scale=ov('pag_adaptive_scale', self.pag_adaptive_scales),
                image_guidance_scale=ov('image_guidance_scale', self.image_guidance_scales),
                guidance_rescale=ov('guidance_rescale', self.guidance_rescales),
                sigmas=ov('sigmas', self.sigmas),
                inference_steps=ov('inference_steps', self.inference_steps),
                sdxl_high_noise_fraction=ov('sdxl_high_noise_fraction', self.sdxl_high_noise_fractions),
                second_model_inference_steps=ov('second_model_inference_steps', self.second_model_inference_steps),
                second_model_guidance_scale=ov('second_model_guidance_scale', self.second_model_guidance_scales),
                sdxl_refiner_sigmas=ov('sdxl_refiner_sigmas', self.sdxl_refiner_sigmas),
                sdxl_refiner_freeu_params=ov('sdxl_refiner_freeu_params', self.sdxl_refiner_freeu_params),
                sdxl_refiner_pag_scale=ov('sdxl_refiner_pag_scale', self.sdxl_refiner_pag_scales),
                sdxl_refiner_pag_adaptive_scale=ov('sdxl_refiner_pag_adaptive_scale',
                                                   self.sdxl_refiner_pag_adaptive_scales),
                sdxl_refiner_guidance_rescale=ov('sdxl_refiner_guidance_rescale',
                                                 self.sdxl_refiner_guidance_rescales),
                upscaler_noise_level=ov('upscaler_noise_level', self.upscaler_noise_levels),
                sdxl_aesthetic_score=ov('sdxl_aesthetic_score', self.sdxl_aesthetic_scores),
                sdxl_original_size=ov('sdxl_original_size', self.sdxl_original_sizes),
                sdxl_target_size=ov('sdxl_target_size', self.sdxl_target_sizes),
                sdxl_crops_coords_top_left=ov('sdxl_crops_coords_top_left', self.sdxl_crops_coords_top_left),
                sdxl_negative_aesthetic_score=ov('sdxl_negative_aesthetic_score',
                                                 self.sdxl_negative_aesthetic_scores),
                sdxl_negative_original_size=ov('sdxl_negative_original_size', self.sdxl_negative_original_sizes),
                sdxl_negative_target_size=ov('sdxl_negative_target_size', self.sdxl_negative_target_sizes),
                sdxl_negative_crops_coords_top_left=ov('sdxl_negative_crops_coords_top_left',
                                                       self.sdxl_negative_crops_coords_top_left),
                sdxl_refiner_aesthetic_score=ov('sdxl_refiner_aesthetic_score', self.sdxl_refiner_aesthetic_scores),
                sdxl_refiner_original_size=ov('sdxl_refiner_original_size', self.sdxl_refiner_original_sizes),
                sdxl_refiner_target_size=ov('sdxl_refiner_target_size', self.sdxl_refiner_target_sizes),
                sdxl_refiner_crops_coords_top_left=ov('sdxl_refiner_crops_coords_top_left',
                                                      self.sdxl_refiner_crops_coords_top_left),
                sdxl_refiner_negative_aesthetic_score=ov('sdxl_refiner_negative_aesthetic_score',
                                                         self.sdxl_refiner_negative_aesthetic_scores),
                sdxl_refiner_negative_original_size=ov('sdxl_refiner_negative_original_size',
                                                       self.sdxl_refiner_negative_original_sizes),
                sdxl_refiner_negative_target_size=ov('sdxl_refiner_negative_target_size',
                                                     self.sdxl_refiner_negative_target_sizes),
                sdxl_refiner_negative_crops_coords_top_left=ov('sdxl_refiner_negative_crops_coords_top_left',
                                                               self.sdxl_refiner_negative_crops_coords_top_left),
                adetailer_class_filter=ov('adetailer_class_filter', [self.adetailer_class_filter]),
                adetailer_index_filter=ov('adetailer_index_filter', [self.adetailer_index_filter]),
                adetailer_mask_shape=ov('adetailer_mask_shape', self.adetailer_mask_shapes),
                adetailer_detector_padding=ov('adetailer_detector_padding', self.adetailer_detector_paddings),
                adetailer_mask_padding=ov('adetailer_mask_padding', self.adetailer_mask_paddings),
                adetailer_mask_blur=ov('adetailer_mask_blur', self.adetailer_mask_blurs),
                adetailer_mask_dilation=ov('adetailer_mask_dilation', self.adetailer_mask_dilations),
                adetailer_size=ov('adetailer_size', self.adetailer_sizes),
                adetailer_model_masks=ov('adetailer_model_masks', [self.adetailer_model_masks]),
                output_latents=ov('output_latents', [self.is_output_latents()]),
                denoising_start=ov('denoising_start', [self.denoising_start]),
                denoising_end=ov('denoising_end', [self.denoising_end]),
                latents=ov('latents', [self.latents]),
                latents_processors=ov('latents_processors', [self.latents_processors]),
                latents_post_processors=ov('latents_post_processors', [self.latents_post_processors]),
                img2img_latents_processors=ov(
                    'img2img_latents_processors',
                    [self.img2img_latents_processors]
                ),
                inpaint_crop=ov('inpaint_crop', [self.inpaint_crop]),
                inpaint_crop_padding=ov('inpaint_crop_padding', self.inpaint_crop_paddings),
                inpaint_crop_masked=ov('inpaint_crop_masked', [self.inpaint_crop_masked]),
                inpaint_crop_feather=ov('inpaint_crop_feather', self.inpaint_crop_feathers)
        ):
            if self.output_size is not None:
                arg.width = self.output_size[0]
                arg.height = self.output_size[1]
                arg.aspect_correct = not self.no_aspect

            arg.prompt.set_embedded_args_on(
                on_object=arg,
                forbidden_checker=_pipelinewrapper.DiffusionArguments.prompt_embedded_arg_checker)

            yield arg

    def _non_null_attr_that_start_with(self, s: typing.Union[str, typing.List[str]]) -> typing.Iterator[str]:
        """
        Return an iterator of attribute names that start with the given prefix(es) and have non-None values.
        
        Args:
            s: A string prefix or list of string prefixes
            
        Returns:
            Iterator of attribute names that match the prefix(es) and have non-None values
        """
        if not isinstance(s, list):
            s = [s]
        return (a for a in dir(self) if any(a.startswith(k) for k in s) and getattr(self, a) is not None)

    def _non_null_attr_that_end_with(self, s: str) -> typing.Iterator[str]:
        """
        Return an iterator of attribute names that end with the given suffix and have non-None values.
        
        Args:
            s: A string suffix
            
        Returns:
            Iterator of attribute names that match the suffix and have non-None values
        """
        return (a for a in dir(self) if a.endswith(s) and getattr(self, a) is not None)

    def _non_null_second_model_arguments(self, prefix: typing.Optional[str] = None) -> typing.Iterator[str]:
        """
        Return an iterator of attribute names related to second models that have non-None values.
        
        Args:
            prefix: An optional prefix to further filter attributes
            
        Returns:
            Iterator of attribute names related to second models that have non-None values
        """

        def check(a: str) -> bool:
            if prefix and a.startswith(prefix):
                return True
            if a.startswith('second_model'):
                # short circuit and include
                # second model prompt arguments
                return True
            if a.startswith('second') and not \
                    (a.endswith('prompts') or a.endswith('prompt_upscaler_uri')):
                # reject primary model 'second_prompts' and `second_prompt_upscaler_uri`
                # and include everything else
                return True
            return False

        return (a for a in dir(self) if check(a) and getattr(self, a) is not None)

    def is_output_latents(self) -> bool:
        """
        Check if the current image_format results in outputting latents.
        
        :return: ``True`` if the image output format indicates to output latents.
        """
        return self.image_format in _mediaoutput.get_supported_tensor_formats()
