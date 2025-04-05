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

import dgenerate.image as _image
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.prompt as _prompt
import dgenerate.promptweighters as _promptweighters
import dgenerate.promptupscalers as _promptupscalers
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.torchutil as _torchutil

IMAGE_PROCESSOR_SEP = '+'
"""
The character that is used to separate image processor chains
when specifying processors for individual images in a group
of input images.
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
    dgenerate command line tool.  The Stable Cascade Decoder does not 
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

    inference_steps: _types.Integers
    """
    List of inference steps values, this corresponds to the ``--inference-steps`` argument of the
    dgenerate command line tool.
    """

    clip_skips: _types.OptionalIntegers = None
    """
    List of clip skip values. Clip skip is the number of layers to be skipped from CLIP while computing the 
    prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing 
    the prompt embeddings. Only supported for ``model_type`` values ``torch`` and ``torch-sdxl``, including with 
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
    of the dgenerate command line tool that is used for the :py:attr:`dgenerate.pipelinewrapper.ModelType.TORCH_UPSCALER_X4`
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

    second_model_quantizer_uri: _types.OptionalUri = None
    """
    Global quantizer URI for secondary pipeline (SDXL Refiner or Stable Cascade decoder), 
    this corresponds to the ``--second-model-quantizer`` argument of the dgenerate command line tool.
    
    The quantization backend and settings specified by this URI will be used globally 
    on the the most appropriate models associated with the secondary diffusion pipeline 
    (SDXL Refiner, Stable Cascade Decoder).
    """

    scheduler_uri: _types.OptionalUriOrUris = None
    """
    Optional primary model scheduler/sampler class name specification, this corresponds to the ``--scheduler``
    argument of the dgenerate command line tool. Setting this to 'help' will yield a help message to stdout
    describing scheduler names compatible with the current configuration upon running. Passing 'helpargs' will
    yield a help message with a list of overridable arguments for each scheduler and their typical defaults.
    
    This may be a list of schedulers, indicating to try each scheduler in turn.
    """

    hi_diffusion: bool = False
    """
    Activate HiDiffusion for the primary model? 
            
    This can increase the resolution at which the model can
    output images while retaining quality with no overhead, and 
    possibly improved performance.
    
    See: https://github.com/megvii-research/HiDiffusion
    
    This is supported for: ``--model-type torch, torch-sdxl, and --torch-kolors``.
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
    
    This is supported for: ``--model-type torch-flux*``.
    
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

    This is supported for: ``--model-type torch-flux*``.
    """

    deep_cache: bool = False
    """
    Activate DeepCache for the main model?
    
    DeepCache caches intermediate attention layer outputs to speed up
    the diffusion process. This is beneficial for higher inference steps.
                  
    See: https://github.com/horseee/DeepCache
    
    This is supported for Stable Diffusion, Stable Diffusion XL, Kolors based models.
    """
    
    deep_cache_intervals: _types.OptionalIntegers = None
    """
    Cache intervals to try for DeepCache for the main model.
    
    Controls how frequently the attention layers are cached during
    the diffusion process. Lower values cache more frequently, potentially
    resulting in more speedup but using more memory.
    
    Each value will be tried in turn.
    
    This is supported for Stable Diffusion, Stable Diffusion XL, Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.deep_cache` is enabled.
    
    (default: 5)
    """
    
    deep_cache_branch_ids: _types.OptionalIntegers = None
    """
    Branch IDs to try for DeepCache for the main model.
    
    Controls which branches of the UNet attention blocks the caching
    is applied to. Advanced usage only.
    
    Each value will be tried in turn.
    
    This is supported for Stable Diffusion, Stable Diffusion XL, Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.deep_cache` is enabled.
    
    (default: 1)
    """

    second_model_deep_cache: _types.OptionalBoolean = None
    """
    Activate DeepCache for the second model (SDXL Refiner)?
    
    See: :py:attr:`RenderLoopConfig.deep_cache`
    
    This is supported for Stable Diffusion XL and Kolors based models.
    """
    
    second_model_deep_cache_intervals: _types.OptionalIntegers = None
    """
    Cache intervals to try for DeepCache for the second model (SDXL Refiner).
    
    Controls how frequently the attention layers are cached during
    the diffusion process. Lower values cache more frequently, potentially
    resulting in more speedup but using more memory.
    
    Each value will be tried in turn.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.second_model_deep_cache` is enabled.
    
    (default: 5)
    """
    
    second_model_deep_cache_branch_ids: _types.OptionalIntegers = None
    """
    Branch IDs to try for DeepCache for the second model (SDXL Refiner).
    
    Controls which branches of the UNet attention blocks the caching
    is applied to. Advanced usage only.
    
    Each value will be tried in turn.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.second_model_deep_cache` is enabled.
    
    (default: 1)
    """

    ras: bool = False
    """
    Activate RAS (Region-Adaptive Sampling) for the primary model? 
            
    This can increase inference speed with SD3.
    
    See: https://github.com/microsoft/ras
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_index_fusion: _types.OptionalBoolean = None
    """
    Enable index fusion in RAS (Region-Adaptive Sampling) for the primary model?
    
    This can improve attention computation in RAS for SD3.
    
    See: https://github.com/microsoft/ras
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_sample_ratios: _types.OptionalFloats = None
    """
    Sample ratios to try for RAS (Region-Adaptive Sampling).
    
    For instance, setting this to 0.5 on a sequence of 4096 tokens will result in the 
    noise of averagely 2048 tokens to be updated during each RAS step. Must be between 0 and 1.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_high_ratios: _types.OptionalFloats = None
    """
    High ratios to try for RAS (Region-Adaptive Sampling).
    
    Based on the metric selected, the ratio of the high value chosen to be cached.
    Default value is 1.0, but can be set between 0 and 1 to balance the sample ratio 
    between the main subject and the background.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_starvation_scales: _types.OptionalFloats = None
    """
    Starvation scales to try for RAS (Region-Adaptive Sampling).
    
    RAS tracks how often a token is dropped and incorporates this count as a scaling factor in the
    metric for selecting tokens. This scale factor prevents excessive blurring or noise in the 
    final generated image. Larger scaling factor will result in more uniform sampling.
    Usually set between 0.0 and 1.0.
    """

    ras_metrics: _types.OptionalStrings = None
    """
    One or more RAS metrics to try.
    
    This controls how RAS measures the importance of tokens for caching.
    Valid values are "std" (standard deviation) or "l2norm" (L2 norm).
    Defaults to "std".
    """

    ras_error_reset_steps: _types.OptionalStrings = None
    """
    Error reset step patterns to try for RAS (Region-Adaptive Sampling).
    
    The dense sampling steps inserted between the RAS steps to reset the accumulated error.
    Should be a comma-separated string of step numbers, e.g. "12,22".
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_start_steps: _types.OptionalIntegers = None
    """
    Starting steps to try for RAS (Region-Adaptive Sampling).
    
    This controls when RAS begins applying its sampling strategy.
    Must be greater than or equal to 1.
    Defaults to 4 if not specified.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_end_steps: _types.OptionalIntegers = None
    """
    Ending steps to try for RAS (Region-Adaptive Sampling).
    
    This controls when RAS stops applying its sampling strategy.
    Must be greater than or equal to 1.
    Defaults to the number of inference steps if not specified.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_skip_num_steps: _types.OptionalIntegers = None
    """
    Skip steps to try for RAS (Region-Adaptive Sampling).
    
    This controls the number of steps to skip between RAS steps.
    
    The actual number of tokens skipped will be rounded down to the nearest multiple of 64.
    This ensures efficient memory access patterns for the attention computation.
    
    When used with skip_num_step_length > 0, this value determines how much to increase/decrease
    the number of skipped tokens over time. A positive value will increase the number of skipped
    tokens, while a negative value will decrease it.
    
    Each value will be tried in turn.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    
    (default: 0)
    """

    ras_skip_num_step_lengths: _types.OptionalIntegers = None
    """
    Skip step lengths to try for RAS (Region-Adaptive Sampling).
    
    This controls the length of steps to skip between RAS steps.
    Must be greater than or equal to 0.
    
    When set to 0, static dropping is used where the same number of tokens are skipped
    at each step (except for error reset steps and steps before scheduler_start_step).
    
    When greater than 0, dynamic dropping is used where the number of skipped tokens
    varies over time based on skip_num_step. The pattern repeats every skip_num_step_length steps.
    
    Each value will be tried in turn.
    
    Supplying any value implies that :py:attr:`RenderLoopConfig.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    
    (default: 0)
    """

    sdxl_refiner_hi_diffusion: _types.OptionalBoolean = None
    """
    Activate HiDiffusion for the SDXL refiner? See: :py:attr:`RenderLoopConfig.hi_diffusion`
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

    model_type: _pipelinewrapper.ModelType = _pipelinewrapper.ModelType.TORCH
    """
    Corresponds to the ``--model-type`` argument of the dgenerate command line tool.
    """

    device: _types.Name = _torchutil.default_device()
    """
    Processing device specification, for example "cuda" or "cuda:N" where N is an 
    alternate GPU id as reported by nvidia-smi if you want to specify a specific GPU.
    This corresponds to the ``--device`` argument of the dgenerate command line tool.

    The default device on MacOS is "mps".
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
    Write config text to the metadata of all written images? this data is not written to animated files, only PNGs.
    This corresponds to the ``--output-metadata`` argument of the dgenerate command line tool.
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
    
    Anything other than "png" is not compatible with ``output_metadata=True`` and a
    :py:exc:`.RenderLoopConfigError` will be raised upon running the render loop if 
    ``output_metadata=True`` and this value is not "png"
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
    Avoid ever connecting to Hugging Face hub to download models? this can be used if 
    all your models are cached or if you are only ever using hub models that exist on disk. 
    Corresponds to the ``--offline-mode`` argument of the dgenerate command line tool.
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

    adetailer_index_filter: _types.OptionalIntegers = None
    """
    A list index values that indicates what YOLO detection indices to keep, 
    the index values start at zero. Detections are sorted by their top left bounding box 
    coordinate from left to right, top to bottom, by (confidence descending). The order of 
    detections in the image is identical to the reading order of words on a page (english). 
    Inpainting will only be preformed on the specified detection indices, if no indices 
    are specified, then inpainting will be preformed on all detections.
    """

    adetailer_detector_uris: _types.OptionalUris = None
    """
    One or more adetailer YOLO detector model URIs. Corresponds directly to --adetailer-detectors.
    
    Specification of this argument enables the adetailer inpainting algorithm and requires the
    use of :py:attr:`.RenderLoopConfig.image_seeds`
    """

    adetailer_mask_shapes: _types.OptionalNames = None
    """
    One or more adetailer mask shapes to try.
    
    This indicates what mask shape adetailer should attempt to draw around a detected feature,
    the default value is "rectangle". You may also specify "circle" to generate an ellipsoid 
    shaped mask, which might be helpful for achieving better blending.
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

    adetailer_crop_control_image: _types.OptionalBoolean = None
    """
    Should adetailer crop ControlNet control images to the feature detection area? 
    Your input image and control image should be the the same dimension, 
    otherwise this argument is ignored with a warning. When this argument 
    is not specified as ``True``, the control image provided is simply resized
    to the same size as the detection area.
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

    def _create_attribute_namer(self, attribute_namer: typing.Optional[typing.Callable[[str], str]]) -> typing.Callable[[str], str]:
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
            raise RenderLoopConfigError(e)

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
                f'You cannot specify "help" or "helpargs" to {a_namer("text_encoders")} '
                f'with multiple values involved.'
            )

        if second_model_text_encoder_help and len(self.second_model_text_encoder_uris) > 1:
            raise RenderLoopConfigError(
                f'You cannot specify "help" or "helpargs" to {a_namer("second_model_text_encoders")} '
                f'with multiple values involved.'
            )
        
        return help_mode

    def _check_optimization_features(self, a_namer: typing.Callable[[str], str]):
        """Check optimization features compatibility with the selected model type."""
        # Check TeaCache compatibility
        tea_cache_enabled = (self.tea_cache or any(self._non_null_attr_that_start_with('tea_cache_')))
        if tea_cache_enabled and not _pipelinewrapper.model_type_is_flux(self.model_type):
            raise RenderLoopConfigError(
                f'{a_namer("tea_cache")} and related arguments are only '
                f'compatible with {a_namer("model_type")} torch-flux*'
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
                f'compatible with {a_namer("model_type")} torch-sd3'
            )
        if ras_enabled and self.model_cpu_offload:
            raise RenderLoopConfigError(
                f'{a_namer("model_cpu_offload")} is not compatible '
                f'with {a_namer("ras")} and related arguments.'
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
        if not _pipelinewrapper_util.is_single_file_model_load(self.model_path):
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
                    not _pipelinewrapper_util.is_single_file_model_load(
                        _pipelinewrapper.uris.SDXLRefinerUri.parse(self.sdxl_refiner_uri).model):
                raise RenderLoopConfigError(
                    f'You cannot specify {a_namer("second_model_original_config")} '
                    f'when the {a_namer("sdxl_refiner_uri")} model is not a '
                    f'single file checkpoint.'
                )
            if self.s_cascade_decoder_uri and \
                    not _pipelinewrapper_util.is_single_file_model_load(
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
            _pipelinewrapper.ModelType.TORCH,
            _pipelinewrapper.ModelType.TORCH_SDXL,
            _pipelinewrapper.ModelType.TORCH_KOLORS,
            _pipelinewrapper.ModelType.TORCH_SD3,
            _pipelinewrapper.ModelType.TORCH_FLUX,
            _pipelinewrapper.ModelType.TORCH_FLUX_FILL
        }:
            raise RenderLoopConfigError(
                f'{a_namer("adetailer_detector_uris")} is only compatible with '
                f'{a_namer("model_type")} torch, torch-sdxl, torch-kolors, torch-sd3, and torch-flux')

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
                
            if self.model_type == _pipelinewrapper.ModelType.TORCH_FLUX_FILL:
                raise RenderLoopConfigError(
                    f'you cannot use {a_namer("model_type")} '
                    f'torch-flux-fill without {a_namer("image_seeds")}.'
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
                    
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))
                
            # Check pipeline class compatibility
            try:
                _pipelinewrapper.get_torch_pipeline_class(
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
                raise RenderLoopConfigError(str(e))

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
        
        if self.image_format not in _mediaoutput.get_supported_static_image_formats():
            raise RenderLoopConfigError(
                f'Unsupported {a_namer("image_format")} value "{self.image_format}". Must be one of '
                f'{_textprocessing.oxford_comma(_mediaoutput.get_supported_static_image_formats(), "or")}')
        
        if self.image_format != "png" and self.output_metadata:
            raise RenderLoopConfigError(
                f'{a_namer("image_format")} value "{self.image_format}" is '
                f'unsupported when {a_namer("output_metadata")} is enabled.')
        
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
                f'{a_namer("device")} must be "cuda" (optionally with a device ordinal "cuda:N") or "cpu", '
                f'or other device supported by torch.')

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
        if self.clip_skips and not (self.model_type == _pipelinewrapper.ModelType.TORCH or
                                    self.model_type == _pipelinewrapper.ModelType.TORCH_SDXL or
                                    self.model_type == _pipelinewrapper.ModelType.TORCH_SD3):
            raise RenderLoopConfigError(
                f'you cannot specify {a_namer("clip_skips")} for '
                f'{a_namer("model_type")} values other than '
                f'"torch", "torch-sdxl", or "torch-sd3"')

    def _check_model_specific_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check specific requirements for different model types."""
        self._check_stable_cascade_requirements(a_namer)
        self._check_upscaler_requirements(a_namer)
        self._check_pix2pix_requirements(a_namer)
        self._check_transformer_compatibility(a_namer)
        self._check_flux_model_requirements(a_namer)
        self._check_sd3_model_requirements(a_namer)
        self._check_sdxl_model_requirements(a_namer)
        self._check_vae_compatibility(a_namer)
        self._check_deep_cache_compatibility(a_namer)

    def _check_stable_cascade_requirements(self, a_namer: typing.Callable[[str], str]):
        """Check Stable Cascade specific requirements."""
        if self.model_type == _pipelinewrapper.ModelType.TORCH_S_CASCADE_DECODER:
            raise RenderLoopConfigError(
                f'Stable Cascade decoder {a_namer("model_type")} may not be used as the primary model.')
        
        if self.model_type == _pipelinewrapper.ModelType.TORCH_S_CASCADE:
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
                f'{a_namer("s_cascade_decoder_uri")} may only be used with "torch-s-cascade"')

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
            if self.model_type == _pipelinewrapper.ModelType.TORCH_UPSCALER_X4:
                upscaler_noise_levels_default_set = True
                self.upscaler_noise_levels = [_pipelinewrapper.constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL]
        elif self.model_type != _pipelinewrapper.ModelType.TORCH_UPSCALER_X4 and \
                not _pipelinewrapper.model_type_is_floyd_ifs(self.model_type):
            raise RenderLoopConfigError(
                f'you cannot specify {a_namer("upscaler_noise_levels")} for an upscaler '
                f'model type that is not "torch-upscaler-x4" or "torch-ifs-*", '
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
                    f'{a_namer("model_type")} torch-sd3 and torch-flux.')

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

    def _check_deep_cache_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check if DeepCache can be used"""
        deep_cache_enabled = (self.deep_cache or any(self._non_null_attr_that_start_with('deep_cache_')))
        if deep_cache_enabled and not (_pipelinewrapper.model_type_is_sd15(self.model_type) or
                                       _pipelinewrapper.model_type_is_sd2(self.model_type) or
                                       _pipelinewrapper.model_type_is_sdxl(self.model_type) or
                                       _pipelinewrapper.model_type_is_kolors(self.model_type)):
            raise RenderLoopConfigError(
                f'{a_namer("deep_cache")} and related arguments are only '
                f'compatible with Stable Diffusion, Stable Diffusion XL, and Kolors.'
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

    def _check_vae_compatibility(self, a_namer: typing.Callable[[str], str]):
        """Check VAE compatibility with model type."""
        if not _pipelinewrapper.model_type_is_torch(self.model_type):
            if self.vae_tiling or self.vae_slicing:
                raise RenderLoopConfigError(
                    f'{a_namer("vae_tiling")}/{a_namer("vae_slicing")} not supported for '
                    f'non torch model type, see: {a_namer("model_type")}.')

    def _process_image_seeds(self, a_namer: typing.Callable[[str], str], help_mode: bool):
        """Process image seeds and verify compatibility."""
        if not self.image_seeds:
            return

        # Check if model type supports image seed strength
        no_seed_strength = (_pipelinewrapper.model_type_is_upscaler(self.model_type) or
                            _pipelinewrapper.model_type_is_pix2pix(self.model_type) or
                            _pipelinewrapper.model_type_is_s_cascade(self.model_type) or
                            self.model_type == _pipelinewrapper.ModelType.TORCH_FLUX_FILL)
        
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
                    f'cannot be used with pix2pix, upscaler, or stablecascade models.')
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
                _pipelinewrapper.get_torch_pipeline_class(
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
            raise RenderLoopConfigError(str(e))

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
            if self.model_type == _pipelinewrapper.ModelType.TORCH_FLUX_FILL \
                    and not self.adetailer_detector_uris:
                raise RenderLoopConfigError(
                    f'Only inpainting {a_namer("image_seeds")} '
                    f'definitions can be used with {a_namer("model_type")} torch-flux-fill.')
        
        # Check deep floyd inpainting mode compatibility
        if all(p.mask_images is None for p in parsed_image_seeds):
            if self.model_type == _pipelinewrapper.ModelType.TORCH_IFS:
                self.image_seed_strengths = None
                if user_provided_image_seed_strengths:
                    raise RenderLoopConfigError(
                        f'{a_namer("image_seed_strengths")} '
                        f'cannot be used with {a_namer("model_type")} torch-ifs '
                        f'when no inpainting is taking place in any image seed '
                        f'specification.')

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
            raise RenderLoopConfigError(e)

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
        if _pipelinewrapper.model_type_is_sd3(self.model_type):
            if not parsed.is_single_spec:
                # Only simple img2img and inpaint are supported
                if parsed.control_images or parsed.adapter_images:
                    raise RenderLoopConfigError(
                        f'{a_namer("image_seeds")} configurations other than plain img2img and '
                        f'inpaint are currently not supported for {a_namer("model_type")} '
                        f'{_pipelinewrapper.get_model_type_string(self.model_type)}')

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
        if self.model_type == _pipelinewrapper.ModelType.TORCH_IFS_IMG2IMG or \
                (parsed.mask_images and _pipelinewrapper.model_type_is_floyd_ifs(self.model_type)):

            if not parsed.floyd_image:
                raise RenderLoopConfigError(
                    f'You must specify a floyd image with the floyd argument '
                    f'IE: "my-seed.png;{mask_part}floyd=previous-stage-image.png" '
                    f'in your {a_namer("image_seeds")} "{uri}" to disambiguate this '
                    f'usage of Deep Floyd IF super-resolution.')

    def _normalized_schedulers(self) -> typing.Tuple[typing.List[typing.Optional[_types.Uri]], typing.List[typing.Optional[_types.Uri]]]:
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
            raise RenderLoopConfigError(e)

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
            self.second_model_deep_cache_intervals,
            self.second_model_deep_cache_branch_ids
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
                hi_diffusion=ov('hi_diffusion', [self.hi_diffusion]),
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
                second_model_deep_cache=ov('second_model_deep_cache', [self.second_model_deep_cache]),
                second_model_deep_cache_interval=ov('second_model_deep_cache_interval', 
                                                    self.second_model_deep_cache_intervals),
                second_model_deep_cache_branch_id=ov('second_model_deep_cache_branch_id', 
                                                     self.second_model_deep_cache_branch_ids),
                pag_scale=ov('pag_scale', self.pag_scales),
                pag_adaptive_scale=ov('pag_adaptive_scale', self.pag_adaptive_scales),
                image_guidance_scale=ov('image_guidance_scale', self.image_guidance_scales),
                guidance_rescale=ov('guidance_rescale', self.guidance_rescales),
                inference_steps=ov('inference_steps', self.inference_steps),
                sdxl_high_noise_fraction=ov('sdxl_high_noise_fraction', self.sdxl_high_noise_fractions),
                second_model_inference_steps=ov('second_model_inference_steps', self.second_model_inference_steps),
                second_model_guidance_scale=ov('second_model_guidance_scale', self.second_model_guidance_scales),
                sdxl_refiner_hi_diffusion=ov('sdxl_refiner_hi_diffusion', [self.sdxl_refiner_hi_diffusion]),
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
                adetailer_index_filter=ov('adetailer_index_filter', [self.adetailer_index_filter]),
                adetailer_mask_shape=ov('adetailer_mask_shape', self.adetailer_mask_shapes),
                adetailer_detector_padding=ov('adetailer_detector_padding', self.adetailer_detector_paddings),
                adetailer_mask_padding=ov('adetailer_mask_padding', self.adetailer_mask_paddings),
                adetailer_mask_blur=ov('adetailer_mask_blur', self.adetailer_mask_blurs),
                adetailer_mask_dilation=ov('adetailer_mask_dilation', self.adetailer_mask_dilations)):
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


