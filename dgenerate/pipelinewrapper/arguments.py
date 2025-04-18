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

import typing
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class DiffusionArguments(_types.SetFromMixin):
    """
    Represents all possible arguments for a :py:class:`.DiffusionPipelineWrapper` call.
    """
    prompt: _prompt.OptionalPrompt = None
    """
    Primary prompt
    """

    second_prompt: _prompt.OptionalPrompt = None
    """
    Secondary Prompt for SDXL, SD3, Flux.
    """

    second_model_prompt: _prompt.OptionalPrompt = None
    """
    Primary prompt for SDXL Refiner or Stable Cascade Decoder.
    """

    second_model_second_prompt: _prompt.OptionalPrompt = None
    """
    Secondary Prompt for SDXL Refiner, the Stable Cascade Decoder does not support this argument.
    """

    third_prompt: _prompt.OptionalPrompt = None
    """
    Tertiary Prompt for SD3.
    """

    scheduler_uri: _types.OptionalUri = None
    """
    Primary model scheduler URI
    """

    second_model_scheduler_uri: _types.OptionalUri = None
    """
    SDXL refiner scheduler / Stable Cascade Decoder URI, if not specified, defaults to :py:attr:`DiffusionArguments.scheduler`
    """

    prompt_weighter_uri: _types.OptionalUri = None
    """
    Default prompt weighter plugin to use for all models.
    """

    second_model_prompt_weighter_uri: _types.OptionalUri = None
    """
    The URI of a prompt-weighter implementation supported by dgenerate 
    to use with the SDXL refiner or Stable Cascade Decoder.
    
    Defaults to :py:attr:`DiffusionArguments.prompt_weighter_uri` if not specified.
    
    This corresponds to the ``--second-model-prompt-weighter`` argument of the dgenerate command line tool.
    """

    vae_slicing: bool = False
    """
    Enable VAE slicing?
    """

    vae_tiling: bool = False
    """
    Enable VAE tiling?
    """

    images: _types.OptionalImages = None
    """
    Images for img2img operations, or the base for inpainting operations.
    
    All input images involved in a generation except for ``adapter_images`` must match in dimension and be aligned by 8 pixels,
    except in the case of Stable Cascade, which can accept multiple images of any dimension for the purpose of image based
    prompting similar to IP Adapters.
    
    All other pipelines interpret multiple image inputs as a batching request.
    """

    mask_images: _types.OptionalImages = None
    """
    Mask images for inpainting operations.
    
    The amount of img2img ``images`` must be equal to the amount of ``mask_images`` supplied.
    
    All input images involved in a generation except for ``adapter_images``  must match in dimension and be aligned by 8 pixels,
    except in the case of Stable Cascade, which can accept multiple images of any dimension for the purpose of image based
    prompting similar to IP Adapters.  Stable Cascade cannot perform inpainting, so ``mask_images`` is irrelevant in
    this case. All other pipelines interpret multiple image inputs as a batching request.
    """

    control_images: _types.OptionalImages = None
    """
    ControlNet guidance images to use if ``controlnet_uris`` were given to the 
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    All input images involved in a generation must match in dimension and be aligned by 8 pixels.
    """

    ip_adapter_images: _types.OptionalImagesSequence = None
    """
    IP Adapter images to use if ``ip_adapter_uris`` were given to the
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    This should be a list of ``Sequence[PIL.Image]``
    
    Each list entry corresponds to an IP adapter URI.
    
    Multiple IP Adapter URIs can be provided, each IP Adapter can get its own set of images.
    """

    floyd_image: _types.OptionalImage = None
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

    max_sequence_length: _types.OptionalInteger = None
    """
    Max number of prompt tokens that the T5EncoderModel (text encoder 3) of Stable Diffusion 3 or Flux can handle.
    
    This defaults to 256 for SD3 when not specified, and 512 for Flux.
    
    The maximum value is 512 and the minimum value is 1.
    
    High values result in more resource usage and processing time.
    """

    sdxl_refiner_edit: _types.OptionalBoolean = None
    """
    Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode.
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

    second_model_inference_steps: _types.OptionalInteger = None
    """
    Override the default amount of inference steps preformed by the SDXL refiner or Stable Cascade decoder.
    """

    second_model_guidance_scale: _types.OptionalFloat = None
    """
    Override the guidance scale used by the SDXL refiner or Stable Cascade decoder.
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
    
    This is supported for Flux, TeaCache uses a novel caching mechanism 
    in the forward pass of the flux transformer to reduce the amount of
    computation needed to generate an image, this can speed up inference
    with small amounts of quality loss.
    
    See: https://github.com/ali-vilab/TeaCache
    
    Also see: :py:attr:`DiffusionArguments.tea_cache_rel_l1_threshold`
    
    This is supported for: ``--model-type torch-flux*``.
    
    """

    tea_cache_rel_l1_threshold: _types.OptionalFloat = None
    """
    TeaCache relative L1 threshold when :py:attr:`DiffusionArguments.tea_cache` is enabled.
    
    Higher values mean more speedup.
    
    Defaults to 0.6 (2.0x speedup). 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 
    0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    
    See: https://github.com/ali-vilab/TeaCache
    
    Supplying any value implies that :py:attr:`DiffusionArguments.tea_cache` is enabled.
    
    This is supported for: ``--model-type torch-flux*``.
    """

    ras: bool = False
    """
    Activate RAS (Region-Adaptive Sampling) for the primary model? 
            
    This can increase inference speed with SD3.
    
    See: https://github.com/microsoft/ras
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_index_fusion: _types.OptionalBoolean = None
    """
    Enable index fusion in RAS (Region-Adaptive Sampling) for the primary model?
    
    This can improve attention computation in RAS for SD3.
    
    See: https://github.com/microsoft/ras
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``, (but not for SD3.5 models)
    """

    ras_sample_ratio: _types.OptionalFloat = None
    """
    Average sample ratio for each RAS step.
    
    For instance, setting this to 0.5 on a sequence of 4096 tokens will result in 
    the noise of averagely 2048 tokens to be updated during each RAS step. Must be between 0.0 and 1.0.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_high_ratio: _types.OptionalFloat = None
    """
    Ratio of high value tokens to be cached in RAS.
    
    Based on the metric selected, the ratio of the high value chosen to be cached.
    Default value is 1.0, but can be set between 0.0 and 1.0 to balance the sample ratio
    between the main subject and the background.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_starvation_scale: _types.OptionalFloat = None
    """
    Starvation scale for RAS patch selection.
    
    RAS tracks how often a token is dropped and incorporates this count as a scaling factor in the
    metric for selecting tokens. This scale factor prevents excessive blurring or noise in the 
    final generated image. Larger scaling factor will result in more uniform sampling.
    Usually set between 0.0 and 1.0.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_error_reset_steps: _types.OptionalString = None
    """
    Dense sampling steps to reset accumulated error in RAS.
    
    The dense sampling steps inserted between the RAS steps to reset the accumulated error.
    Should be a comma-separated string of step numbers, e.g. "12,22".
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_start_step: _types.OptionalInteger = None
    """
    Starting step for RAS (Region-Adaptive Sampling).
    
    This controls when RAS begins applying its sampling strategy. 
    Must be greater than or equal to 1.
    
    Defaults to 4 if not specified.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_end_step: _types.OptionalInteger = None
    """
    Ending step for RAS (Region-Adaptive Sampling).
    
    This controls when RAS stops applying its sampling strategy. 
    Must be greater than or equal to 1.
    
    Defaults to the number of inference steps if not specified.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_metric: _types.OptionalString = None
    """
    Metric to use for RAS (Region-Adaptive Sampling).
    
    This controls how RAS measures the importance of tokens for caching.
    Valid values are "std" (standard deviation) or "l2norm" (L2 norm).
    Defaults to "std".
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_skip_num_step: _types.OptionalInteger = None
    """
    Skip steps for RAS (Region-Adaptive Sampling). Controls the number of steps to skip between RAS steps.
    The actual number of tokens skipped will be rounded down to the nearest multiple of 64 to ensure
    efficient memory access patterns for attention computation. 
    
    When used with :py:attr:`DiffusionArguments.ras_skip_num_step_length` greater than 0, this 
    value determines how the number of skipped tokens changes over time.
    
    Positive values will increase the number of skipped tokens over time, while negative values will
    decrease it. 
    
    Each value will be tried in turn. 
    
    Supplying any values implies :py:attr:`DiffusionArguments.ras`.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    ras_skip_num_step_length: _types.OptionalInteger = None
    """
    Skip step lengths for RAS (Region-Adaptive Sampling). Controls the length of steps to skip between
    RAS steps. When set to 0, static dropping is used where the number of skipped tokens remains
    constant throughout the generation process. 
    
    When greater than 0, dynamic dropping is enabled where the number of skipped tokens varies over 
    time based on :py:attr:`DiffusionArguments.ras_skip_num_step`. 
    
    The pattern of skipping will repeat every :py:attr:`DiffusionArguments.ras_skip_num_step_length` steps. 
    
    Each value will be tried in turn. 
    
    Supplying any values implies :py:attr:`DiffusionArguments.ras`.
    
    This is supported for: ``--model-type torch-sd3``.
    """

    sdxl_refiner_hi_diffusion: _types.OptionalBoolean = None
    """
    Activate HiDiffusion on the SDXL refiner for this generation?, See: :py:attr:`DiffusionArguments.hi_diffusion`
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

    adetailer_index_filter: _types.OptionalIntegers = None
    """
    A list index values that indicates what YOLO detection indices to keep, 
    the index values start at zero. Detections are sorted by their top left bounding box 
    coordinate from left to right, top to bottom, by (confidence descending). The order of 
    detections in the image is identical to the reading order of words on a page (english). 
    Inpainting will only be preformed on the specified detection indices, if no indices 
    are specified, then inpainting will be preformed on all detections.
    """

    adetailer_mask_shape: _types.OptionalName = None
    """
    This indicates what mask shape adetailer should attempt to draw around a detected feature,
    the default value is "rectangle". You may also specify "circle" to generate an ellipsoid 
    shaped mask, which might be helpful for achieving better blending.
    """

    adetailer_detector_padding: _types.OptionalPadding = None
    """
    This value specifies the amount of padding
    that will be added to the detection rectangle which is used to
    generate a masked area. The default is 0, you can make the mask
    area around the detected feature larger with positive padding
    and smaller with negative padding.
    
    Example:
    
    32 (32px Uniform, all sides)
    
    (10, 20) (10px Horizontal, 20px Vertical)
    
    (10, 20, 30, 40) (10px Left, 20px Top, 30px Right, 40px Bottom)
    
    Defaults to 0.
    """

    adetailer_mask_padding: _types.OptionalPadding = None
    """
    This value indicates how much padding to place around the masked 
    area when cropping out the image to be inpainted, this value must be large
    enough to accommodate any feathering on the edge of the mask caused
    by :py:attr:`.DiffusionArguments.mask_blur` or 
    :py:attr:`.DiffusionArguments.mask_dilation for the best result.
    
    Example:
    
    32 (32px Uniform, all sides)
    
    (10, 20) (10px Horizontal, 20px Vertical)
    
    (10, 20, 30, 40) (10px Left, 20px Top, 30px Right, 40px Bottom)
    
    Defaults to 32.
    """

    adetailer_mask_blur: _types.OptionalInteger = None
    """
    Indicates the level of gaussian blur to apply
    to the inpaint mask generated by adetailer, which can help with 
    smooth blending of the inpainted feature. Defaults to 4.
    """

    adetailer_mask_dilation: _types.OptionalInteger = None
    """
    Indicates the amount of dilation applied to the generated adetailer inpaint mask, 
    see: cv2.dilate. Defaults to 4.
    """

    deep_cache: bool = False
    """
    Enable DeepCache acceleration for the main model? DeepCache caches the intermediate 
    attention layer outputs to speed up the diffusion process. This is beneficial for 
    higher inference steps.
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    """

    deep_cache_interval: _types.OptionalInteger = None
    """
    Controls the frequency of caching intermediate outputs in DeepCache 
    for the main model. 
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.deep_cache` is enabled.
    
    Defaults to 5.
    """

    deep_cache_branch_id: _types.OptionalInteger = None
    """
    Controls which branch ID DeepCache should operate on in the UNet 
    for the main model. 
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.deep_cache` is enabled.
    
    Defaults to 1.
    """

    sdxl_refiner_deep_cache: _types.OptionalBoolean = None
    """
    Enable DeepCache acceleration for the SDXL Refiner?
    
    This is supported for Stable Diffusion XL and Kolors based models.
    """

    sdxl_refiner_deep_cache_interval: _types.OptionalInteger = None
    """
    Controls the frequency of caching intermediate outputs in DeepCache 
    for the SDXL Refiner. 
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.sdxl_refiner_deep_cache` is enabled.
    
    Defaults to 5.
    """

    sdxl_refiner_deep_cache_branch_id: _types.OptionalInteger = None
    """
    Controls which branch ID DeepCache should operate on in the UNet 
    for the SDXL Refiner.
    
    This is supported for Stable Diffusion XL and Kolors based models. 
    
    Supplying any value implies that :py:attr:`DiffusionArguments.sdxl_refiner_deep_cache` is enabled.
    
    Defaults to 1.
    """

    @staticmethod
    def prompt_embedded_arg_checker(name):
        """
        Checks if a class member is forbidden to use
        with a prompt embedded argument specification.

        :param name: the argument name
        """
        return 'prompt' in name

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
    def _describe_prompt(prompt_format, prompt: _prompt.Prompt, pos_title, neg_title, weighter_title=None):
        if prompt is None:
            return

        if weighter_title and prompt.weighter:
            prompt_format.append(f'{weighter_title}: "{prompt.weighter}"')

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
            "Negative Prompt",
            "Prompt Weighter")

        DiffusionArguments._describe_prompt(
            prompt_format, self.second_prompt,
            "Secondary Prompt",
            "Secondary Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.third_prompt,
            "Tertiary Prompt",
            "Tertiary Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.second_model_prompt,
            "Second Model Prompt",
            "Second Model Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.second_model_second_prompt,
            "Second Model Second Prompt",
            "Second Model Second Negative Prompt")

        prompt_format = '\n'.join(prompt_format)
        if prompt_format:
            prompt_format = '\n' + prompt_format

        inputs = []

        descriptions = [
            (self.scheduler_uri, "Scheduler:"),
            (self.second_model_scheduler_uri, "Second Model Scheduler:"),
            (self.seed, "Seed:"),
            (self.clip_skip, "Clip Skip:"),
            (self.sdxl_refiner_clip_skip, "SDXL Refiner Clip Skip:"),
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.second_model_inference_steps, 'Second Model Inference Steps:'),
            (self.second_model_guidance_scale, 'Second Model Guidance Scale:'),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_t2i_adapter_factor, "SDXL T2I Adapter Factor:"),
            (self.sdxl_refiner_pag_scale, 'SDXL Refiner PAG Scale:'),
            (self.sdxl_refiner_pag_adaptive_scale, 'SDXL Refiner PAG Adaptive Scale:'),
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
            (self.tea_cache_rel_l1_threshold, "TeaCache Relative L1 Threshold:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:"),
            (self.adetailer_index_filter, "Adetailer Index Filter:"),
            (self.adetailer_mask_shape, "Adetailer Mask Shape:"),
            (self.adetailer_detector_padding, "Adetailer Detector Padding:"),
            (self.adetailer_mask_padding, "Adetailer Mask Padding:"),
            (self.adetailer_mask_blur, "Adetailer Mask Blur:"),
            (self.adetailer_mask_dilation, "Adetailer Mask Dilation:"),
            (self.ras_sample_ratio, "RAS Sample Ratio:"),
            (self.ras_high_ratio, "RAS High Ratio:"),
            (self.ras_starvation_scale, "RAS Starvation Scale:"),
            (self.ras_error_reset_steps, "RAS Error Reset Steps:"),
            (self.ras_start_step, "RAS Start Step:"),
            (self.ras_end_step, "RAS End Step:"),
            (self.ras_metric, "RAS Metric:"),
            (self.ras_skip_num_step, "RAS Skip Num Step:"),
            (self.ras_skip_num_step_length, "RAS Skip Num Step Length:"),
            (self.deep_cache_interval, "DeepCache Interval:"),
            (self.deep_cache_branch_id, "DeepCache Branch ID:"),
            (self.sdxl_refiner_deep_cache_interval, "SDXL Refiner DeepCache Interval:"),
            (self.sdxl_refiner_deep_cache_branch_id, "SDXL Refiner DeepCache Branch ID:")
        ]

        if not self.prompt.weighter:
            descriptions.append(
                (self.prompt_weighter_uri, 'Prompt Weighter:'))

        if not self.second_model_prompt or not self.second_model_prompt.weighter:
            descriptions.append(
                (self.second_model_prompt_weighter_uri, 'Second Model Prompt Weighter:'))

        for val, desc in descriptions:
            if val is not None:
                if isinstance(val, tuple):
                    inputs.append(desc + ' ' + _textprocessing.format_size(val))
                else:
                    inputs.append(desc + ' ' + str(val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format
