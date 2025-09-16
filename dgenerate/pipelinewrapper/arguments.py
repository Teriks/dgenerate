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

    latents_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing raw input latents before pipeline execution.
    
    These processors are applied to latents provided through the :py:attr:`DiffusionArguments.latents` 
    argument (raw latents used as noise initialization). The processors are applied in sequence
    before the latents are passed to the diffusion pipeline.
    """

    latents_post_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing output latents when outputting to latents.
    
    These processors are applied to latents when :py:attr:`DiffusionArguments.output_latents` is `True`. 
    The processors are applied in sequence after the diffusion pipeline generates the latents 
    but before they are returned in the result.
    """

    img2img_latents_processors: _types.OptionalUris = None
    """
    One or more latents processor URI strings for processing ``img2img`` latents before pipeline execution.
    
    These processors are applied to latent tensors provided through the :py:attr:`DiffusionArguments.images` 
    argument when doing ``img2img`` with tensor inputs. The processors are applied in sequence and may occur 
    before VAE decoding (for models that decode ``img2img`` latents) or before direct pipeline usage.
    """

    decoded_latents_image_processor_uris: _types.OptionalUris = None
    """
    One or more image processor URI strings for processing images decoded from incoming latents.
    
    These processors are applied to images that are decoded from latent tensors provided through the 
    :py:attr:`DiffusionArguments.images` argument when doing ``img2img`` with tensor inputs. The processors 
    are applied in sequence after VAE decoding but before the images are used in the pipeline.
    
    The processing flow is: decoded images → pre-resize processing → resize to user dimensions → post-resize processing.
    This allows both preprocessing the raw decoded images and postprocessing after they are resized to the target dimensions.
    
    If no processors are specified, images are simply resized to user dimensions without additional processing.
    """

    vae_slicing: bool = False
    """
    Enable VAE slicing?
    """

    vae_tiling: bool = False
    """
    Enable VAE tiling?
    """

    images: _types.OptionalImagesOrTensors = None
    """
    Images or tensors for img2img operations, or the base for inpainting operations.
    
    All inputs must be either PIL Images or torch Tensors - mixing both types in the same sequence is not supported.
    
    When tensors are provided, they represent latent space data and bypass VAE encoding.
    Tensor inputs cannot be resized or processed with image processors.
    
    All input images involved in a generation except for ``ip_adapter_images`` must match in dimension,
    except in the case of Stable Cascade, which can accept multiple images of any dimension for the purpose of image based
    prompting similar to IP Adapters.
    
    All incoming images will be aligned by 8 automatically, if they need to be aligned by
    a value higher than this, a warning will be issued.
    
    All other pipelines interpret multiple image inputs as a batching request.
    """

    latents: _types.OptionalTensors = None
    """
    Noisy latents to serve as a starting point for generation, this should be a list of tensors
    in the format ``[C, H, W]`` or ``[B, C, H, W]``, A list of tensors
    with a batch dimension will be concatenated intelligently.
    """

    mask_images: _types.OptionalImages = None
    """
    Mask images for inpainting operations.
    
    The amount of img2img ``images`` must be equal to the amount of ``mask_images`` supplied.
    
    Note: Mask images are always PIL Images, tensor masks are not supported.
    
    All input images involved in a generation except for ``ip_adapter_images`` must match in dimension.
    
    All incoming mask images will be aligned by 8 automatically, if they need to be aligned by a value
    higher than this, a warning will be issued to ``stdout`` via :py:mod:`dgenerate.messages`.
    """

    inpaint_crop: bool = False
    """
    Enable cropping to mask bounds for inpainting. When enabled, input images will be
    automatically cropped to the bounds of their masks (plus any padding) before processing, 
    then the generated result will be pasted back onto the original uncropped image. This 
    allows inpainting at higher effective resolutions for better quality results.
    
    Batching Behavior:
    
    - Cannot be used with multiple input images/masks in the same call
    - Each image/mask pair must be processed individually as different masks may have different crop bounds
    - However, ``batch_size`` > 1 is supported for generating multiple variations of a single crop
    - Multiple images require separate pipeline calls, not batch processing
    
    Auto-enabling:
    
    This is automatically enabled when :py:attr:`DiffusionArguments.inpaint_crop_padding`, 
    :py:attr:`DiffusionArguments.inpaint_crop_feather`, or :py:attr:`DiffusionArguments.inpaint_crop_masked` 
    are specified.
    """

    inpaint_crop_padding: int | tuple[int, int] | tuple[int, int, int, int] | None = None
    """
    Padding values to use around mask bounds when inpaint_crop is enabled.
    
    Supported formats:
    
    - int: Same padding on all sides
    - tuple[int, int]: (horizontal, vertical) padding  
    - tuple[int, int, int, int]: (left, top, right, bottom) padding
    
    Specifying this value automatically enables :py:attr:`DiffusionArguments.inpaint_crop` if it is not already enabled.
    Default value when :py:attr:`DiffusionArguments.inpaint_crop` is enabled but no padding is specified: 32 pixels.
    
    Note: Inpaint crop cannot be used with multiple input images. See :py:attr:`DiffusionArguments.inpaint_crop` for batching details.
    """

    inpaint_crop_masked: bool = False
    """
    When inpaint_crop is enabled, use the mask when pasting the generated result back 
    onto the original image. This means only the masked areas will be replaced. 
    Cannot be used together with :py:attr:`DiffusionArguments.inpaint_crop_feather`.
    
    Specifying this value automatically enables :py:attr:`DiffusionArguments.inpaint_crop` if it is not already enabled.
    
    Note: Inpaint crop cannot be used with multiple input images. See :py:attr:`DiffusionArguments.inpaint_crop` for batching details.
    """

    inpaint_crop_feather: int | None = None
    """
    Feather value to use when pasting the generated result back onto the original 
    image when :py:attr:`DiffusionArguments.inpaint_crop` is enabled. Feathering creates smooth transitions from 
    opaque to transparent. Cannot be used together with :py:attr:`DiffusionArguments.inpaint_crop_masked`.
    
    Specifying this value automatically enables :py:attr:`DiffusionArguments.inpaint_crop` if it is not already enabled.
    
    Note: Inpaint crop cannot be used with multiple input images. See :py:attr:`DiffusionArguments.inpaint_crop` for batching details.
    """

    control_images: _types.OptionalImages = None
    """
    ControlNet guidance images to use if ``controlnet_uris`` were given to the 
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    Note: Control images must be PIL Images, tensors are not supported since ControlNet/T2I-Adapter 
    operate in pixel space.
    
    All input images involved in a generation must match in dimension.
    
    All incoming ControlNet images will be aligned by 8 automatically, if they need to be 
    aligned by a value higher than this, a warning will be issued to ``stdout`` via :py:mod:`dgenerate.messages`.
    """

    ip_adapter_images: _types.OptionalImagesSequence = None
    """
    IP Adapter images to use if ``ip_adapter_uris`` were given to the
    constructor of :py:class:`.DiffusionPipelineWrapper`.
    
    Note: IP Adapter images must be PIL Images, tensors are not supported since IP-Adapter 
    operates in pixel space.
    
    This should be a list of ``Sequence[PIL.Image]``
    
    Each list entry corresponds to an IP adapter URI.
    
    Multiple IP Adapter URIs can be provided, each IP Adapter can get its own set of images.
    
    All incoming IP Adapter images will be aligned by 8 automatically, if they need to be 
    aligned by a value higher than this, a warning will be issued to ``stdout`` via :py:mod:`dgenerate.messages`.
    """

    floyd_image: _types.OptionalImageOrTensor = None
    """
    The output image or tensor of the last stage when performing img2img or 
    inpainting generation with Deep Floyd. When performing txt2img 
    generation :py:attr:`DiffusionArguments.image` is used.
    
    When a tensor is provided, it represents latent space data from a previous Floyd stage.
    
    Incoming floyd images will be automatically aligned by 8.
    """

    width: _types.OptionalInteger = None
    """
    Output image width.
    
    Will be automatically aligned by 8.
    
    If alignments of more than 8 need to be forced, 
    a warning will be issued to ``stdout`` via :py:mod:`dgenerate.messages`.
    """

    height: _types.OptionalInteger = None
    """
    Output image height.
    
    Will be automatically aligned by 8.
    
    If alignments of more than 8 need to be forced, 
    a warning will be issued to ``stdout`` via :py:mod:`dgenerate.messages`.
    """

    aspect_correct: bool = False
    """
    When resizing input images according to :py:attr:`DiffusionArguments.width` 
    and :py:attr:`DiffusionArguments.height`, should the resize be aspect correct?
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
    Upscaler noise level for the :py:attr:`dgenerate.pipelinewrapper.ModelType.UPSCALER_X4` model type only.
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
    Override the default amount of inference steps performed by the SDXL refiner or Stable Cascade decoder.
    """

    second_model_guidance_scale: _types.OptionalFloat = None
    """
    Override the guidance scale used by the SDXL refiner or Stable Cascade decoder.
    """

    sdxl_refiner_sigmas: collections.abc.Sequence[float] | str | None = None
    """
    Sigma values, this is supported when using a 
    :py:attr:`DiffusionArguments.second_model_scheduler_uri` that supports 
    setting sigmas.
    
    These sigma values control the noise schedule specifically for the SDXL 
    refiner's diffusion process, allowing for customized denoising behavior
    during the refinement stage. This can be particularly useful for fine-tuning
    the level of detail and quality in the refined image.
    
    Format: A list of floating point values in descending order, typically ranging 
    from higher values (more noise) to lower values (less noise).
    
    Or: a string expression involving sigmas from the selected scheduler such as ``sigmas * 0.95``,
    sigmas will be represented as a numpy array, numpy is available through the namespace ``np``, 
    this uses asteval.
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

    sigmas: collections.abc.Sequence[float] | str | None = None
    """
    Sigma values, this is supported when using a when using 
    a :py:attr:`DiffusionArguments.scheduler_uri` that supports setting sigmas.
    
    Sigma values control the noise schedule in the diffusion process, allowing for 
    fine-grained control over how noise is added and removed during image generation.
    Custom sigma values can be used to achieve specific artistic effects or to 
    optimize the generation process for particular types of images.
    
    Format: A list of floating point values in descending order, typically ranging 
    from higher values (more noise) to lower values (less noise).
    
    Or: a string expression involving sigmas from the selected scheduler such as ``sigmas * 0.95``,
    sigmas will be represented as a numpy array, numpy is available through the namespace ``np``, 
    this uses asteval.
    """

    freeu_params: typing.Optional[tuple[float, float, float, float]] = None
    """
    FreeU is a technique for improving image quality by re-balancing the contributions from 
    the UNet's skip connections and backbone feature maps.
    
    This can be used with no cost to performance, to potentially improve image quality.
    
    This argument can be used to specify The FreeU parameters: s1, s2, b1, and b2 in that order.
    
    This argument only applies to models that utilize a UNet: SD1.5/2, SDXL, and Kolors
    
    See: https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
    
    And: https://github.com/ChenyangSi/FreeU
    """

    sdxl_refiner_freeu_params: typing.Optional[tuple[float, float, float, float]] = None
    """
    FreeU parameters for the SDXL refiner
    
    See: :py:attr:`DiffusionArguments.freeu_params` for clarification.
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

    sada_max_downsample: _types.OptionalInteger = None
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
    
    SADA is not compatible with HiDiffusion, DeepCache, or TeaCache.
    """

    sada_sx: _types.OptionalInteger = None
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
    """

    sada_sy: _types.OptionalInteger = None  
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
    """

    sada_acc_range: _types.OptionalRange = None
    """
    SADA acceleration range start / end step for the primary model.
    
    Defines the starting step for SADA acceleration. 
    
    Starting step must be at least 3 as SADA leverages third-order dynamics.
    
    Defaults to [10,47].
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    """

    sada_lagrange_term: _types.OptionalInteger = None
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
    """

    sada_lagrange_int: _types.OptionalInteger = None
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
    """

    sada_lagrange_step: _types.OptionalInteger = None
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
    """

    sada_max_fix: _types.OptionalInteger = None
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
    """

    sada_max_interval: _types.OptionalInteger = None
    """
    SADA maximum interval for optimization for the primary model.
    
    Maximum interval between optimizations in the SADA algorithm.
    
    Defaults to 4.
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    Supplying any SADA parameter implies that SADA is enabled.
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
    """

    sada: bool = False
    """
    Enable SADA (Stability-guided Adaptive Diffusion Acceleration) with default parameters for the primary model.
    
    This is equivalent to setting all SADA parameters to their default values.
    
    See: https://github.com/Ting-Justin-Jiang/sada-icml
    
    This is supported for: ``--model-type sd, sdxl, kolors, flux*``.
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
    
    This is supported for: ``--model-type flux*``.
    
    """

    tea_cache_rel_l1_threshold: _types.OptionalFloat = None
    """
    TeaCache relative L1 threshold when :py:attr:`DiffusionArguments.tea_cache` is enabled.
    
    Higher values mean more speedup.
    
    Defaults to 0.6 (2.0x speedup). 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 
    0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    
    See: https://github.com/ali-vilab/TeaCache
    
    Supplying any value implies that :py:attr:`DiffusionArguments.tea_cache` is enabled.
    
    This is supported for: ``--model-type flux*``.
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
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``, (but not for SD3.5 models)
    """

    ras_sample_ratio: _types.OptionalFloat = None
    """
    Average sample ratio for each RAS step.
    
    For instance, setting this to 0.5 on a sequence of 4096 tokens will result in 
    the noise of averagely 2048 tokens to be updated during each RAS step. Must be between 0.0 and 1.0.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_high_ratio: _types.OptionalFloat = None
    """
    Ratio of high value tokens to be cached in RAS.
    
    Based on the metric selected, the ratio of the high value chosen to be cached.
    Default value is 1.0, but can be set between 0.0 and 1.0 to balance the sample ratio
    between the main subject and the background.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_starvation_scale: _types.OptionalFloat = None
    """
    Starvation scale for RAS patch selection.
    
    RAS tracks how often a token is dropped and incorporates this count as a scaling factor in the
    metric for selecting tokens. This scale factor prevents excessive blurring or noise in the 
    final generated image. Larger scaling factor will result in more uniform sampling.
    Usually set between 0.0 and 1.0.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_error_reset_steps: _types.OptionalIntegers = None
    """
    Dense sampling steps to reset accumulated error in RAS.
    
    The dense sampling steps inserted between the RAS steps to reset the accumulated error.
    A list of step numbers, e.g. [12, 22].
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_start_step: _types.OptionalInteger = None
    """
    Starting step for RAS (Region-Adaptive Sampling).
    
    This controls when RAS begins applying its sampling strategy. 
    Must be greater than or equal to 1.
    
    Defaults to 4 if not specified.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_end_step: _types.OptionalInteger = None
    """
    Ending step for RAS (Region-Adaptive Sampling).
    
    This controls when RAS stops applying its sampling strategy. 
    Must be greater than or equal to 1.
    
    Defaults to the number of inference steps if not specified.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
    """

    ras_metric: _types.OptionalString = None
    """
    Metric to use for RAS (Region-Adaptive Sampling).
    
    This controls how RAS measures the importance of tokens for caching.
    Valid values are "std" (standard deviation) or "l2norm" (L2 norm).
    Defaults to "std".
    
    Supplying any value implies that :py:attr:`DiffusionArguments.ras` is enabled.
    
    This is supported for: ``--model-type sd3``.
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
    
    This is supported for: ``--model-type sd3``.
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
    
    This is supported for: ``--model-type sd3``.
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
    ``model_type`` values ``sd`` and ``sdxl``, including with ``controlnet_uris`` defined.
    """

    sdxl_refiner_clip_skip: _types.OptionalInteger = None
    """
    Clip skip override value for the SDXL refiner, which normally defaults to that of 
    :py:attr:`.DiffusionArguments.clip_skip` when it is defined.
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

    adetailer_model_masks: _types.OptionalBoolean = None
    """
    Indicates that masks generated by the model itself should be preferred over 
    masks generated from the detection bounding box. If this is ``True``, and the model itself
    returns mask data, :py:attr:`DiffusionArguments.adetailer_mask_shape`, :py:attr:`DiffusionArguments.adetailer_mask_padding`, 
    and :py:attr:`DiffusionArguments.adetailer_detector_padding` will all be ignored.
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

    adetailer_size: _types.OptionalInteger = None
    """
    Target size for processing detected areas.
    When specified, detected areas will always be scaled to this target size (with aspect ratio preserved)
    for processing, then scaled back to the original size for compositing.
    This can significantly improve detail quality for small detected features like faces or hands,
    or reduce processing time for overly large detected areas.
    The scaling is based on the larger dimension (width or height) of the detected area.
    The optimal resampling method is automatically selected for both upscaling and downscaling.
    Must be an integer greater than 1. Defaults to none (process at native resolution).
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
    
    This value must be greater than zero.
    
    This is supported for Stable Diffusion, Stable Diffusion XL,
    Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.deep_cache` is enabled.
    
    Defaults to 5.
    """

    deep_cache_branch_id: _types.OptionalInteger = None
    """
    Controls which branch ID DeepCache should operate on in the UNet 
    for the main model. 
    
    This value must be greater than or equal to 0.
    
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
    
    This value must be greater than zero.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.sdxl_refiner_deep_cache` is enabled.
    
    Defaults to 5.
    """

    sdxl_refiner_deep_cache_branch_id: _types.OptionalInteger = None
    """
    Controls which branch ID DeepCache should operate on in the UNet 
    for the SDXL Refiner.
    
    This value must be greater than or equal to 0.
    
    This is supported for Stable Diffusion XL and Kolors based models.
    
    Supplying any value implies that :py:attr:`DiffusionArguments.sdxl_refiner_deep_cache` is enabled.
    
    Defaults to 1.
    """

    output_latents: bool = False
    """
    Whether to output raw latent tensors instead of decoded PIL Images.
    
    When ``True``, the pipeline will return raw latent tensors instead of decoded images.
    This is useful for saving latent representations or for chaining multiple pipeline operations.
    
    Defaults to False (outputs PIL Images).
    """

    denoising_start: _types.OptionalFloat = None
    """
    Denoising should start at this fraction of total timesteps (0.0 to 1.0).
    
    This is useful continuing denoising on noisy latents generated with :py:attr:`DiffusionArguments.denoising_end`
    
    Scheduler Compatibility:
    
    * SD 1.5 models: Only stateless schedulers are supported (``EulerDiscreteScheduler``, 
      ``LMSDiscreteScheduler``, ``EDMEulerScheduler``, ``DPMSolverMultistepScheduler``, 
      ``DDIMScheduler``, ``DDPMScheduler``, ``PNDMScheduler``)
    * SDXL models: All schedulers supported via native denoising_start/denoising_end
    * SD3/Flux models: FlowMatchEulerDiscreteScheduler and standard schedulers supported
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
    """

    @staticmethod
    def prompt_embedded_arg_checker(name: str, value: typing.Any):
        """
        Checks if a class member / value is forbidden to use
        with a prompt embedded argument specification.

        :param name: the argument name
        :param value: the argument value
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

        def format_freeu(params: tuple):
            if params is not None:
                return f"s1={params[0]}, s2={params[1]}, b1={params[2]}, b2={params[3]}"
            return None

        def format_size(size: tuple):
            if size is not None:
                return _textprocessing.format_size(size)
            return None

        descriptions = [
            (self.scheduler_uri, "Scheduler:"),
            (self.second_model_scheduler_uri, "Second Model Scheduler:"),
            (self.seed, "Seed:"),
            (self.clip_skip, "Clip Skip:"),
            (self.sdxl_refiner_clip_skip, "SDXL Refiner Clip Skip:"),
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.inpaint_crop_padding, "Inpaint Crop Padding:"),
            (self.inpaint_crop_feather, "Inpaint Crop Feather:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.second_model_inference_steps, 'Second Model Inference Steps:'),
            (self.second_model_guidance_scale, 'Second Model Guidance Scale:'),
            (format_freeu(self.freeu_params), 'FreeU Params:'),
            (format_freeu(self.sdxl_refiner_freeu_params), "SDXL Refiner FreeU Params:"),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_t2i_adapter_factor, "SDXL T2I Adapter Factor:"),
            (self.sdxl_refiner_pag_scale, 'SDXL Refiner PAG Scale:'),
            (self.sdxl_refiner_pag_adaptive_scale, 'SDXL Refiner PAG Adaptive Scale:'),
            (self.sdxl_refiner_guidance_rescale, "SDXL Refiner Guidance Rescale:"),
            (self.sdxl_refiner_sigmas, "SDXL Refiner Sigmas:"),
            (self.sdxl_aesthetic_score, "SDXL Aesthetic Score:"),
            (format_size(self.sdxl_original_size), "SDXL Original Size:"),
            (format_size(self.sdxl_target_size), "SDXL Target Size:"),
            (format_size(self.sdxl_crops_coords_top_left), "SDXL Top Left Crop Coords:"),
            (self.sdxl_negative_aesthetic_score, "SDXL Negative Aesthetic Score:"),
            (format_size(self.sdxl_negative_original_size), "SDXL Negative Original Size:"),
            (format_size(self.sdxl_negative_target_size), "SDXL Negative Target Size:"),
            (format_size(self.sdxl_negative_crops_coords_top_left), "SDXL Negative Top Left Crop Coords:"),
            (self.sdxl_refiner_aesthetic_score, "SDXL Refiner Aesthetic Score:"),
            (format_size(self.sdxl_refiner_original_size), "SDXL Refiner Original Size:"),
            (format_size(self.sdxl_refiner_target_size), "SDXL Refiner Target Size:"),
            (format_size(self.sdxl_refiner_crops_coords_top_left), "SDXL Refiner Top Left Crop Coords:"),
            (self.sdxl_refiner_negative_aesthetic_score, "SDXL Refiner Negative Aesthetic Score:"),
            (format_size(self.sdxl_refiner_negative_original_size), "SDXL Refiner Negative Original Size:"),
            (format_size(self.sdxl_refiner_negative_target_size), "SDXL Refiner Negative Target Size:"),
            (format_size(self.sdxl_refiner_negative_crops_coords_top_left),
             "SDXL Refiner Negative Top Left Crop Coords:"),
            (self.pag_scale, "PAG Scale:"),
            (self.pag_adaptive_scale, "PAG Adaptive Scale:"),
            (self.guidance_scale, "Guidance Scale:"),
            (self.sigmas, "Sigmas:"),
            (self.tea_cache_rel_l1_threshold, "TeaCache Relative L1 Threshold:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:"),
            (self.adetailer_class_filter, "Adetailer Class Filter:"),
            (self.adetailer_index_filter, "Adetailer Index Filter:"),
            (self.adetailer_mask_shape, "Adetailer Mask Shape:"),
            (format_size(self.adetailer_detector_padding), "Adetailer Detector Padding:"),
            (format_size(self.adetailer_mask_padding), "Adetailer Mask Padding:"),
            (self.adetailer_mask_blur, "Adetailer Mask Blur:"),
            (self.adetailer_mask_dilation, "Adetailer Mask Dilation:"),
            (self.adetailer_size, "Adetailer Processing Size:"),
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
            (self.sdxl_refiner_deep_cache_branch_id, "SDXL Refiner DeepCache Branch ID:"),
            (self.sada_max_downsample, "SADA Max Downsample:"),
            (self.sada_sx, "SADA SX:"),
            (self.sada_sy, "SADA SY:"),
            (self.sada_acc_range, "SADA Acc Range:"),
            (self.sada_lagrange_term, "SADA Lagrange Term:"),
            (self.sada_lagrange_int, "SADA Lagrange Int:"),
            (self.sada_lagrange_step, "SADA Lagrange Step:"),
            (self.sada_max_fix, "SADA Max Fix:"),
            (self.sada_max_interval, "SADA Max Interval:")
        ]

        if not self.prompt.weighter:
            descriptions.append(
                (self.prompt_weighter_uri, 'Prompt Weighter:'))

        if not self.second_model_prompt or not self.second_model_prompt.weighter:
            descriptions.append(
                (self.second_model_prompt_weighter_uri, 'Second Model Prompt Weighter:'))

        for val, desc in descriptions:
            if val is not None:
                if not isinstance(val, str) and isinstance(val, collections.abc.Iterable):
                    inputs.append(desc + ' ' + repr(list(val)))
                else:
                    inputs.append(desc + ' ' + str(val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format
