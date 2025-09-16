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
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Argument reconstruction functionality for DiffusionPipelineWrapper.

This module provides utilities to reconstruct dgenerate command line arguments
from DiffusionArguments and DiffusionPipelineWrapper state.
"""

import collections.abc
import typing

import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.util as _util
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
from dgenerate.pipelinewrapper.arguments import DiffusionArguments
from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper


# noinspection PyTypeChecker
def reconstruct_dgenerate_opts(
        wrapper: DiffusionPipelineWrapper,
        args: DiffusionArguments | None = None,
        extra_opts: collections.abc.Sequence[
                        tuple[str] | tuple[str, typing.Any]] | None = None,
        omit_device: bool = False,
        shell_quote: bool = True,
        overrides: dict[str, typing.Any] = None
) -> list[tuple[str] | tuple[str, typing.Any]]:
    """
    Reconstruct dgenerate's command line arguments from a particular set of pipeline wrapper call arguments.
    
    This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
    this is because there is not enough information in :py:class:`.DiffusionArguments` to
    accurately reproduce it.

    :param wrapper: DiffusionPipelineWrapper instance to extract configuration from
    :param args: :py:class:`.DiffusionArguments` object to take values from
    :param extra_opts: Extra option pairs to be added to the end of reconstructed options,
        this should be a sequence of tuples of length 1 (switch only) or length 2 (switch with args)
    :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
        make sense to include the device specification. And instead simply fallback to whatever 
        the default device is, which is generally ``cuda``
    :param shell_quote: Shell quote and format the argument values? or return them raw.
    :param overrides: pipeline wrapper keyword arguments, these will override values derived from
        any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
        :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
    :return: List of tuples of length 1 or 2 representing the option
    """
    from .arguments import DiffusionArguments

    copy_args = DiffusionArguments()

    if args is not None:
        copy_args.set_from(args)

    copy_args.set_from(overrides, missing_value_throws=False)

    args = copy_args

    opts = [(wrapper.model_path,),
            ('--model-type', wrapper.model_type_string)]

    if wrapper.original_config:
        opts.append(('--original-config', wrapper.original_config))

    if wrapper.second_model_original_config:
        opts.append(('--second-model-original-config', wrapper.second_model_original_config))

    if wrapper.quantizer_uri:
        opts.append(('--quantizer', wrapper.quantizer_uri))

    if wrapper.second_model_quantizer_uri:
        opts.append(('--second-model-quantizer', wrapper.second_model_quantizer_uri))

    if not omit_device:
        opts.append(('--device', wrapper.device))

    opts.append(('--inference-steps', args.inference_steps))
    opts.append(('--guidance-scales', args.guidance_scale))

    if args.sigmas is not None:
        if isinstance(args.sigmas, str):
            opts.append(('--sigmas', f'expr: {args.sigmas}'))
        else:
            opts.append(('--sigmas', ','.join(map(str, args.sigmas))))

    opts.append(('--seeds', args.seed))

    if wrapper.dtype_string != 'auto':
        opts.append(('--dtype', wrapper.dtype_string))

    if args.batch_size is not None and args.batch_size > 1:
        opts.append(('--batch-size', args.batch_size))

    if args.guidance_rescale is not None:
        opts.append(('--guidance-rescales', args.guidance_rescale))

    if args.image_guidance_scale is not None:
        opts.append(('--image-guidance-scales', args.image_guidance_scale))

    if args.prompt_weighter_uri:
        opts.append(('--prompt-weighter', args.prompt_weighter_uri))

    if args.second_model_prompt_weighter_uri:
        opts.append(('--second-model-prompt-weighter', args.second_model_prompt_weighter_uri))

    if args.prompt is not None:
        opts.append(('--prompts', args.prompt))

    if args.second_prompt is not None:
        opts.append(('--second-prompts', args.second_prompt))

    if args.third_prompt is not None:
        opts.append(('--third-prompts', args.third_prompt))

    if args.second_model_prompt is not None:
        opts.append(('--second-model-prompts', args.second_model_prompt))

    if args.second_model_second_prompt is not None:
        opts.append(('--second-model-second-prompts', args.second_model_second_prompt))

    if args.max_sequence_length is not None:
        opts.append(('--max-sequence-length', args.max_sequence_length))

    if args.clip_skip is not None:
        opts.append(('--clip-skips', args.clip_skip))

    if args.sdxl_refiner_clip_skip is not None:
        opts.append(('--sdxl-refiner-clip-skips', args.sdxl_refiner_clip_skip))

    if wrapper.adetailer_detector_uris:
        opts.append(('--adetailer-detectors', wrapper.adetailer_detector_uris))

    if args.adetailer_model_masks:
        opts.append(('--adetailer-model-masks',))

    if args.adetailer_class_filter:
        opts.append(('--adetailer-class-filter', ' '.join(str(s) for s in args.adetailer_class_filter)))

    if args.adetailer_index_filter is not None:
        opts.append(('--adetailer-index-filter',
                     ' '.join(str(i) for i in args.adetailer_index_filter)))

    if args.adetailer_mask_shape is not None:
        opts.append(('--adetailer-mask-shapes', args.adetailer_mask_shape))

    if args.adetailer_detector_padding is not None:
        opts.append(('--adetailer-detector-paddings',
                     _textprocessing.format_size(args.adetailer_detector_padding)))

    if args.adetailer_mask_padding is not None:
        opts.append(('--adetailer-mask-paddings',
                     _textprocessing.format_size(args.adetailer_mask_padding)))

    if args.adetailer_mask_blur is not None:
        opts.append(('--adetailer-mask-blurs', args.adetailer_mask_blur))

    if args.adetailer_mask_dilation is not None:
        opts.append(('--adetailer-mask-dilations', args.adetailer_mask_dilation))

    if args.adetailer_size is not None:
        opts.append(('--adetailer-sizes', args.adetailer_size))

    if wrapper.adetailer_crop_control_image:
        opts.append(('--adetailer-crop-control-image',))

    # Inpaint crop arguments
    if args.inpaint_crop:
        opts.append(('--inpaint-crop',))

    if (args.inpaint_crop_padding is not None and
        args.inpaint_crop_padding != _constants.DEFAULT_INPAINT_CROP_PADDING):
        opts.append(('--inpaint-crop-paddings',
                     _textprocessing.format_size(args.inpaint_crop_padding)))

    if args.inpaint_crop_masked:
        opts.append(('--inpaint-crop-masked',))

    if args.inpaint_crop_feather is not None:
        opts.append(('--inpaint-crop-feathers', args.inpaint_crop_feather))

    if wrapper.text_encoder_uris:
        opts.append(('--text-encoders', ['+' if x is None else x for x in wrapper.text_encoder_uris]))

    if wrapper.second_model_text_encoder_uris:
        opts.append(('--second-model-text-encoders',
                     ['+' if x is None else x for x in wrapper.second_model_text_encoder_uris]))

    if wrapper.s_cascade_decoder_uri is not None:
        opts.append(('--s-cascade-decoder', wrapper.s_cascade_decoder_uri))

    if wrapper.revision is not None and wrapper.revision != 'main':
        opts.append(('--revision', wrapper.revision))

    if wrapper.variant is not None:
        opts.append(('--variant', wrapper.variant))

    if wrapper.subfolder is not None:
        opts.append(('--subfolder', wrapper.subfolder))

    if wrapper.unet_uri is not None:
        opts.append(('--unet', wrapper.unet_uri))

    if wrapper.second_model_unet_uri is not None:
        opts.append(('--second-model-unet', wrapper.second_model_unet_uri))

    if wrapper.transformer_uri is not None:
        opts.append(('--transformer', wrapper.transformer_uri))

    if wrapper.vae_uri is not None:
        opts.append(('--vae', wrapper.vae_uri))

    if args.vae_tiling:
        opts.append(('--vae-tiling',))

    if args.vae_slicing:
        opts.append(('--vae-slicing',))

    if wrapper.model_cpu_offload:
        opts.append(('--model-cpu-offload',))

    if wrapper.model_sequential_offload:
        opts.append(('--model-sequential-offload',))

    if wrapper.second_model_cpu_offload:
        opts.append(('--second-model-cpu-offload',))

    if wrapper.second_model_sequential_offload:
        opts.append(('--second-model-sequential-offload',))

    if wrapper.sdxl_refiner_uri is not None:
        opts.append(('--sdxl-refiner', wrapper.sdxl_refiner_uri))

    if args.sdxl_refiner_edit:
        opts.append(('--sdxl-refiner-edit',))

    if wrapper.lora_uris:
        opts.append(('--loras', wrapper.lora_uris))

    if wrapper.lora_fuse_scale is not None:
        opts.append(('--lora-fuse-scale', wrapper.lora_fuse_scale))

    if wrapper.image_encoder_uri:
        opts.append(('--image-encoder', wrapper.image_encoder_uri))

    if wrapper.ip_adapter_uris:
        opts.append(('--ip-adapters', wrapper.ip_adapter_uris))

    if wrapper.textual_inversion_uris:
        opts.append(('--textual-inversions', wrapper.textual_inversion_uris))

    if wrapper.controlnet_uris:
        opts.append(('--control-nets', wrapper.controlnet_uris))

    if wrapper.t2i_adapter_uris:
        opts.append(('--t2i-adapters', wrapper.t2i_adapter_uris))

    if args.sdxl_t2i_adapter_factor is not None:
        opts.append(('--sdxl-t2i-adapter-factors', args.sdxl_t2i_adapter_factor))

    if args.scheduler_uri is not None:
        opts.append(('--scheduler', args.scheduler_uri))

    if args.second_model_scheduler_uri is not None:
        if args.second_model_scheduler_uri != args.scheduler_uri:
            opts.append(('--second-model-scheduler', args.second_model_scheduler_uri))

    if args.freeu_params is not None:
        opts.append(('--freeu-params', ','.join(map(str, args.freeu_params))))

    if args.hi_diffusion:
        opts.append(('--hi-diffusion',))

    if args.hi_diffusion_no_win_attn:
        opts.append(('--hi-diffusion-no-win-attn',))

    if args.hi_diffusion_no_raunet:
        opts.append(('--hi-diffusion-no-raunet',))

    if args.sdxl_refiner_freeu_params is not None:
        opts.append(('--sdxl-refiner-freeu-params', ','.join(map(str, args.sdxl_refiner_freeu_params))))

    if args.tea_cache:
        opts.append(('--tea-cache',))

    if args.tea_cache_rel_l1_threshold is not None and \
            args.tea_cache_rel_l1_threshold != _constants.DEFAULT_TEA_CACHE_REL_L1_THRESHOLD:
        opts.append(('--tea-cache-rel-l1-thresholds', args.tea_cache_rel_l1_threshold))

    if args.ras:
        opts.append(('--ras',))

    if args.ras_index_fusion:
        opts.append(('--ras-index-fusion',))

    if args.ras_sample_ratio is not None and \
            args.ras_sample_ratio != _constants.DEFAULT_RAS_SAMPLE_RATIO:
        opts.append(('--ras-sample-ratios', args.ras_sample_ratio))

    if args.ras_high_ratio is not None and \
            args.ras_high_ratio != _constants.DEFAULT_RAS_HIGH_RATIO:
        opts.append(('--ras-high-ratios', args.ras_high_ratio))

    if args.ras_starvation_scale is not None \
            and args.ras_starvation_scale != _constants.DEFAULT_RAS_STARVATION_SCALE:
        opts.append(('--ras-starvation-scales', args.ras_starvation_scale))

    if args.ras_error_reset_steps is not None and \
            args.ras_error_reset_steps != _constants.DEFAULT_RAS_ERROR_RESET_STEPS:
        opts.append(('--ras-error-reset-steps', ','.join(map(str, args.ras_error_reset_steps))))

    if args.ras_metric is not None and \
            args.ras_metric != _constants.DEFAULT_RAS_METRIC:
        opts.append(('--ras-metrics', args.ras_metric))

    if args.ras_start_step is not None and \
            args.ras_start_step != _constants.DEFAULT_RAS_START_STEP:
        opts.append(('--ras-start-steps', args.ras_start_step))

    if args.ras_end_step is not None and \
            args.ras_end_step != args.inference_steps:
        opts.append(('--ras-end-steps', args.ras_end_step))

    if args.ras_skip_num_step is not None and \
            args.ras_skip_num_step != _constants.DEFAULT_RAS_SKIP_NUM_STEP:
        opts.append(('--ras-skip-num-steps', args.ras_skip_num_step))

    if args.ras_skip_num_step_length is not None and \
            args.ras_skip_num_step_length != _constants.DEFAULT_RAS_SKIP_NUM_STEP_LENGTH:
        opts.append(('--ras-skip-num-step-lengths', args.ras_skip_num_step_length))

    if args.deep_cache:
        opts.append(('--deep-cache',))

    if args.deep_cache_interval is not None and \
            args.deep_cache_interval != _constants.DEFAULT_DEEP_CACHE_INTERVAL:
        opts.append(('--deep-cache-intervals', args.deep_cache_interval))

    if args.deep_cache_branch_id is not None and \
            args.deep_cache_branch_id != _constants.DEFAULT_DEEP_CACHE_BRANCH_ID:
        opts.append(('--deep-cache-branch-ids', args.deep_cache_branch_id))

    if args.sdxl_refiner_deep_cache:
        opts.append(('--sdxl-refiner-deep-cache',))

    if args.sdxl_refiner_deep_cache_interval is not None and \
            args.sdxl_refiner_deep_cache_interval != _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL:
        opts.append(('--sdxl-refiner-deep-cache-intervals', args.sdxl_refiner_deep_cache_interval))

    if args.sdxl_refiner_deep_cache_branch_id is not None and \
            args.sdxl_refiner_deep_cache_branch_id != _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID:
        opts.append(('--sdxl-refiner-deep-cache-branch-ids', args.sdxl_refiner_deep_cache_branch_id))

    if args.pag_scale == _constants.DEFAULT_PAG_SCALE \
            and args.pag_adaptive_scale == _constants.DEFAULT_PAG_ADAPTIVE_SCALE:
        opts.append(('--pag',))
    else:
        if args.pag_scale is not None:
            opts.append(('--pag-scales', args.pag_scale))
        if args.pag_adaptive_scale is not None:
            opts.append(('--pag-adaptive-scales', args.pag_adaptive_scale))

    if args.sdxl_refiner_pag_scale == _constants.DEFAULT_SDXL_REFINER_PAG_SCALE and \
            args.sdxl_refiner_pag_adaptive_scale == _constants.DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE:
        opts.append(('--sdxl-refiner-pag',))
    else:
        if args.sdxl_refiner_pag_scale is not None:
            opts.append(('--sdxl-refiner-pag-scales', args.sdxl_refiner_pag_scale))
        if args.sdxl_refiner_pag_adaptive_scale is not None:
            opts.append(('--sdxl-refiner-pag-adaptive-scales', args.sdxl_refiner_pag_adaptive_scale))

    if args.sdxl_high_noise_fraction is not None:
        opts.append(('--sdxl-high-noise-fractions', args.sdxl_high_noise_fraction))

    if args.second_model_inference_steps is not None:
        opts.append(('--second-model-inference-steps', args.second_model_inference_steps))

    if args.second_model_guidance_scale is not None:
        opts.append(('--second-model-guidance-scales', args.second_model_guidance_scale))

    if args.sdxl_refiner_sigmas is not None:
        if isinstance(args.sdxl_refiner_sigmas, str):
            opts.append(('--sdxl-refiner-sigmas',
                         f'expr: {args.sdxl_refiner_sigmas}'))
        else:
            opts.append(('--sdxl-refiner-sigmas',
                         ','.join(map(str, args.sdxl_refiner_sigmas))))

    if args.sdxl_refiner_guidance_rescale is not None:
        opts.append(('--sdxl-refiner-guidance-rescales', args.sdxl_refiner_guidance_rescale))

    if args.sdxl_aesthetic_score is not None:
        opts.append(('--sdxl-aesthetic-scores', args.sdxl_aesthetic_score))

    if args.sdxl_original_size is not None:
        opts.append(('--sdxl-original-sizes', args.sdxl_original_size))

    if args.sdxl_target_size is not None:
        opts.append(('--sdxl-target-sizes', args.sdxl_target_size))

    if args.sdxl_crops_coords_top_left is not None:
        opts.append(('--sdxl-crops-coords-top-left', args.sdxl_crops_coords_top_left))

    if args.sdxl_negative_aesthetic_score is not None:
        opts.append(('--sdxl-negative-aesthetic-scores', args.sdxl_negative_aesthetic_score))

    if args.sdxl_negative_original_size is not None:
        opts.append(('--sdxl-negative-original-sizes', args.sdxl_negative_original_size))

    if args.sdxl_negative_target_size is not None:
        opts.append(('--sdxl-negative-target-sizes', args.sdxl_negative_target_size))

    if args.sdxl_negative_crops_coords_top_left is not None:
        opts.append(('--sdxl-negative-crops-coords-top-left', args.sdxl_negative_crops_coords_top_left))

    if args.sdxl_refiner_aesthetic_score is not None:
        opts.append(('--sdxl-refiner-aesthetic-scores', args.sdxl_refiner_aesthetic_score))

    if args.sdxl_refiner_original_size is not None:
        opts.append(('--sdxl-refiner-original-sizes', args.sdxl_refiner_original_size))

    if args.sdxl_refiner_target_size is not None:
        opts.append(('--sdxl-refiner-target-sizes', args.sdxl_refiner_target_size))

    if args.sdxl_refiner_crops_coords_top_left is not None:
        opts.append(('--sdxl-refiner-crops-coords-top-left', args.sdxl_refiner_crops_coords_top_left))

    if args.sdxl_refiner_negative_aesthetic_score is not None:
        opts.append(('--sdxl-refiner-negative-aesthetic-scores', args.sdxl_refiner_negative_aesthetic_score))

    if args.sdxl_refiner_negative_original_size is not None:
        opts.append(('--sdxl-refiner-negative-original-sizes', args.sdxl_refiner_negative_original_size))

    if args.sdxl_refiner_negative_target_size is not None:
        opts.append(('--sdxl-refiner-negative-target-sizes', args.sdxl_refiner_negative_target_size))

    if args.sdxl_refiner_negative_crops_coords_top_left is not None:
        opts.append(
            ('--sdxl-refiner-negative-crops-coords-top-left', args.sdxl_refiner_negative_crops_coords_top_left))

    # SADA arguments - only include if they differ from model-specific defaults
    model_defaults = _util.get_sada_model_defaults(wrapper.model_type)
    
    # Check if all SADA parameters match their model defaults (treating None as default)
    sada_params_match_defaults = (
        (args.sada_max_downsample is None or args.sada_max_downsample == model_defaults['max_downsample']) and
        (args.sada_sx is None or args.sada_sx == model_defaults['sx']) and
        (args.sada_sy is None or args.sada_sy == model_defaults['sy']) and
        (args.sada_acc_range is None or args.sada_acc_range == model_defaults['acc_range']) and
        (args.sada_lagrange_term is None or args.sada_lagrange_term == model_defaults['lagrange_term']) and
        (args.sada_lagrange_int is None or args.sada_lagrange_int == model_defaults['lagrange_int']) and
        (args.sada_lagrange_step is None or args.sada_lagrange_step == model_defaults['lagrange_step']) and
        (args.sada_max_fix is None or args.sada_max_fix == model_defaults['max_fix']) and
        (args.sada_max_interval is None or args.sada_max_interval == model_defaults['max_interval'])
    )
    
    # Check if any SADA parameter is set (indicating SADA is enabled)
    sada_enabled = (
        args.sada or
        args.sada_max_downsample is not None or
        args.sada_sx is not None or
        args.sada_sy is not None or
        args.sada_acc_range is not None or
        args.sada_lagrange_term is not None or
        args.sada_lagrange_int is not None or
        args.sada_lagrange_step is not None or
        args.sada_max_fix is not None or
        args.sada_max_interval is not None
    )
    
    if sada_enabled:
        if args.sada or sada_params_match_defaults:
            # Use the simple --sada flag if all parameters match defaults
            opts.append(('--sada',))
        else:
            # Include individual parameters that differ from defaults
            if args.sada_max_downsample is not None and args.sada_max_downsample != model_defaults['max_downsample']:
                opts.append(('--sada-max-downsamples', args.sada_max_downsample))
            
            if args.sada_sx is not None and args.sada_sx != model_defaults['sx']:
                opts.append(('--sada-sxs', args.sada_sx))
            
            if args.sada_sy is not None and args.sada_sy != model_defaults['sy']:
                opts.append(('--sada-sys', args.sada_sy))
            
            if args.sada_acc_range is not None and args.sada_acc_range != model_defaults['acc_range']:
                opts.append(('--sada-acc-ranges', ','.join(map(str, args.sada_acc_range))))

            if args.sada_lagrange_term is not None and args.sada_lagrange_term != model_defaults['lagrange_term']:
                opts.append(('--sada-lagrange-terms', args.sada_lagrange_term))
            
            if args.sada_lagrange_int is not None and args.sada_lagrange_int != model_defaults['lagrange_int']:
                opts.append(('--sada-lagrange-ints', args.sada_lagrange_int))
            
            if args.sada_lagrange_step is not None and args.sada_lagrange_step != model_defaults['lagrange_step']:
                opts.append(('--sada-lagrange-steps', args.sada_lagrange_step))
            
            if args.sada_max_fix is not None and args.sada_max_fix != model_defaults['max_fix']:
                opts.append(('--sada-max-fixes', args.sada_max_fix))
            
            if args.sada_max_interval is not None and args.sada_max_interval != model_defaults['max_interval']:
                opts.append(('--sada-max-intervals', args.sada_max_interval))

    if args.width is not None and args.height is not None:
        opts.append(('--output-size', f'{args.width}x{args.height}'))
    elif args.width is not None:
        opts.append(('--output-size', f'{args.width}'))
    
    # aspect_correct defaults to False, so only include --no-aspect when it's explicitly False
    # Note: --no-aspect is the inverse of aspect_correct
    if not args.aspect_correct:
        opts.append(('--no-aspect',))

    # Add image_seed_strength if it differs from default
    if args.image_seed_strength is not None and args.image_seed_strength != _constants.DEFAULT_IMAGE_SEED_STRENGTH:
        opts.append(('--image-seed-strengths', args.image_seed_strength))

    # Add upscaler_noise_level if it differs from model-specific default
    if args.upscaler_noise_level is not None:
        # Determine the appropriate default based on model type
        upscaler_default = None
        if wrapper.model_type == _enums.ModelType.UPSCALER_X4:
            upscaler_default = _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL
        elif wrapper.model_type_string == _enums.ModelType.IFS:
            upscaler_default = _constants.DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL
        elif wrapper.model_type_string == _enums.ModelType.IFS_IMG2IMG:
            upscaler_default = _constants.DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL
        
        # Only include if different from the model-specific default
        if upscaler_default is not None and args.upscaler_noise_level != upscaler_default:
            opts.append(('--upscaler-noise-levels', args.upscaler_noise_level))

    if args.denoising_start is not None:
        opts.append(('--denoising-start', args.denoising_start))

    if args.denoising_end is not None:
        opts.append(('--denoising-end', args.denoising_end))

    if args.latents_processors:
        opts.append(('--latents-processors', args.latents_processors))

    if args.img2img_latents_processors:
        opts.append(('--img2img-latents-processors', args.img2img_latents_processors))

    if args.latents_post_processors:
        opts.append(('--latents-post-processors', args.latents_post_processors))

    if args.decoded_latents_image_processor_uris:
        # these are specified with --seed-image-processors
        opts.append(('--seed-image-processors', args.decoded_latents_image_processor_uris))

    if extra_opts is not None:
        for opt in extra_opts:
            opts.append(opt)

    if shell_quote:
        for idx, option in enumerate(opts):
            if len(option) > 1:
                name, value = option
                if isinstance(value, (str, _prompt.Prompt)):
                    opts[idx] = (name, _textprocessing.shell_quote(str(value)))
                elif isinstance(value, tuple):
                    opts[idx] = (name, _textprocessing.format_size(value))
                else:
                    opts[idx] = (name, str(value))
            else:
                solo_val = str(option[0])
                if not solo_val.startswith('-'):
                    # not a solo switch option, some value
                    opts[idx] = (_textprocessing.shell_quote(solo_val),)

    return opts


def _set_opt_value_syntax(val):
    """Helper function to format option values with proper syntax."""
    if isinstance(val, tuple):
        return _textprocessing.format_size(val)
    if isinstance(val, str):
        return _textprocessing.shell_quote(str(val))

    try:
        val_iter = iter(val)
    except TypeError:
        return _textprocessing.shell_quote(str(val))

    return ' '.join(_set_opt_value_syntax(v) for v in val_iter)


def _format_option_pair(val):
    """Helper function to format option pairs for command line output."""
    if len(val) > 1:
        opt_name, opt_value = val

        if isinstance(opt_value, _prompt.Prompt):
            header_len = len(opt_name) + 2
            prompt_text = \
                _textprocessing.wrap(
                    _textprocessing.shell_quote(str(opt_value)),
                    subsequent_indent=' ' * header_len,
                    width=75)

            prompt_text = ' \\\n'.join(prompt_text.split('\n'))

            if '\n' in prompt_text:
                # need to escape the comment token
                prompt_text = prompt_text.replace('#', r'\#')

            return f'{opt_name} {prompt_text}'

        return f'{opt_name} {_set_opt_value_syntax(opt_value)}'

    solo_val = str(val[0])

    if solo_val.startswith('-'):
        return solo_val

    # Not a switch option, some value
    return _textprocessing.shell_quote(solo_val)


def gen_dgenerate_config(
        wrapper: DiffusionPipelineWrapper,
        args: DiffusionArguments | None = None,
        extra_opts: collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
        extra_comments: collections.abc.Iterable[str] | None = None,
        omit_device: bool = False,
        overrides: dict[str, typing.Any] = None
) -> str:
    """
    Generate a valid dgenerate config file with a single invocation that reproduces the 
    arguments associated with :py:class:`.DiffusionArguments`.
    
    This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
    this is because there is not enough information in :py:class:`.DiffusionArguments` to
    accurately reproduce it.

    :param wrapper: DiffusionPipelineWrapper instance to extract configuration from
    :param args: :py:class:`.DiffusionArguments` object to take values from
    :param extra_opts: Extra option pairs to be added to the end of reconstructed options
        of the dgenerate invocation, this should be a sequence of tuples of length 1 (switch only)
        or length 2 (switch with args)
    :param extra_comments: Extra strings to use as comments after the initial
        version check directive
    :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
        make sense to include the device specification. And instead simply fallback to whatever 
        the default device is, which is generally ``cuda``
    :param overrides: pipeline wrapper keyword arguments, these will override values derived from
        any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
        :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
    :return: The configuration as a string
    """
    from dgenerate import __version__

    config = f'#! /usr/bin/env dgenerate --file\n#! dgenerate {__version__}\n\n'

    if extra_comments:
        wrote_comments = False
        for comment in extra_comments:
            wrote_comments = True
            for part in comment.split('\n'):
                config += '# ' + part.rstrip()

        if wrote_comments:
            config += '\n\n'

    opts = reconstruct_dgenerate_opts(
        wrapper=wrapper,
        args=args,
        overrides=overrides,
        shell_quote=False,
        omit_device=omit_device)

    if extra_opts is not None:
        for opt in extra_opts:
            opts.append(opt)

    for opt in opts[:-1]:
        config += f'{_format_option_pair(opt)} \\\n'

    last = opts[-1]

    return config + _format_option_pair(last)


def gen_dgenerate_command(wrapper: DiffusionPipelineWrapper,
                          args: DiffusionArguments | None = None,
                          extra_opts: collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                          omit_device: bool = False,
                          overrides: dict[str, typing.Any] = None) -> str:
    """
    Generate a valid dgenerate command line invocation that reproduces the 
    arguments associated with :py:class:`.DiffusionArguments`.
    
    This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
    this is because there is not enough information in :py:class:`.DiffusionArguments` to
    accurately reproduce it.

    :param wrapper: DiffusionPipelineWrapper instance to extract configuration from
    :param args: :py:class:`.DiffusionArguments` object to take values from
    :param extra_opts: Extra option pairs to be added to the end of reconstructed options
        of the dgenerate invocation, this should be a sequence of tuples of length 1 (switch only)
        or length 2 (switch with args)
    :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
        make sense to include the device specification. And instead simply fallback to whatever 
        the default device is, which is generally ``cuda``
    :param overrides: pipeline wrapper keyword arguments, these will override values derived from
        any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
        :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
    :return: A string containing the dgenerate command line needed to reproduce this result.
    """
    opt_string = \
        ' '.join(
            f"{_format_option_pair(opt)}"
            for opt in reconstruct_dgenerate_opts(
                wrapper=wrapper,
                args=args,
                overrides=overrides,
                extra_opts=extra_opts,
                omit_device=omit_device,
                shell_quote=False))

    return f'dgenerate {opt_string}'
