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

"""Known defaults for optional diffusion features.

:func:`apply_known_defaults` fills feature parameters and is also used when
describing a step (without changing the live arguments). Omitted guidance and
inference steps stay unset in step text (**auto**); the pipeline wrapper
resolves them in :meth:`~dgenerate.pipelinewrapper.wrapper.PipelineWrapper._get_pipeline_defaults`
once the model is loaded (video models use
:mod:`dgenerate.pipelinewrapper.videopipelines` instead).

:func:`fill_render_loop_config_defaults` only applies :func:`apply_known_defaults`
to :class:`~dgenerate.renderloopconfig.RenderLoopConfig` list fields (via
:meth:`RenderLoopConfig.fill_defaults`), plus RegionE step resolution. It does
not materialize ``guidance_scales`` or ``inference_steps``.
"""

import inspect

import dgenerate.extras.regione.runtime as _regione
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _util
import dgenerate.types as _types


def _prefixed_value_set(args, prefix: str) -> bool:
    for name, value in args.__dict__.items():
        if not name.startswith(prefix):
            continue
        if isinstance(value, bool):
            if value:
                return True
        elif value is not None:
            return True
    return False


def _feature_on(args, name: str) -> bool:
    if getattr(args, name, False):
        return True
    return _prefixed_value_set(args, name + '_')


def _set_missing(args, name: str, value):
    if getattr(args, name) is None:
        setattr(args, name, value)


def pipeline_config_bool(pipeline, name: str) -> bool | None:
    """A bool stored on a loaded pipeline config, or ``None`` when it is absent.

    A missing key is not the same as false. Krea 2's ``is_distilled`` flag is
    how Turbo is told from Raw, and it is not known until that config exists.
    """
    config = getattr(pipeline, 'config', None)
    if config is None:
        return None
    if hasattr(config, 'get'):
        try:
            present = name in config
        except TypeError:
            present = False
        if not present:
            return None
        return bool(config.get(name))
    if hasattr(config, name):
        return bool(getattr(config, name))
    return None


def _optional_int(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _guidance_effect(model_type) -> str:
    """How guidance relates to the Spectrum schedule.

    ``embed`` is one forward, so the guidance number does not change flex.
    ``cfg`` multiplies a pair of forecasts. ``checkpoint`` is HunyuanImage,
    which uses an embedding or a guider depending on the loaded weights, so
    flex stays unset until the pipeline exists.
    """
    if (_enums.model_type_is_flux_family(model_type)
            or _enums.model_type_is_flux2_family(model_type)):
        return 'embed'
    if _enums.model_type_is_hyimage_21_family(model_type):
        return 'checkpoint'
    return 'cfg'


def _known_sampling(model_type, *, distilled=None, inference_steps=None,
                    guidance_scale=None) -> tuple[int | None, float | None]:
    """Step count and guidance already fixed by the user or by this model type.

    ``None`` means the value is still omitted and this model type does not
    have its own fixed default.
    """
    steps = _optional_int(inference_steps)
    guidance = guidance_scale
    if isinstance(guidance, (list, tuple)):
        guidance = guidance[0] if guidance else None
    if guidance is not None:
        try:
            guidance = float(guidance)
        except (TypeError, ValueError):
            guidance = None
    if steps is None and _enums.model_type_is_krea2(model_type) and distilled is not None:
        steps = 8 if distilled else 28
    if guidance is None and _enums.model_type_is_krea2(model_type) and distilled is not None:
        guidance = 0.0 if distilled else 4.5
    if steps is None and _enums.model_type_is_hyimage_21_refiner(model_type):
        steps = 4
    if steps is None and _enums.model_type_is_hyimage_21(model_type):
        steps = 50
    if steps is None and _enums.model_type_is_kandinsky5_image(model_type):
        steps = 50
    if guidance is None and _enums.model_type_is_kandinsky5_image(model_type):
        guidance = 3.5
    return steps, guidance


def spectrum_defaults(model_type, *, distilled=None, inference_steps=None,
                      guidance_scale=None) -> dict:
    """Spectrum parameters for one model, before any value the user set.

    Warmup and flex follow the step count. Fifty steps or more use warmup 5
    and flex 0.75. Shorter schedules hold flex at 0, with warmup 4, or warmup
    2 at 8 steps or fewer, or a full warmup when there are 4 steps or fewer.
    Classifier-free guidance above 1 also holds flex at 0. When the step count
    or that guidance value is still omitted, the dependent field is ``None``
    so step text can show ``auto``.
    """
    import dgenerate.extras.spectrum as _spectrum

    defaults = {
        'weight': _constants.DEFAULT_SPECTRUM_WEIGHT,
        'order': _constants.DEFAULT_SPECTRUM_ORDER,
        'lam': _constants.DEFAULT_SPECTRUM_LAMBDA,
        'warmup_steps': None,
        'window_size': _constants.DEFAULT_SPECTRUM_WINDOW_SIZE,
        'flex_window': None,
    }
    steps, guidance = _known_sampling(
        model_type, distilled=distilled, inference_steps=inference_steps,
        guidance_scale=guidance_scale)
    if steps is None:
        return defaults
    effect = _guidance_effect(model_type)
    if steps >= 50 and (effect == 'checkpoint' or (effect == 'cfg' and guidance is None)):
        defaults['warmup_steps'] = 5
        return defaults
    cfg = guidance if effect == 'cfg' and guidance is not None and guidance > 1 else None
    warmup, flex = _spectrum.schedule_for(steps, cfg)
    defaults['warmup_steps'] = warmup
    defaults['flex_window'] = flex
    return defaults


def sada_for_steps(num_steps: int, lagrange_step: int, lagrange_int: int) -> tuple[tuple[int, int], int]:
    """Place the 50-step acceleration window and Lagrange step on ``num_steps``.

    The demo window is steps 10 through 46 (``range(10, 47)``), a head of 10
    and a tail of 3. The Lagrange step stays a multiple of ``lagrange_int``.
    """
    steps = max(int(num_steps), 1)
    reference = 50
    start = max(3, round(10 * steps / reference))
    end = steps - (reference - 47)
    if end <= start:
        start, end = 3, 3
    interval = max(int(lagrange_int), 1)
    target = (lagrange_step / reference) * steps
    inside = [multiple for multiple in range(interval, end, interval) if multiple >= start]
    if inside:
        step = min(inside, key=lambda candidate: (abs(candidate - target), candidate))
    else:
        step = int(round(target / interval)) * interval
        if step < interval:
            step = interval
        if steps > interval and step >= steps:
            step = ((steps - 1) // interval) * interval or interval
    return (start, end), step


def configured_sample_sigmas(pipeline) -> list | None:
    """Author sigma list on a loaded pipeline, excluding the terminal sigma.

    Qwen-Image 2.1 Turbo stores this list. Its length is the step count, and
    the pipeline uses it instead of ``num_inference_steps``.
    """
    config = getattr(pipeline, 'config', None)
    if config is None or not hasattr(config, 'get'):
        return None
    try:
        present = 'sample_sigmas' in config
    except TypeError:
        present = False
    if not present:
        return None
    sigmas = config.get('sample_sigmas')
    if not isinstance(sigmas, (list, tuple)) or not sigmas:
        return None
    return list(sigmas)


def schedule_step_count(args) -> int | None:
    """Step count a Spectrum or SADA schedule can follow.

    A numeric ``--sigmas`` list is the schedule the sampler runs. An omitted
    step count stays ``None`` so step text can show ``auto``.
    """
    sigmas = getattr(args, 'sigmas', None)
    if isinstance(sigmas, (list, tuple)) and sigmas and not isinstance(sigmas[0], str):
        return len(sigmas)
    return _optional_int(getattr(args, 'inference_steps', None))


def sada_values(args, model_type) -> dict:
    """SADA parameters for one step, with model defaults for anything omitted.

    The acceleration range and Lagrange step follow the inference step count,
    or the length of a numeric ``--sigmas`` list when that is the schedule.
    They stay ``None`` while that count is omitted.
    """
    defaults = _util.get_sada_model_defaults(model_type)
    steps = schedule_step_count(args)
    acc_range = args.sada_acc_range
    lagrange_step = args.sada_lagrange_step or None
    if steps is not None and (acc_range is None or lagrange_step is None):
        auto_range, auto_step = sada_for_steps(
            steps, defaults['lagrange_step'], defaults['lagrange_int'])
        if acc_range is None:
            acc_range = auto_range
        if lagrange_step is None:
            lagrange_step = auto_step
    return {
        'max_downsample': _types.default(
            args.sada_max_downsample, defaults['max_downsample']),
        'sx': _types.default(args.sada_sx, defaults['sx']),
        'sy': _types.default(args.sada_sy, defaults['sy']),
        'acc_range': acc_range,
        'lagrange_term': _types.default(
            args.sada_lagrange_term, defaults['lagrange_term']),
        'lagrange_int': args.sada_lagrange_int or defaults['lagrange_int'],
        'lagrange_step': lagrange_step,
        'max_fix': _types.default(args.sada_max_fix, defaults['max_fix']),
        'max_interval': _types.default(
            args.sada_max_interval, defaults['max_interval']),
    }


def _regione_defaults(model_type) -> dict:
    if _enums.model_type_is_qwen_image_edit(model_type):
        return _regione.defaults_for_pipeline('QwenImageEditPipeline')
    return _regione.defaults_for_pipeline('FluxKontextPipeline')


def video_fps_default(model_type):
    """Frame rate used when ``--video-fps`` is omitted, or ``None`` for a non-video model."""
    if _enums.model_type_is_ltx_family(model_type):
        return 24.0
    if _enums.model_type_is_wan(model_type):
        return 16.0
    if _enums.model_type_is_wan_animate(model_type):
        return 30.0
    if _enums.model_type_is_wan_animate_2(model_type):
        return 24.0
    if _enums.model_type_is_hyvideo_15(model_type):
        return 15.0
    if _enums.model_type_is_motif_video(model_type):
        return 24.0
    if _enums.model_type_is_kandinsky5_video(model_type):
        return 24.0
    return None


def omitted_image_seed_strength(model_type, parsed, *,
                                control_guidance: bool = False,
                                adetailer: bool = False):
    """Img2img strength for one parsed seed when the user omitted it.

    ``None`` when this seed does not use img2img strength. Video condition
    strength is a different default and is left to the video path.
    """
    if parsed is None or _enums.model_type_is_video_family(model_type):
        return None
    if (_enums.model_type_is_upscaler_family(model_type)
            or _enums.model_type_is_pix2pix_family(model_type)
            or _enums.model_type_is_s_cascade_family(model_type)
            or _enums.model_type_is_flux_fill(model_type)
            or _enums.model_type_is_flux_kontext(model_type)):
        return None
    images = getattr(parsed, 'images', None)
    masks = getattr(parsed, 'mask_images', None)
    if not images:
        return None
    if control_guidance and getattr(parsed, 'is_single_spec', False):
        return None
    if (_enums.model_type_image_is_condition(model_type)
            and not adetailer and not masks):
        return None
    if _enums.model_type_is_ifs(model_type) and not masks:
        return None
    return _constants.DEFAULT_IMAGE_SEED_STRENGTH


def omitted_upscaler_noise_level(model_type, image_seed=None, mask_images=None):
    """Noise level for an upscaler or Floyd model when the user omitted it.

    ``None`` when this model does not use a noise level. Floyd ``ifs`` inpaint
    is 0 when a mask is present, and 250 otherwise.
    """
    if _enums.model_type_is_upscaler_x4(model_type):
        return _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL
    if _enums.model_type_is_ifs_img2img(model_type):
        return _constants.DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL
    if not _enums.model_type_is_ifs(model_type):
        return None
    masks = getattr(image_seed, 'mask_images', None) if image_seed is not None else None
    if not masks:
        masks = mask_images
    if masks:
        return _constants.DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL
    return _constants.DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL


def _wan_video_to_video(model_type, args) -> bool:
    """Same mode choice as the Wan caller: a clip, and not VACE or first-last-frame."""
    if not _enums.model_type_is_wan(model_type):
        return False
    if (getattr(args, 'vace_video_frames', None)
            or getattr(args, 'vace_mask_frames', None)
            or getattr(args, 'vace_reference_images', None)
            or getattr(args, 'end_images', None)):
        return False
    return bool(getattr(args, 'video_frames', None))


def apply_known_defaults(args, model_type, *, refiner: bool = False,
                         image_seed=None, control_guidance: bool = False,
                         adetailer: bool = False, distilled: bool | None = None) -> None:
    """Fill omitted fields that have a known default once their feature is on.

    ``args`` is updated in place. Parent flags are left as they are.
    Does not set ``guidance_scale`` or ``inference_steps`` (those stay omitted
    for step text and are resolved when the pipeline wrapper runs).
    """
    if (_feature_on(args, 'pag')
            or args.pag_scale is not None
            or args.pag_adaptive_scale is not None
            or args.pag_applied_layers):
        _set_missing(args, 'pag_scale', _constants.DEFAULT_PAG_SCALE)
        _set_missing(args, 'pag_adaptive_scale', _constants.DEFAULT_PAG_ADAPTIVE_SCALE)

    if (getattr(args, 'sdxl_refiner_pag_scale', None) is not None
            or getattr(args, 'sdxl_refiner_pag_adaptive_scale', None) is not None
            or getattr(args, 'sdxl_refiner_pag_applied_layers', None)):
        _set_missing(args, 'sdxl_refiner_pag_scale',
                     _constants.DEFAULT_SDXL_REFINER_PAG_SCALE)
        _set_missing(args, 'sdxl_refiner_pag_adaptive_scale',
                     _constants.DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE)

    if _feature_on(args, 'dype'):
        _set_missing(args, 'dype_method', _constants.DEFAULT_DYPE_METHOD)
        _set_missing(args, 'dype_mscale', _constants.DEFAULT_DYPE_MSCALE)
        _set_missing(args, 'dype_axis_scaling', _constants.DEFAULT_DYPE_AXIS_SCALING)

    if _feature_on(args, 'tea_cache'):
        _set_missing(args, 'tea_cache_rel_l1_threshold',
                     _constants.DEFAULT_TEA_CACHE_REL_L1_THRESHOLD)

    if _feature_on(args, 'spectrum'):
        spectrum = spectrum_defaults(
            model_type,
            distilled=distilled,
            inference_steps=getattr(args, 'inference_steps', None),
            guidance_scale=getattr(args, 'guidance_scale', None))
        _set_missing(args, 'spectrum_weight', spectrum['weight'])
        _set_missing(args, 'spectrum_order', spectrum['order'])
        _set_missing(args, 'spectrum_lambda', spectrum['lam'])
        if spectrum['warmup_steps'] is not None:
            _set_missing(args, 'spectrum_warmup_steps', spectrum['warmup_steps'])
        _set_missing(args, 'spectrum_window_size', spectrum['window_size'])
        if spectrum['flex_window'] is not None:
            _set_missing(args, 'spectrum_flex_window', spectrum['flex_window'])

    if _feature_on(args, 'regione'):
        defaults = _regione_defaults(model_type)
        _set_missing(args, 'regione_warmup_step', defaults['warmup_step'])
        _set_missing(args, 'regione_post_step', defaults['post_step'])
        _set_missing(args, 'regione_refresh_step', defaults['refresh_step'])
        _set_missing(args, 'regione_threshold', defaults['threshold'])
        _set_missing(args, 'regione_cache_threshold', defaults['cache_threshold'])
        _set_missing(args, 'regione_erosion_dilation', defaults['erosion_dilation'])

    if _feature_on(args, 'ras'):
        _set_missing(args, 'ras_sample_ratio', _constants.DEFAULT_RAS_SAMPLE_RATIO)
        _set_missing(args, 'ras_high_ratio', _constants.DEFAULT_RAS_HIGH_RATIO)
        _set_missing(args, 'ras_starvation_scale',
                     _constants.DEFAULT_RAS_STARVATION_SCALE)
        _set_missing(args, 'ras_error_reset_steps',
                     list(_constants.DEFAULT_RAS_ERROR_RESET_STEPS))
        _set_missing(args, 'ras_metric', _constants.DEFAULT_RAS_METRIC)
        _set_missing(args, 'ras_start_step', _constants.DEFAULT_RAS_START_STEP)
        # Leave the end unset while the step count is still omitted. The
        # pipeline wrapper fills it from the resolved step count. Pinning the
        # fallback of 30 here would disagree with a pipeline whose own default
        # is different, such as Stable Diffusion 3 at 28.
        if args.inference_steps is not None:
            _set_missing(args, 'ras_end_step', args.inference_steps)
        _set_missing(args, 'ras_skip_num_step', _constants.DEFAULT_RAS_SKIP_NUM_STEP)
        _set_missing(args, 'ras_skip_num_step_length',
                     _constants.DEFAULT_RAS_SKIP_NUM_STEP_LENGTH)

    if _feature_on(args, 'deep_cache'):
        _set_missing(args, 'deep_cache_interval',
                     _constants.DEFAULT_DEEP_CACHE_INTERVAL)
        _set_missing(args, 'deep_cache_branch_id',
                     _constants.DEFAULT_DEEP_CACHE_BRANCH_ID)

    if (args.sdxl_refiner_deep_cache
            or _prefixed_value_set(args, 'sdxl_refiner_deep_cache_')):
        _set_missing(args, 'sdxl_refiner_deep_cache_interval',
                     _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL)
        _set_missing(args, 'sdxl_refiner_deep_cache_branch_id',
                     _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID)

    if _feature_on(args, 'sada'):
        values = sada_values(args, model_type)
        args.sada_max_downsample = values['max_downsample']
        args.sada_sx = values['sx']
        args.sada_sy = values['sy']
        if values['acc_range'] is not None:
            args.sada_acc_range = values['acc_range']
        args.sada_lagrange_term = values['lagrange_term']
        args.sada_lagrange_int = values['lagrange_int']
        if values['lagrange_step'] is not None:
            args.sada_lagrange_step = values['lagrange_step']
        args.sada_max_fix = values['max_fix']
        args.sada_max_interval = values['max_interval']

    if (args.inpaint_crop or args.inpaint_crop_padding is not None
            or args.inpaint_crop_feather is not None or args.inpaint_crop_masked):
        _set_missing(args, 'inpaint_crop_padding',
                     _constants.DEFAULT_INPAINT_CROP_PADDING)

    if args.video_fps is None:
        _set_missing(args, 'video_fps', video_fps_default(model_type))

    noise = omitted_upscaler_noise_level(
        model_type, image_seed, getattr(args, 'mask_images', None))
    if noise is not None:
        _set_missing(args, 'upscaler_noise_level', noise)

    if _enums.model_type_is_pix2pix_family(model_type):
        _set_missing(args, 'image_guidance_scale',
                     _constants.DEFAULT_IMAGE_GUIDANCE_SCALE)

    if _enums.model_type_is_s_cascade(model_type):
        _set_missing(args, 'second_model_guidance_scale',
                     _constants.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE)
        _set_missing(args, 'second_model_inference_steps',
                     _constants.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS)

    if refiner and (_enums.model_type_is_sdxl_family(model_type)
                    or _enums.model_type_is_kolors(model_type)):
        _set_missing(args, 'sdxl_high_noise_fraction',
                     _constants.DEFAULT_SDXL_HIGH_NOISE_FRACTION)

    if args.image_seed_strength is None:
        strength = omitted_image_seed_strength(
            model_type, image_seed,
            control_guidance=control_guidance, adetailer=adetailer)
        if strength is None and _wan_video_to_video(model_type, args):
            strength = _constants.DEFAULT_IMAGE_SEED_STRENGTH
        if strength is not None:
            args.image_seed_strength = strength


# Config list attributes paired with :class:`~dgenerate.pipelinewrapper.GenerationArguments`
# fields filled by :func:`apply_known_defaults` (not guidance or step counts).
_RENDER_LOOP_LIST_ARG_FIELDS = (
    ('pag_scales', 'pag_scale'),
    ('pag_adaptive_scales', 'pag_adaptive_scale'),
    ('sdxl_refiner_pag_scales', 'sdxl_refiner_pag_scale'),
    ('sdxl_refiner_pag_adaptive_scales', 'sdxl_refiner_pag_adaptive_scale'),
    ('dype_methods', 'dype_method'),
    ('dype_mscales', 'dype_mscale'),
    ('dype_axis_scalings', 'dype_axis_scaling'),
    ('tea_cache_rel_l1_thresholds', 'tea_cache_rel_l1_threshold'),
    ('spectrum_weights', 'spectrum_weight'),
    ('spectrum_orders', 'spectrum_order'),
    ('spectrum_lambdas', 'spectrum_lambda'),
    ('spectrum_warmup_steps', 'spectrum_warmup_steps'),
    ('spectrum_window_sizes', 'spectrum_window_size'),
    ('spectrum_flex_windows', 'spectrum_flex_window'),
    ('regione_warmup_steps', 'regione_warmup_step'),
    ('regione_post_steps', 'regione_post_step'),
    ('regione_refresh_steps', 'regione_refresh_step'),
    ('regione_thresholds', 'regione_threshold'),
    ('regione_cache_thresholds', 'regione_cache_threshold'),
    ('ras_sample_ratios', 'ras_sample_ratio'),
    ('ras_high_ratios', 'ras_high_ratio'),
    ('ras_starvation_scales', 'ras_starvation_scale'),
    ('ras_error_reset_steps', 'ras_error_reset_steps'),
    ('ras_metrics', 'ras_metric'),
    ('ras_start_steps', 'ras_start_step'),
    ('ras_end_steps', 'ras_end_step'),
    ('ras_skip_num_steps', 'ras_skip_num_step'),
    ('ras_skip_num_step_lengths', 'ras_skip_num_step_length'),
    ('deep_cache_intervals', 'deep_cache_interval'),
    ('deep_cache_branch_ids', 'deep_cache_branch_id'),
    ('sdxl_refiner_deep_cache_intervals', 'sdxl_refiner_deep_cache_interval'),
    ('sdxl_refiner_deep_cache_branch_ids', 'sdxl_refiner_deep_cache_branch_id'),
    ('sada_max_downsamples', 'sada_max_downsample'),
    ('sada_sxs', 'sada_sx'),
    ('sada_sys', 'sada_sy'),
    ('sada_acc_ranges', 'sada_acc_range'),
    ('sada_lagrange_terms', 'sada_lagrange_term'),
    ('sada_lagrange_ints', 'sada_lagrange_int'),
    ('sada_lagrange_steps', 'sada_lagrange_step'),
    ('sada_max_fixes', 'sada_max_fix'),
    ('sada_max_intervals', 'sada_max_interval'),
    ('inpaint_crop_paddings', 'inpaint_crop_padding'),
    ('video_fps', 'video_fps'),
    ('upscaler_noise_levels', 'upscaler_noise_level'),
    ('image_guidance_scales', 'image_guidance_scale'),
    ('second_model_guidance_scales', 'second_model_guidance_scale'),
    ('second_model_inference_steps', 'second_model_inference_steps'),
    ('sdxl_high_noise_fractions', 'sdxl_high_noise_fraction'),
    ('image_seed_strengths', 'image_seed_strength'),
)

_RENDER_LOOP_SCALAR_ARG_FIELDS = (
    ('regione_erosion_dilation', 'regione_erosion_dilation'),
)


def _config_feature_on(config, name: str) -> bool:
    if getattr(config, name, False):
        return True
    prefix = name + '_'
    for key, value in config.__dict__.items():
        if key.startswith(prefix) and value is not None:
            return True
    return False


def _first_list_value(values):
    if values is None:
        return None
    if isinstance(values, list):
        return values[0] if values else None
    return values


def _prototype_from_render_loop_config(config):
    import dgenerate.pipelinewrapper.arguments as _arguments

    args = _arguments.GenerationArguments()
    args.set_from(config, missing_value_throws=False)
    for config_name, arg_name in _RENDER_LOOP_LIST_ARG_FIELDS:
        setattr(args, arg_name, _first_list_value(getattr(config, config_name, None)))
    for config_name, arg_name in _RENDER_LOOP_SCALAR_ARG_FIELDS:
        if getattr(config, config_name, None) is not None:
            setattr(args, arg_name, getattr(config, config_name))
    return args


def _write_config_list_default(config, config_name, value) -> None:
    if getattr(config, config_name, None) is not None or value is None:
        return
    if config_name == 'ras_error_reset_steps':
        # One pattern per generation. The pattern is itself a list of steps,
        # so [12, 22] must be stored as [[12, 22]]. A flat list is read as
        # two generations whose values are the integers 12 and 22.
        setattr(config, config_name, [list(value)])
        return
    if isinstance(value, list):
        setattr(config, config_name, list(value))
    else:
        setattr(config, config_name, [value])


def _write_config_scalar_default(config, config_name, value) -> None:
    if getattr(config, config_name, None) is not None or value is None:
        return
    setattr(config, config_name, value)


def fill_render_loop_config_defaults(
        config,
        *,
        pipeline=None,
        image_seed=None,
        control_guidance: bool = False) -> None:
    """Fill omitted :class:`~dgenerate.renderloopconfig.RenderLoopConfig` lists.

    Applies :func:`apply_known_defaults` for enabled features (SADA, PAG,
    RegionE thresholds, and similar). Does not set ``guidance_scales`` or
    ``inference_steps``; those stay omitted until the pipeline wrapper or video
    pipeline resolves them. RegionE still normalizes omitted step
    lists to 28 via :func:`~dgenerate.extras.regione.runtime.resolve_regione_inference_steps_list`.

    Call after :meth:`RenderLoopConfig.check` when ``parsed_image_seeds`` is
    available for img2img strength defaults. Updates ``config`` in place.
    """
    model_type = getattr(config, 'model_type', None)
    if model_type is None:
        return

    if image_seed is None:
        parsed = getattr(config, 'parsed_image_seeds', None)
        if parsed:
            image_seed = parsed[0]

    prototype = _prototype_from_render_loop_config(config)
    refiner = bool(getattr(config, 'sdxl_refiner_uri', None))
    adetailer = bool(getattr(config, 'adetailer_detector_uris', None))
    distilled = None
    if pipeline is not None and _enums.model_type_is_krea2(model_type):
        distilled = pipeline_config_bool(pipeline, 'is_distilled')

    apply_known_defaults(
        prototype, model_type, refiner=refiner, image_seed=image_seed,
        control_guidance=control_guidance, adetailer=adetailer,
        distilled=distilled)

    for config_name, arg_name in _RENDER_LOOP_LIST_ARG_FIELDS:
        _write_config_list_default(config, config_name, getattr(prototype, arg_name))

    for config_name, arg_name in _RENDER_LOOP_SCALAR_ARG_FIELDS:
        _write_config_scalar_default(config, config_name, getattr(prototype, arg_name))

    if _config_feature_on(config, 'regione'):
        config.inference_steps = _regione.resolve_regione_inference_steps_list(
            config.inference_steps)
