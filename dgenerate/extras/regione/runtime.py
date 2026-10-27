"""
RegionE instruction-edit acceleration.

https://github.com/Peyton-Chen/RegionE
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import typing

import dgenerate.messages as _messages

# Defaults from RegionE/RegionE/tool/RegionE.py (per pipeline class).
_PIPELINE_DEFAULTS: dict[str, dict[str, typing.Any]] = {
    'FluxKontextPipeline': {
        'warmup_step': 6,
        'post_step': 2,
        'refresh_step': 16,
        'threshold': 0.93,
        'cache_threshold': 0.04,
        'erosion_dilation': True,
    },
    'QwenImageEditPipeline': {
        'warmup_step': 6,
        'post_step': 2,
        'refresh_step': 16,
        'threshold': 0.80,
        'cache_threshold': 0.03,
        'erosion_dilation': True,
    },
    'QwenImageEditPlusPipeline': {
        'warmup_step': 6,
        'post_step': 2,
        'refresh_step': 16,
        'threshold': 0.80,
        'cache_threshold': 0.03,
        'erosion_dilation': True,
    },
}

REGIONE_INFERENCE_STEPS = 28


def _default_inference_steps() -> int:
    import dgenerate.pipelinewrapper.constants as _constants

    return _constants.DEFAULT_INFERENCE_STEPS


def resolve_regione_inference_steps_list(
        steps: typing.Sequence[int] | None) -> list[int]:
    """
    RegionE requires 28 steps. Treat ``None`` and the global default (30) as omitted.
    """
    default = _default_inference_steps()
    if steps is None:
        return [REGIONE_INFERENCE_STEPS]
    normalized = list(steps)
    if normalized == [default]:
        return [REGIONE_INFERENCE_STEPS]
    return normalized


def resolve_regione_inference_step_count(steps: int | None) -> int:
    """Like :func:`resolve_regione_inference_steps_list` for a single step count."""
    default = _default_inference_steps()
    if steps is None or steps == default:
        return REGIONE_INFERENCE_STEPS
    return steps


@dataclasses.dataclass(frozen=True)
class RegionESettings:
    warmup_step: int
    post_step: int
    refresh_step: int
    threshold: float
    cache_threshold: float
    erosion_dilation: bool


def pipeline_class_supports_regione(class_name: str) -> bool:
    if 'Inpaint' in class_name:
        return False
    return class_name in _PIPELINE_DEFAULTS


def defaults_for_pipeline(class_name: str) -> dict[str, typing.Any]:
    try:
        return dict(_PIPELINE_DEFAULTS[class_name])
    except KeyError as e:
        raise ValueError(f'Unsupported pipeline class for RegionE: {class_name}') from e


def validate_settings(settings: RegionESettings) -> None:
    """
    Raise ``ValueError`` when a RegionE schedule cannot be applied.

    Upstream checks these with ``assert``, which disappears under ``python -O``
    and is not a configuration error.
    """
    steps = REGIONE_INFERENCE_STEPS
    if settings.warmup_step < 1:
        raise ValueError('RegionE warmup steps must be at least 1.')
    if settings.post_step < 0:
        raise ValueError('RegionE post steps cannot be negative.')
    refresh = settings.refresh_step
    if refresh <= settings.warmup_step + 1:
        raise ValueError(
            'RegionE refresh step must be greater than warmup step + 1 '
            f'(warmup {settings.warmup_step}, refresh {refresh}).')
    latest = steps - settings.post_step - 1
    if refresh > latest:
        raise ValueError(
            'RegionE refresh step must be at most '
            f'inference steps - post steps - 1 ({latest}); got {refresh}.')


def _attention_blocks(transformer):
    for name in ('transformer_blocks', 'single_transformer_blocks'):
        for block in getattr(transformer, name, None) or ():
            attn = getattr(block, 'attn', None)
            if attn is not None and hasattr(attn, 'set_processor'):
                yield attn


def _snapshot_pipeline(pipeline):
    transformer = pipeline.transformer
    return {
        'cls': pipeline.__class__,
        'scheduler': pipeline.scheduler,
        'forward': transformer.forward,
        'processors': [(attn, attn.processor) for attn in _attention_blocks(transformer)],
    }


def _restore_pipeline(pipeline, snapshot) -> None:
    pipeline.__class__ = snapshot['cls']
    pipeline.scheduler = snapshot['scheduler']
    pipeline.transformer.forward = snapshot['forward']
    for attn, processor in snapshot['processors']:
        attn.set_processor(processor)


def build_settings(
        pipeline_class_name: str,
        *,
        warmup_step: int | None,
        post_step: int | None,
        refresh_step: int | None,
        threshold: float | None,
        cache_threshold: float | None,
        erosion_dilation: bool | None) -> RegionESettings:
    defaults = defaults_for_pipeline(pipeline_class_name)
    return RegionESettings(
        warmup_step=int(warmup_step if warmup_step is not None else defaults['warmup_step']),
        post_step=int(post_step if post_step is not None else defaults['post_step']),
        refresh_step=int(refresh_step if refresh_step is not None else defaults['refresh_step']),
        threshold=float(threshold if threshold is not None else defaults['threshold']),
        cache_threshold=float(
            cache_threshold if cache_threshold is not None else defaults['cache_threshold']),
        erosion_dilation=bool(
            erosion_dilation if erosion_dilation is not None else defaults['erosion_dilation']),
    )


@contextlib.contextmanager
def regione_context(pipeline, settings: RegionESettings | None):
    if settings is None:
        yield
        return

    class_name = pipeline.__class__.__name__
    if not pipeline_class_supports_regione(class_name):
        raise ValueError(
            f'RegionE does not support {class_name}. '
            f'Supported pipelines: {", ".join(sorted(_PIPELINE_DEFAULTS))}.')

    validate_settings(settings)

    from dgenerate.extras.regione.upstream import RegionEHelper

    helper = RegionEHelper(pipeline)
    helper.set_params(
        num_inference_steps=REGIONE_INFERENCE_STEPS,
        warmup_step=settings.warmup_step,
        post_step=settings.post_step,
        refresh_step=str(settings.refresh_step),
        threshold=settings.threshold,
        cache_threshold=settings.cache_threshold,
        erosion_dilation=settings.erosion_dilation,
    )
    _messages.debug_log(
        f'regione: enable on {class_name} with {settings}')

    # warp_modules replaces the class, scheduler, forward, and attention
    # processors and unwarp_modules puts back stock implementations. Restore
    # the objects this pipeline was already using, including after a failed warp.
    snapshot = _snapshot_pipeline(pipeline)
    wrapped_call = False
    try:
        try:
            helper.enable()
        except AssertionError as e:
            raise ValueError(str(e) or 'RegionE rejected this schedule.') from e
        call = pipeline.__call__
        visible = inspect.unwrap(call)
        if '_auto_resize' in inspect.signature(visible).parameters:
            def _call_with_requested_size(*args, **kwargs):
                # Keep the image size dgenerate already resized to.
                kwargs['_auto_resize'] = False
                return call(*args, **kwargs)

            pipeline.__call__ = _call_with_requested_size
            wrapped_call = True
        yield
    finally:
        if wrapped_call:
            del pipeline.__call__
        _restore_pipeline(pipeline, snapshot)
