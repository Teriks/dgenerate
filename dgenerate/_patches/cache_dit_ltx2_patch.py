# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Make Cache-DiT's LTX-2 support work with the installed diffusers LTX-2 transformer.

Cache-DiT's ``LTX2PatchFunctor`` replaces the transformer forward with a copy of
an older diffusers implementation that calls blocks positionally. That copy drops
newer forward arguments (prompt modulation, attention masks, STG perturbation,
modality isolation), and Cache-DiT bypasses it entirely when Accelerate offload
hooks are present, which fails with "got multiple values for argument
'audio_hidden_states'". Leave the diffusers forward alone, and instead remap its
keyword block call at Cache-DiT's cached block wrapper, where
``audio_hidden_states`` takes the ``encoder_hidden_states`` slot of
``ForwardPattern.Pattern_0``.

The LTX-2 pipeline also runs STG and modality isolation transformer passes within
each denoising step. Each kind of pass is cached in its own Cache-DiT context, so
residuals are only reused between passes of the same kind.

Cache-DiT is optional, so this module is imported when Cache-DiT is enabled
rather than at dgenerate startup.
"""

import torch
from cache_dit.caching.cache_blocks import pattern_base as _pattern_base


def _guidance_pass_cache_context(cached_blocks, kind: str) -> str:
    """Return the Cache-DiT context name for one kind of LTX-2 guidance pass.

    The context mirrors the cached blocks' own context, and is recreated whenever
    Cache-DiT recreates that context for a new pipeline call.
    """
    main_name = cached_blocks.cache_context
    try:
        main_context = cached_blocks.context_manager.get_context(main_name)
    except Exception:
        # Let Cache-DiT handle the missing context as it would for the main pass.
        return main_name
    name = f'{main_name}_dgenerate_ltx2_{kind}'
    sources = cached_blocks.__dict__.setdefault('_dgenerate_ltx2_guidance_contexts', {})
    if sources.get(kind) is not main_context:
        cached_blocks.context_manager.reset_context(
            name, *main_context._init_args, **main_context._init_kwargs)
        sources[kind] = main_context
    return name


def _guidance_pass_pre_hook(module, args, kwargs):
    """Record whether this LTX-2 transformer call is an STG or modality isolation pass."""
    stg_blocks = kwargs.get('spatio_temporal_guidance_blocks') or []
    isolate_modalities = bool(kwargs.get('isolate_modalities', False))
    if not stg_blocks and not isolate_modalities:
        module._dgenerate_ltx2_guidance_pass = None
        return None
    kind = '_'.join(name for name, used in
                    (('stg', stg_blocks), ('isolate', isolate_modalities)) if used)
    # Mirror LTX2VideoTransformer3DModel.forward, which builds this mask
    # and hands it only to the blocks listed in spatio_temporal_guidance_blocks.
    perturbation_mask = kwargs.get('perturbation_mask')
    if stg_blocks and perturbation_mask is None:
        hidden_states = kwargs.get('hidden_states', args[0] if args else None)
        perturbation_mask = torch.zeros((hidden_states.shape[0],))
    if perturbation_mask is not None and perturbation_mask.ndim == 1:
        perturbation_mask = perturbation_mask[:, None, None]
    all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
    module._dgenerate_ltx2_guidance_pass = (kind, set(stg_blocks), perturbation_mask, all_perturbed)
    return None


def _block_stg_pre_hook(transformer, index: int):
    """Apply STG perturbation to one LTX-2 block by its index.

    Cache-DiT passes every block in its group the arguments the transformer gave
    the first block, so the per-block perturbation mask is restored here.
    """

    def hook(block, args, kwargs):
        guidance_pass = transformer._dgenerate_ltx2_guidance_pass
        if guidance_pass is None or not guidance_pass[1] or 'perturbation_mask' not in kwargs:
            return None
        in_stg = index in guidance_pass[1]
        kwargs['perturbation_mask'] = guidance_pass[2] if in_stg else None
        kwargs['all_perturbed'] = guidance_pass[3] if in_stg else False
        return args, kwargs

    return hook


def register_hooks(pipeline) -> None:
    """Install the LTX-2 guidance pass hooks used by the Cache-DiT block remapping."""
    try:
        from diffusers import LTX2VideoTransformer3DModel
    except ImportError:
        return
    for name in ('transformer', 'transformer_2'):
        transformer = getattr(pipeline, name, None)
        if not isinstance(transformer, LTX2VideoTransformer3DModel):
            continue
        if getattr(transformer, '_dgenerate_ltx2_guidance_hook', False):
            continue
        transformer._dgenerate_ltx2_guidance_pass = None
        transformer.register_forward_pre_hook(_guidance_pass_pre_hook, with_kwargs=True)
        for index, block in enumerate(transformer.transformer_blocks):
            block.register_forward_pre_hook(
                _block_stg_pre_hook(transformer, index), with_kwargs=True)
        transformer._dgenerate_ltx2_guidance_hook = True


def _wrap_forward(cls):
    original_forward = cls.forward

    def forward(self, *args, **kwargs):
        if args or 'audio_hidden_states' not in kwargs:
            return original_forward(self, *args, **kwargs)
        block_args = (
            kwargs.pop('hidden_states'),
            kwargs.pop('audio_hidden_states'),
            kwargs.pop('encoder_hidden_states'),
            kwargs.pop('audio_encoder_hidden_states'))
        guidance_pass = getattr(
            getattr(self, 'transformer', None), '_dgenerate_ltx2_guidance_pass', None)
        if guidance_pass is None:
            return original_forward(self, *block_args, **kwargs)
        main_context = self.cache_context
        self.cache_context = _guidance_pass_cache_context(self, guidance_pass[0])
        try:
            return original_forward(self, *block_args, **kwargs)
        finally:
            self.cache_context = main_context

    cls.forward = forward


def _install():
    cached_blocks = _pattern_base.CachedBlocks_Pattern_Base
    if getattr(cached_blocks, '_dgenerate_ltx2_keyword_blocks', False):
        return

    try:
        from cache_dit.caching.patch_functors import LTX2PatchFunctor
    except Exception:
        LTX2PatchFunctor = None
    if LTX2PatchFunctor is not None:
        LTX2PatchFunctor._apply = lambda self, transformer, **kwargs: transformer

    _wrap_forward(cached_blocks)
    _wrap_forward(_pattern_base.PrunedBlocks_Pattern_Base)
    cached_blocks._dgenerate_ltx2_keyword_blocks = True


_install()
