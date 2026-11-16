# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Give Wan FLF2V single-file loads a real position table.

``convert_wan_transformer_to_diffusers`` renames ``img_emb.proj.*`` and leaves
``img_emb.emb_pos`` under that name. The Diffusers parameter
``condition_embedder.image_embedder.pos_embed`` is never filled, stays a meta
tensor, and ``module.to()`` then raises "Cannot copy out of meta tensor".
"""

import diffusers.loaders.single_file_model as _single_file_model
import diffusers.loaders.single_file_utils as _single_file_utils


def _install():
    original = _single_file_utils.convert_wan_transformer_to_diffusers
    if getattr(original, '_dgenerate_wan_flf_pos', False):
        return

    def convert_wan_transformer_to_diffusers(checkpoint, **kwargs):
        converted = original(checkpoint, **kwargs)
        import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
        return _ggufcompat.assign_wan_flf_pos_embed(converted)

    convert_wan_transformer_to_diffusers._dgenerate_wan_flf_pos = True
    _single_file_utils.convert_wan_transformer_to_diffusers = convert_wan_transformer_to_diffusers
    for entry in _single_file_model.SINGLE_FILE_LOADABLE_CLASSES.values():
        if entry.get('checkpoint_mapping_fn') is original:
            entry['checkpoint_mapping_fn'] = convert_wan_transformer_to_diffusers


_install()
