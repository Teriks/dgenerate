# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Keep streamed group-offload weights aligned with a later dtype cast.

CUDA and XPU group offload pins a CPU copy of each parameter when the hook is
installed, and every later onload copies that snapshot back onto the
accelerator. ``module.to(dtype=...)`` replaces the live parameter storage but
leaves the snapshot alone. The module then reports the new dtype — pipelines
cast activations to it — while the next forward restores the old bias and
weight dtype.

Wan hits this when ``--dtype bfloat16`` loads the VAE, group offload snapshots
it, and the VAE is then cast to float32. Decode feeds float32 latents to a
bfloat16 ``post_quant_conv`` bias. The same split happens for any pipeline that
upcasts a VAE after offload, including ``force_upcast``.
"""

import diffusers.hooks.group_offloading as _group_offloading


def _live_offload_tensors(group):
    seen = []
    seen_ids = set()

    def add(tensor):
        ident = id(tensor)
        if ident in seen_ids:
            return
        seen_ids.add(ident)
        seen.append(tensor)

    for module in group.modules:
        for param in module.parameters():
            add(param)
        for buffer in module.buffers():
            add(buffer)
    for param in group.parameters:
        add(param)
    for buffer in group.buffers:
        add(buffer)
    return seen


def _packed_quant_tensor(tensor) -> bool:
    """True for GGUF, bitsandbytes, and SDNQ storage that ``.data`` must not replace."""
    if type(tensor).__name__ in {'Params4bit', 'Int8Params', 'GGUFParameter', 'SDNQTensor'}:
        return True
    if getattr(tensor, 'quant_type', None) is not None:
        return True
    return getattr(tensor, 'quant_state', None) is not None


def _snapshot_matches(snapshot, tensor) -> bool:
    return (
        snapshot is not None
        and snapshot.dtype == tensor.dtype
        and snapshot.shape == tensor.shape)


def refresh_cpu_snapshots(group) -> bool:
    """
    Rebuild streamed CPU snapshots whose dtype or shape no longer matches.

    No change when streams are off: that path moves the live parameters and
    already keeps a dtype cast. Returns whether any snapshot was replaced.
    """
    if group.stream is None:
        return False
    tensors = _live_offload_tensors(group)
    if any(_packed_quant_tensor(tensor) for tensor in tensors):
        return False
    snapshots = group.cpu_param_dict
    if snapshots and len(snapshots) == len(tensors) and all(
            _snapshot_matches(snapshots.get(tensor), tensor) for tensor in tensors):
        return False
    group.cpu_param_dict = {
        tensor: (
            snapshots[tensor]
            if _snapshot_matches(snapshots.get(tensor), tensor)
            else group._to_cpu(tensor, group.low_cpu_mem_usage))
        for tensor in tensors
    }
    return True


def _install():
    module_group = _group_offloading.ModuleGroup
    if getattr(module_group, '_dgenerate_dtype_snapshots', False):
        return

    original_onload = module_group._onload_from_memory
    original_offload = module_group._offload_to_memory

    def _onload_from_memory(self):
        refresh_cpu_snapshots(self)
        return original_onload(self)

    def _offload_to_memory(self):
        refresh_cpu_snapshots(self)
        return original_offload(self)

    module_group._onload_from_memory = _onload_from_memory
    module_group._offload_to_memory = _offload_to_memory
    module_group._dgenerate_dtype_snapshots = True


_install()
