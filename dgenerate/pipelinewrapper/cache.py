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
import gc
import typing

import diffusers

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.types as _types

_TORCH_PIPELINE_CACHE = dict()
"""Global in memory cache for torch diffusers pipelines"""

_FLAX_PIPELINE_CACHE = dict()
"""Global in memory cache for flax diffusers pipelines"""

_PIPELINE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all pipelines cached in memory"""

_TORCH_CONTROL_NET_CACHE = dict()
"""Global in memory cache for torch ControlNet models"""

_FLAX_CONTROL_NET_CACHE = dict()
"""Global in memory cache for flax ControlNet models"""

_CONTROL_NET_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all ControlNet models cached in memory"""

_TORCH_VAE_CACHE = dict()
"""Global in memory cache for torch VAE models"""

_FLAX_VAE_CACHE = dict()
"""Global in memory cache for flax VAE models"""

_TORCH_UNET_CACHE = dict()
"""Global in memory cache for torch UNet models"""

_FLAX_UNET_CACHE = dict()
"""Global in memory cache for flax UNet models"""

_VAE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all VAE models cached in memory"""

_UNET_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all UNet models cached in memory"""


def pipeline_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all diffusers pipelines currently cached in memory.

    :return: memory usage in bytes.
    """
    return _PIPELINE_CACHE_SIZE


def vae_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified VAEs currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _VAE_CACHE_SIZE


def unet_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified UNets currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _UNET_CACHE_SIZE


def control_net_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified ControlNet models currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _CONTROL_NET_CACHE_SIZE


CACHE_MEMORY_CONSTRAINTS: list[str] = ['used_percent > 70']
"""
Cache constraint expressions for when to clear all model caches (DiffusionPipeline, VAE, and ControlNet), 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_cache_constraints` will call
:py:func:`.clear_model_cache` and force a garbage collection.
"""

PIPELINE_CACHE_MEMORY_CONSTRAINTS: list[str] = ['pipeline_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the DiffusionPipeline cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_pipeline_cache_constraints` will call
:py:func:`.clear_pipeline_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``pipeline_size`` (the estimated size of the new pipeline before it is brought into memory, in bytes)
"""

UNET_CACHE_MEMORY_CONSTRAINTS: list[str] = ['unet_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear UNet cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_unet_cache_constraints` will call
:py:func:`.clear_unet_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``unet_size`` (the estimated size of the new UNet before it is brought into memory, in bytes)
"""

VAE_CACHE_MEMORY_CONSTRAINTS: list[str] = ['vae_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear VAE cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_vae_cache_constraints` will call
:py:func:`.clear_vae_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``vae_size`` (the estimated size of the new VAE before it is brought into memory, in bytes)
"""

CONTROL_NET_CACHE_MEMORY_CONSTRAINTS: list[str] = ['control_net_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the ControlNet cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_control_net_cache_constraints` will call
:py:func:`.clear_control_net_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``control_net_size`` (the estimated size of the new ControlNet before it is brought into memory, in bytes)
"""


def clear_pipeline_cache(collect=True):
    """
    Clear DiffusionPipeline cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TORCH_PIPELINE_CACHE, \
        _FLAX_PIPELINE_CACHE, \
        _PIPELINE_CACHE_SIZE

    _TORCH_PIPELINE_CACHE.clear()
    _FLAX_PIPELINE_CACHE.clear()
    _PIPELINE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_pipeline_cache)} calling gc.collect() by request')

        gc.collect()


def clear_control_net_cache(collect=True):
    """
    Clear ControlNet cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TORCH_CONTROL_NET_CACHE, \
        _FLAX_CONTROL_NET_CACHE, \
        _CONTROL_NET_CACHE_SIZE

    _TORCH_CONTROL_NET_CACHE.clear()
    _FLAX_CONTROL_NET_CACHE.clear()

    _CONTROL_NET_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_control_net_cache)} calling gc.collect() by request')

        gc.collect()


def clear_vae_cache(collect=True):
    """
    Clear VAE cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TORCH_VAE_CACHE, \
        _FLAX_VAE_CACHE, \
        _VAE_CACHE_SIZE

    _TORCH_VAE_CACHE.clear()
    _FLAX_VAE_CACHE.clear()

    _VAE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_vae_cache)} calling gc.collect() by request')

        gc.collect()


def clear_unet_cache(collect=True):
    """
    Clear UNet cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TORCH_UNET_CACHE, \
        _FLAX_UNET_CACHE, \
        _UNET_CACHE_SIZE

    _TORCH_UNET_CACHE.clear()
    _FLAX_UNET_CACHE.clear()

    _UNET_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_unet_cache)} calling gc.collect() by request')

        gc.collect()


def clear_model_cache(collect=True):
    """
    Clear all in memory model caches and garbage collect.

    :param collect: Call :py:func:`gc.collect` ?

    """
    global _TORCH_PIPELINE_CACHE, \
        _FLAX_PIPELINE_CACHE, \
        _TORCH_CONTROL_NET_CACHE, \
        _FLAX_CONTROL_NET_CACHE, \
        _TORCH_UNET_CACHE, \
        _FLAX_UNET_CACHE, \
        _TORCH_VAE_CACHE, \
        _FLAX_VAE_CACHE, \
        _PIPELINE_CACHE_SIZE, \
        _CONTROL_NET_CACHE_SIZE, \
        _UNET_CACHE_SIZE, \
        _VAE_CACHE_SIZE

    _TORCH_PIPELINE_CACHE.clear()
    _FLAX_PIPELINE_CACHE.clear()
    _TORCH_CONTROL_NET_CACHE.clear()
    _FLAX_CONTROL_NET_CACHE.clear()
    _TORCH_UNET_CACHE.clear()
    _FLAX_UNET_CACHE.clear()
    _TORCH_VAE_CACHE.clear()
    _FLAX_VAE_CACHE.clear()

    _PIPELINE_CACHE_SIZE = 0
    _CONTROL_NET_CACHE_SIZE = 0
    _UNET_CACHE_SIZE = 0
    _VAE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_model_cache)} calling gc.collect() by request')

        gc.collect()


def enforce_cache_constraints(collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.CACHE_MEMORY_CONSTRAINTS` and clear caches accordingly

    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether any caches were cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.CACHE_MEMORY_CONSTRAINTS =',
                        CACHE_MEMORY_CONSTRAINTS)

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(CACHE_MEMORY_CONSTRAINTS):
        _messages.debug_log(f'{m_name}.CACHE_MEMORY_CONSTRAINTS '
                            f'{CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_model_cache)}.')

        clear_model_cache(collect=collect)
        return True

    return False


def enforce_pipeline_cache_constraints(new_pipeline_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS` and clear the
    :py:class:`diffusers.DiffusionPipeline` cache if needed.

    :param new_pipeline_size: estimated size in bytes of any new pipeline that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.PIPELINE_CACHE_MEMORY_CONSTRAINTS =',
                        PIPELINE_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(pipeline_cache_size())},',
                        f'pipeline_size = {_memory.bytes_best_human_unit(new_pipeline_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(PIPELINE_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': pipeline_cache_size(),
                                              'pipeline_size': new_pipeline_size}):
        _messages.debug_log(f'{m_name}.PIPELINE_CACHE_MEMORY_CONSTRAINTS '
                            f'{PIPELINE_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_pipeline_cache)}.')
        clear_pipeline_cache(collect=collect)
        return True
    return False


def enforce_vae_cache_constraints(new_vae_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS` and clear the
    VAE cache if needed.

    :param new_vae_size: estimated size in bytes of any new vae that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.VAE_CACHE_MEMORY_CONSTRAINTS =',
                        VAE_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(vae_cache_size())},',
                        f'vae_size = {_memory.bytes_best_human_unit(new_vae_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(VAE_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': vae_cache_size(),
                                              'vae_size': new_vae_size}):
        _messages.debug_log(f'{m_name}.VAE_CACHE_MEMORY_CONSTRAINTS '
                            f'{VAE_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_vae_cache)}.')

        clear_vae_cache(collect=collect)
        return True
    return False


def enforce_unet_cache_constraints(new_unet_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS` and clear the
    UNet cache if needed.

    :param new_unet_size: estimated size in bytes of any new unet that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.UNET_CACHE_MEMORY_CONSTRAINTS =',
                        UNET_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(unet_cache_size())},',
                        f'unet_size = {_memory.bytes_best_human_unit(new_unet_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(UNET_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': unet_cache_size(),
                                              'unet_size': new_unet_size}):
        _messages.debug_log(f'{m_name}.UNET_CACHE_MEMORY_CONSTRAINTS '
                            f'{UNET_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_unet_cache)}.')

        clear_unet_cache(collect=collect)
        return True
    return False


def enforce_control_net_cache_constraints(new_control_net_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS` and clear the
    ControlNet cache if needed.

    :param new_control_net_size: estimated size in bytes of any new control net that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS =',
                        CONTROL_NET_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(control_net_cache_size())},',
                        f'control_net_size = {_memory.bytes_best_human_unit(new_control_net_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(CONTROL_NET_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': control_net_cache_size(),
                                              'control_net_size': new_control_net_size}):
        _messages.debug_log(f'{m_name}.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS '
                            f'{CONTROL_NET_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_control_net_cache)}.')

        clear_control_net_cache(collect=collect)
        return True
    return False


def uri_hash_with_parser(parser):
    """
    Create a hash function from a particular URI parser function that hashes a URI string.

    The URI is parsed and then the object that results from parsing is hashed with
    :py:func:`dgenerate.memoize.struct_hasher`.

    :param parser: The URI parser function
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`
    """

    def hasher(path):
        if not path:
            return path

        return _d_memoize.struct_hasher(parser(path))

    return hasher


def uri_list_hash_with_parser(parser):
    """
    Create a hash function from a particular URI parser function that hashes a list of URIs.

    :param parser: The URI parser function
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`
    """

    def hasher(paths):
        if not paths:
            return '[]'

        return '[' + ','.join(uri_hash_with_parser(parser)(path) for path in paths) + ']'

    return hasher


def pipeline_create_update_cache_info(pipeline, estimated_size: int):
    """
    Add additional information about the size of a newly created :py:class:`diffusers.DiffusionPipeline` to the cache.

    Tag the object with an internal tag.

    :param pipeline: the :py:class:`diffusers.DiffusionPipeline` object
    :param estimated_size: size bytes
    """
    global _PIPELINE_CACHE_SIZE

    _PIPELINE_CACHE_SIZE += estimated_size

    # Tag for internal use

    pipeline.DGENERATE_SIZE_ESTIMATE = estimated_size


def controlnet_create_update_cache_info(controlnet, estimated_size: int):
    """
    Add additional information about the size of a newly created ControlNet model to the cache.

    Tag the object with an internal tag.

    :param controlnet: the ControlNet object
    :param estimated_size: size bytes
    """
    global _CONTROL_NET_CACHE_SIZE
    _CONTROL_NET_CACHE_SIZE += estimated_size

    # Tag for internal use
    controlnet.DGENERATE_SIZE_ESTIMATE = estimated_size


def vae_create_update_cache_info(vae, estimated_size: int):
    """
    Add additional information about the size of a newly created VAE model to the cache.

    Tag the object with an internal tag.

    :param vae: the VAE object
    :param estimated_size: size bytes
    """
    global _VAE_CACHE_SIZE
    _VAE_CACHE_SIZE += estimated_size

    # Tag for internal use
    vae.DGENERATE_SIZE_ESTIMATE = estimated_size


def unet_create_update_cache_info(unet, estimated_size: int):
    """
    Add additional information about the size of a newly created UNet model to the cache.

    Tag the object with an internal tag.

    :param unet: the UNet object
    :param estimated_size: size bytes
    """
    global _UNET_CACHE_SIZE
    _UNET_CACHE_SIZE += estimated_size

    # Tag for internal use
    unet.DGENERATE_SIZE_ESTIMATE = estimated_size


def pipeline_to_cpu_update_cache_info(pipeline: diffusers.DiffusionPipeline):
    """
    Update CPU side cache size information when a diffusers pipeline is moved to the CPU

    :param pipeline: the pipeline
    """

    global _PIPELINE_CACHE_SIZE

    enforce_pipeline_cache_constraints(pipeline.DGENERATE_SIZE_ESTIMATE)
    _PIPELINE_CACHE_SIZE += pipeline.DGENERATE_SIZE_ESTIMATE

    _messages.debug_log(f'{_types.class_and_id_string(pipeline)} '
                        f'Size = {pipeline.DGENERATE_SIZE_ESTIMATE} Bytes '
                        f'({_memory.bytes_best_human_unit(pipeline.DGENERATE_SIZE_ESTIMATE)}) '
                        f'is entering CPU side memory, {_types.fullname(pipeline_cache_size)}() '
                        f'is now {pipeline_cache_size()} Bytes '
                        f'({_memory.bytes_best_human_unit(pipeline.DGENERATE_SIZE_ESTIMATE)})')


def unet_to_cpu_update_cache_info(unet):
    """
    Update CPU side cache size information when a UNet module is moved to the CPU

    :param unet: the UNet
    """

    global _UNET_CACHE_SIZE

    if hasattr(unet, 'DGENERATE_SIZE_ESTIMATE'):
        # UNet returning to CPU side memory
        enforce_unet_cache_constraints(unet.DGENERATE_SIZE_ESTIMATE)
        _UNET_CACHE_SIZE += unet.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'{_types.class_and_id_string(unet)} '
                            f'Size = {unet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(unet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(unet_cache_size)}() '
                            f'is now {unet_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(unet.DGENERATE_SIZE_ESTIMATE)})')


def vae_to_cpu_update_cache_info(vae):
    """
    Update CPU side cache size information when a VAE module is moved to the CPU

    :param vae: the VAE
    """

    global _VAE_CACHE_SIZE
    if hasattr(vae, 'DGENERATE_SIZE_ESTIMATE'):
        # vae returning to CPU side memory
        enforce_vae_cache_constraints(vae.DGENERATE_SIZE_ESTIMATE)
        _VAE_CACHE_SIZE += vae.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'Cached VAE {_types.class_and_id_string(vae)} '
                            f'Size = {vae.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(vae.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(vae_cache_size)}() '
                            f'is now {vae_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(vae.DGENERATE_SIZE_ESTIMATE)})')


def controlnet_to_cpu_update_cache_info(
        controlnet: diffusers.models.ControlNetModel |
                    diffusers.pipelines.controlnet.MultiControlNetModel):
    """
    Update CPU side cache size information when a ControlNet module is moved to the CPU

    :param controlnet: the control net, or multi control net
    """

    global _CONTROL_NET_CACHE_SIZE

    if isinstance(controlnet,
                  diffusers.pipelines.controlnet.MultiControlNetModel):
        total_size = 0
        for control_net in controlnet.nets:
            total_size += control_net.DGENERATE_SIZE_ESTIMATE

            _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(control_net)} '
                                f'Size = {control_net.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(control_net.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiControlNetModel" is entering CPU side memory.')

        enforce_control_net_cache_constraints(total_size)
        _CONTROL_NET_CACHE_SIZE += total_size

        _messages.debug_log(f'"MultiControlNetModel" size fully estimated, '
                            f'{_types.fullname(control_net_cache_size)}() '
                            f'is now {control_net_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(control_net_cache_size())})')

    else:
        # ControlNet returning to CPU side memory
        enforce_control_net_cache_constraints(controlnet.DGENERATE_SIZE_ESTIMATE)
        _CONTROL_NET_CACHE_SIZE += controlnet.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} '
                            f'Size = {controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(control_net_cache_size)}() '
                            f'is now {control_net_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(control_net_cache_size())})')


def pipeline_off_cpu_update_cache_info(
        pipeline: diffusers.DiffusionPipeline | diffusers.FlaxDiffusionPipeline):
    """
    Update CPU side cache size information when a diffusers pipeline is moved to a device that is not the CPU

    :param pipeline: the pipeline
    """

    global _PIPELINE_CACHE_SIZE

    _PIPELINE_CACHE_SIZE -= pipeline.DGENERATE_SIZE_ESTIMATE

    if _PIPELINE_CACHE_SIZE < 0:
        _PIPELINE_CACHE_SIZE = 0

    _messages.debug_log(f'Cached Diffusers Pipeline {_types.class_and_id_string(pipeline)} '
                        f'Size = {pipeline.DGENERATE_SIZE_ESTIMATE} Bytes '
                        f'({_memory.bytes_best_human_unit(pipeline.DGENERATE_SIZE_ESTIMATE)}) '
                        f'is leaving CPU side memory, {_types.fullname(pipeline_cache_size)}() '
                        f'is now {pipeline_cache_size()} Bytes '
                        f'({_memory.bytes_best_human_unit(pipeline_cache_size())})')


def unet_off_cpu_update_cache_info(unet):
    """
    Update CPU side cache size information when a UNet module is moved to a device that is not the CPU

    :param unet: the UNet
    """

    global _UNET_CACHE_SIZE

    if hasattr(unet, 'DGENERATE_SIZE_ESTIMATE'):
        _UNET_CACHE_SIZE -= unet.DGENERATE_SIZE_ESTIMATE

        if _UNET_CACHE_SIZE < 0:
            _UNET_CACHE_SIZE = 0

        _messages.debug_log(f'Cached UNet {_types.class_and_id_string(unet)} '
                            f'Size = {unet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(unet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(unet_cache_size)}() '
                            f'is now {unet_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(unet_cache_size())})')


def vae_off_cpu_update_cache_info(vae):
    """
    Update CPU side cache size information when a VAE module is moved to a device that is not the CPU

    :param vae: the VAE
    """
    global _VAE_CACHE_SIZE

    if hasattr(vae, 'DGENERATE_SIZE_ESTIMATE'):
        _VAE_CACHE_SIZE -= vae.DGENERATE_SIZE_ESTIMATE

        if _VAE_CACHE_SIZE < 0:
            _VAE_CACHE_SIZE = 0

        _messages.debug_log(f'Cached VAE {_types.class_and_id_string(vae)} '
                            f'Size = {vae.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(vae.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(vae_cache_size)}() '
                            f'is now {vae_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(vae_cache_size())})')


def controlnet_off_cpu_update_cache_info(
        controlnet: diffusers.models.ControlNetModel |
                    diffusers.pipelines.controlnet.MultiControlNetModel):
    """
    Update CPU side cache size information when a ControlNet module is moved to a device that is not the CPU

    :param controlnet: the control net, or multi control net
    """
    global _CONTROL_NET_CACHE_SIZE

    if isinstance(controlnet, diffusers.pipelines.controlnet.MultiControlNetModel):
        for control_net in controlnet.nets:
            _CONTROL_NET_CACHE_SIZE -= control_net.DGENERATE_SIZE_ESTIMATE

            if _CONTROL_NET_CACHE_SIZE < 0:
                _CONTROL_NET_CACHE_SIZE = 0

            _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(control_net)} Size = '
                                f'{control_net.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(control_net.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiControlNetModel" is leaving CPU side memory, '
                                f'{_types.fullname(control_net_cache_size)}() is now '
                                f'{control_net_cache_size()} Bytes '
                                f'({_memory.bytes_best_human_unit(control_net_cache_size())})')

    elif isinstance(controlnet, diffusers.models.ControlNetModel):
        _CONTROL_NET_CACHE_SIZE -= controlnet.DGENERATE_SIZE_ESTIMATE

        if _CONTROL_NET_CACHE_SIZE < 0:
            _CONTROL_NET_CACHE_SIZE = 0

        _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} '
                            f'Size = {controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(control_net_cache_size)}() '
                            f'is now {control_net_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(control_net_cache_size())})')


__all__ = _types.module_all()
