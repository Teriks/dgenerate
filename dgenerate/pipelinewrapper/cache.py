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

import diffusers

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.types as _types

_TEXT_ENCODER_CACHE = dict()
"""Global in memory cache for torch text encoders"""

_PIPELINE_CACHE = dict()
"""Global in memory cache for torch diffusers pipelines"""

_PIPELINE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all pipelines cached in memory"""

_CONTROLNET_CACHE = dict()
"""Global in memory cache for torch ControlNet models"""

_ADAPTER_CACHE = dict()
"""Global in memory cache for torch T2IAdapter models"""

_ADAPTER_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all T2IAdapter models cached in memory"""

_CONTROLNET_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all ControlNet models cached in memory"""

_VAE_CACHE = dict()
"""Global in memory cache for torch VAE models"""

_UNET_CACHE = dict()
"""Global in memory cache for torch UNet models"""

_VAE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all VAE models cached in memory"""

_UNET_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all UNet models cached in memory"""

_TEXT_ENCODER_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all Text Encoder models cached in memory"""

_IMAGE_ENCODER_CACHE = dict()
"""Global in memory cache for image encoder models"""

_IMAGE_ENCODER_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all Image Encoder models cached in memory"""

_TRANSFORMER_CACHE = dict()
"""Global in-memory cache for transformer models"""

_TRANSFORMER_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all transformer models cached in memory"""


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


def controlnet_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified ControlNet models currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _CONTROLNET_CACHE_SIZE


def adapter_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified T2IAdapter models currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _ADAPTER_CACHE_SIZE


def text_encoder_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified Text Encoder models currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _TEXT_ENCODER_CACHE_SIZE


def image_encoder_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified Image Encoders currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _IMAGE_ENCODER_CACHE_SIZE


def transformer_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all transformer models currently cached in memory.

    :return: memory usage in bytes.
    """
    return _TRANSFORMER_CACHE_SIZE


CACHE_MEMORY_CONSTRAINTS: list[str] = ['used_percent > 70']
"""
Cache constraint expressions for when to clear all model caches (DiffusionPipeline, VAE, ControlNet, and Text Encoder), 
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

CONTROLNET_CACHE_MEMORY_CONSTRAINTS: list[str] = ['controlnet_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the ControlNet cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_controlnet_cache_constraints` will call
:py:func:`.clear_controlnet_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``controlnet_size`` (the estimated size of the new ControlNet before it is brought into memory, in bytes)
"""

ADAPTER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['adapter_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the T2IAdapter cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_adapter_cache_constraints` will call
:py:func:`.clear_adapter_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``adapter_size`` (the estimated size of the new T2IAdapter before it is brought into memory, in bytes)
"""

TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['text_encoder_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the Text Encoder cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_text_encoder_cache_constraints` will call
:py:func:`.clear_text_encoder_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``text_encoder_size`` (the estimated size of the new Text Encoder before it is brought into memory, in bytes)
"""

IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['image_encoder_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the Image Encoder cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_image_encoder_cache_constraints` will call
:py:func:`.clear_image_encoder_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``image_encoder_size`` (the estimated size of the new Image Encoder before it is brought into memory, in bytes)
"""

TRANSFORMER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['transformer_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the transformer model cache, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, a call to :py:func:`.enforce_transformer_cache_constraints` will call
:py:func:`.clear_transformer_cache` and force a garbage collection.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``transformer_size`` (the estimated size of the new transformer model before it is brought into memory, in bytes)
"""


def clear_pipeline_cache(collect=True):
    """
    Clear DiffusionPipeline cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _PIPELINE_CACHE, \
        _PIPELINE_CACHE_SIZE

    _PIPELINE_CACHE.clear()
    _PIPELINE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_pipeline_cache)} calling gc.collect() by request')

        gc.collect()


def clear_controlnet_cache(collect=True):
    """
    Clear ControlNet cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _CONTROLNET_CACHE, \
        _CONTROLNET_CACHE_SIZE

    _CONTROLNET_CACHE.clear()

    _CONTROLNET_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_controlnet_cache)} calling gc.collect() by request')

        gc.collect()


def clear_adapter_cache(collect=True):
    """
    Clear T2IAdapter cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _ADAPTER_CACHE, \
        _ADAPTER_CACHE_SIZE

    _ADAPTER_CACHE.clear()

    _ADAPTER_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_adapter_cache)} calling gc.collect() by request')

        gc.collect()


def clear_vae_cache(collect=True):
    """
    Clear VAE cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _VAE_CACHE, \
        _VAE_CACHE_SIZE

    _VAE_CACHE.clear()

    _VAE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_vae_cache)} calling gc.collect() by request')

        gc.collect()


def clear_text_encoder_cache(collect=True):
    """
    Clear Text Encoder cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TEXT_ENCODER_CACHE, \
        _TEXT_ENCODER_CACHE_SIZE

    _TEXT_ENCODER_CACHE.clear()

    _TEXT_ENCODER_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_text_encoder_cache)} calling gc.collect() by request')

        gc.collect()


def clear_unet_cache(collect=True):
    """
    Clear UNet cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _UNET_CACHE, \
        _UNET_CACHE_SIZE

    _UNET_CACHE.clear()

    _UNET_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_unet_cache)} calling gc.collect() by request')

        gc.collect()


def clear_image_encoder_cache(collect=True):
    """
    Clear Image Encoder cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _IMAGE_ENCODER_CACHE, \
        _IMAGE_ENCODER_CACHE_SIZE

    _IMAGE_ENCODER_CACHE.clear()
    _IMAGE_ENCODER_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_image_encoder_cache)} calling gc.collect() by request')

        gc.collect()


def clear_transformer_cache(collect=True):
    """
    Clear transformer model cache and then garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _TRANSFORMER_CACHE, _TRANSFORMER_CACHE_SIZE

    _TRANSFORMER_CACHE.clear()
    _TRANSFORMER_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_transformer_cache)} calling gc.collect() by request')
        gc.collect()


def clear_model_cache(collect=True):
    """
    Clear all in-memory model caches and garbage collect.

    :param collect: Call :py:func:`gc.collect` ?
    """
    global _PIPELINE_CACHE, \
        _CONTROLNET_CACHE, \
        _UNET_CACHE, \
        _VAE_CACHE, \
        _TEXT_ENCODER_CACHE, \
        _ADAPTER_CACHE, \
        _IMAGE_ENCODER_CACHE, \
        _TRANSFORMER_CACHE, \
        _PIPELINE_CACHE_SIZE, \
        _CONTROLNET_CACHE_SIZE, \
        _UNET_CACHE_SIZE, \
        _VAE_CACHE_SIZE, \
        _TEXT_ENCODER_CACHE_SIZE, \
        _ADAPTER_CACHE_SIZE, \
        _IMAGE_ENCODER_CACHE_SIZE, \
        _TRANSFORMER_CACHE_SIZE

    _PIPELINE_CACHE.clear()
    _CONTROLNET_CACHE.clear()
    _UNET_CACHE.clear()
    _VAE_CACHE.clear()
    _TEXT_ENCODER_CACHE.clear()
    _ADAPTER_CACHE.clear()
    _IMAGE_ENCODER_CACHE.clear()
    _TRANSFORMER_CACHE.clear()

    _PIPELINE_CACHE_SIZE = 0
    _CONTROLNET_CACHE_SIZE = 0
    _UNET_CACHE_SIZE = 0
    _VAE_CACHE_SIZE = 0
    _TEXT_ENCODER_CACHE_SIZE = 0
    _ADAPTER_CACHE_SIZE = 0
    _IMAGE_ENCODER_CACHE_SIZE = 0
    _TRANSFORMER_CACHE_SIZE = 0

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


def enforce_text_encoder_cache_constraints(new_text_encoder_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS` and clear the
    Text Encoder cache if needed.

    :param new_text_encoder_size: estimated size in bytes of any new text encoder that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS =',
                        TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(text_encoder_cache_size())},',
                        f'text_encoder_size = {_memory.bytes_best_human_unit(new_text_encoder_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': text_encoder_cache_size(),
                                              'text_encoder_size': new_text_encoder_size}):
        _messages.debug_log(f'{m_name}.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS '
                            f'{TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_text_encoder_cache)}.')

        clear_text_encoder_cache(collect=collect)
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


def enforce_controlnet_cache_constraints(new_controlnet_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS` and clear the
    ControlNet cache if needed.

    :param new_controlnet_size: estimated size in bytes of any new control net that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.CONTROLNET_CACHE_MEMORY_CONSTRAINTS =',
                        CONTROLNET_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(controlnet_cache_size())},',
                        f'controlnet_size = {_memory.bytes_best_human_unit(new_controlnet_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(CONTROLNET_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': controlnet_cache_size(),
                                              'controlnet_size': new_controlnet_size}):
        _messages.debug_log(f'{m_name}.CONTROLNET_CACHE_MEMORY_CONSTRAINTS '
                            f'{CONTROLNET_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_controlnet_cache)}.')

        clear_controlnet_cache(collect=collect)
        return True
    return False


def enforce_adapter_cache_constraints(new_adapter_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS` and clear the
    ControlNet cache if needed.

    :param new_adapter_size: estimated size in bytes of any new t2i adapter that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.ADAPTER_CACHE_MEMORY_CONSTRAINTS =',
                        ADAPTER_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(adapter_cache_size())},',
                        f'adapter_size = {_memory.bytes_best_human_unit(new_adapter_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(ADAPTER_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': adapter_cache_size(),
                                              'adapter_size': new_adapter_size}):
        _messages.debug_log(f'{m_name}.ADAPTER_CACHE_MEMORY_CONSTRAINTS '
                            f'{ADAPTER_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_adapter_cache)}.')

        clear_adapter_cache(collect=collect)
        return True
    return False


def enforce_image_encoder_cache_constraints(new_image_encoder_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS` and clear the
    Image Encoder cache if needed.

    :param new_image_encoder_size: estimated size in bytes of any new image encoder that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS =',
                        IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(image_encoder_cache_size())},',
                        f'image_encoder_size = {_memory.bytes_best_human_unit(new_image_encoder_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': image_encoder_cache_size(),
                                              'image_encoder_size': new_image_encoder_size}):
        _messages.debug_log(f'{m_name}.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS '
                            f'{IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_image_encoder_cache)}.')

        clear_image_encoder_cache(collect=collect)
        return True
    return False


def enforce_transformer_cache_constraints(new_transformer_size, collect=True):
    """
    Enforce :py:attr:`dgenerate.pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS` and clear the
    transformer model cache if needed.

    :param new_transformer_size: estimated size in bytes of any new transformer model that is about to enter memory
    :param collect: Call :py:func:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS =',
                        TRANSFORMER_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(transformer_cache_size())},',
                        f'transformer_size = {_memory.bytes_best_human_unit(new_transformer_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(TRANSFORMER_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': transformer_cache_size(),
                                              'transformer_size': new_transformer_size}):
        _messages.debug_log(f'{m_name}.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS '
                            f'{TRANSFORMER_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_transformer_cache)}.')
        clear_transformer_cache(collect=collect)
        return True
    return False


def uri_hash_with_parser(parser, exclude: set[str] | None = None):
    """
    Create a hash function from a particular URI parser function that hashes a URI string.

    The URI is parsed and then the object that results from parsing is hashed with
    :py:func:`dgenerate.memoize.struct_hasher`.

    If the parser returns a string, it is regarded as the hash value instead of being
    passed to :py:func:`dgenerate.memoize.struct_hasher`.

    :param parser: The URI parser function
    :param exclude: URI argument names to exclude from hashing
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`
    """

    def hasher(path):
        if not path:
            # None gets hashed to 'None' string
            return str(path)

        parser_result = parser(path)

        if isinstance(parser_result, str):
            # user returned string, override the hash value
            return parser_result

        return _d_memoize.struct_hasher(parser_result,
                                        exclude=exclude)

    return hasher


def uri_list_hash_with_parser(parser, exclude: set[str] | None = None):
    """
    Create a hash function from a particular URI parser function that hashes a list of URIs.

    :param parser: The URI parser function
    :param exclude: URI argument names to exclude from hashing
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`

    """

    def hasher(paths):
        if not paths:
            return '[]'

        return '[' + ','.join(uri_hash_with_parser(parser, exclude=exclude)(path) for path in paths) + ']'

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
    global _CONTROLNET_CACHE_SIZE
    _CONTROLNET_CACHE_SIZE += estimated_size

    # Tag for internal use
    controlnet.DGENERATE_SIZE_ESTIMATE = estimated_size


def adapter_create_update_cache_info(adapter, estimated_size: int):
    """
    Add additional information about the size of a newly created T2IAdapter model to the cache.

    Tag the object with an internal tag.

    :param adapter: the T2IAdapter object
    :param estimated_size: size bytes
    """
    global _ADAPTER_CACHE_SIZE
    _ADAPTER_CACHE_SIZE += estimated_size

    # Tag for internal use
    adapter.DGENERATE_SIZE_ESTIMATE = estimated_size


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


def text_encoder_create_update_cache_info(text_encoder, estimated_size: int):
    """
    Add additional information about the size of a newly created Text Encoder model to the cache.

    Tag the object with an internal tag.

    :param text_encoder: the Text Encoder object
    :param estimated_size: size bytes
    """
    global _TEXT_ENCODER_CACHE_SIZE
    _TEXT_ENCODER_CACHE_SIZE += estimated_size

    # Tag for internal use
    text_encoder.DGENERATE_SIZE_ESTIMATE = estimated_size


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


def image_encoder_create_update_cache_info(image_encoder, estimated_size: int):
    """
    Add additional information about the size of a newly created Image Encoder model to the cache.

    Tag the object with an internal tag.

    :param image_encoder: the Image Encoder object
    :param estimated_size: size bytes
    """
    global _IMAGE_ENCODER_CACHE_SIZE
    _IMAGE_ENCODER_CACHE_SIZE += estimated_size

    # Tag for internal use
    image_encoder.DGENERATE_SIZE_ESTIMATE = estimated_size


def transformer_create_update_cache_info(transformer, estimated_size: int):
    """
    Add additional information about the size of a newly created transformer model to the cache.

    Tag the object with an internal tag.

    :param transformer: the transformer object
    :param estimated_size: size bytes
    """
    global _TRANSFORMER_CACHE_SIZE
    _TRANSFORMER_CACHE_SIZE += estimated_size

    # Tag for internal use
    transformer.DGENERATE_SIZE_ESTIMATE = estimated_size


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
                            f'({_memory.bytes_best_human_unit(vae_cache_size())})')


def text_encoder_to_cpu_update_cache_info(text_encoder):
    """
    Update CPU side cache size information when a Text Encoder module is moved to the CPU

    :param text_encoder: the Text Encoder
    """

    global _TEXT_ENCODER_CACHE_SIZE
    if hasattr(text_encoder, 'DGENERATE_SIZE_ESTIMATE'):
        # text_encoder returning to CPU side memory
        enforce_text_encoder_cache_constraints(text_encoder.DGENERATE_SIZE_ESTIMATE)
        _TEXT_ENCODER_CACHE_SIZE += text_encoder.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'Cached TextEncoder {_types.class_and_id_string(text_encoder)} '
                            f'Size = {text_encoder.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(text_encoder.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(text_encoder_cache_size)}() '
                            f'is now {text_encoder_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(text_encoder_cache_size())})')


def image_encoder_to_cpu_update_cache_info(image_encoder):
    """
    Update CPU side cache size information when an Image Encoder module is moved to the CPU

    :param image_encoder: the Image Encoder
    """

    global _IMAGE_ENCODER_CACHE_SIZE

    if hasattr(image_encoder, 'DGENERATE_SIZE_ESTIMATE'):
        # Image Encoder returning to CPU side memory
        enforce_image_encoder_cache_constraints(image_encoder.DGENERATE_SIZE_ESTIMATE)
        _IMAGE_ENCODER_CACHE_SIZE += image_encoder.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'{_types.class_and_id_string(image_encoder)} '
                            f'Size = {image_encoder.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(image_encoder.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(image_encoder_cache_size)}() '
                            f'is now {image_encoder_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(image_encoder_cache_size())})')


def controlnet_to_cpu_update_cache_info(
        controlnet: diffusers.models.ControlNetModel |
                    diffusers.pipelines.controlnet.MultiControlNetModel |
                    diffusers.SD3ControlNetModel |
                    diffusers.SD3MultiControlNetModel):
    """
    Update CPU side cache size information when a ControlNet module is moved to the CPU

    :param controlnet: the control net, or multi control net
    """

    global _CONTROLNET_CACHE_SIZE

    if isinstance(controlnet,
                  (diffusers.pipelines.controlnet.MultiControlNetModel,
                   diffusers.SD3MultiControlNetModel)):

        total_size = 0
        for controlnet in controlnet.nets:
            total_size += controlnet.DGENERATE_SIZE_ESTIMATE

            _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} '
                                f'Size = {controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiControlNetModel" is entering CPU side memory.')

        enforce_controlnet_cache_constraints(total_size)
        _CONTROLNET_CACHE_SIZE += total_size

        _messages.debug_log(f'"MultiControlNetModel" size fully estimated, '
                            f'{_types.fullname(controlnet_cache_size)}() '
                            f'is now {controlnet_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet_cache_size())})')

    elif isinstance(controlnet, (diffusers.models.ControlNetModel,
                                 diffusers.SD3ControlNetModel)):

        # ControlNet returning to CPU side memory
        enforce_controlnet_cache_constraints(controlnet.DGENERATE_SIZE_ESTIMATE)
        _CONTROLNET_CACHE_SIZE += controlnet.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} '
                            f'Size = {controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(controlnet_cache_size)}() '
                            f'is now {controlnet_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet_cache_size())})')


def adapter_to_cpu_update_cache_info(adapter: diffusers.T2IAdapter | diffusers.MultiAdapter):
    """
    Update CPU side cache size information when a T2IAdapter module is moved to the CPU

    :param adapter: the adapter, or multi adapter
    """

    global _ADAPTER_CACHE_SIZE

    if isinstance(adapter, diffusers.MultiAdapter):

        total_size = 0
        for adapter in adapter.adapters:
            total_size += adapter.DGENERATE_SIZE_ESTIMATE

            _messages.debug_log(f'Cached T2IAdapter {_types.class_and_id_string(adapter)} '
                                f'Size = {adapter.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(adapter.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiAdapter" is entering CPU side memory.')

        enforce_adapter_cache_constraints(total_size)
        _ADAPTER_CACHE_SIZE += total_size

        _messages.debug_log(f'"MultiAdapter" size fully estimated, '
                            f'{_types.fullname(adapter_cache_size)}() '
                            f'is now {adapter_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(adapter_cache_size())})')

    elif isinstance(adapter, diffusers.T2IAdapter):

        enforce_adapter_cache_constraints(adapter.DGENERATE_SIZE_ESTIMATE)
        _ADAPTER_CACHE_SIZE += adapter.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'Cached T2IAdapter {_types.class_and_id_string(adapter)} '
                            f'Size = {adapter.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(adapter.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(adapter_cache_size)}() '
                            f'is now {adapter_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(adapter_cache_size())})')


def transformer_to_cpu_update_cache_info(transformer):
    """
    Update CPU side cache size information when a transformer module is moved to the CPU

    :param transformer: the transformer
    """

    global _TRANSFORMER_CACHE_SIZE

    if hasattr(transformer, 'DGENERATE_SIZE_ESTIMATE'):
        # Transformer returning to CPU side memory
        enforce_transformer_cache_constraints(transformer.DGENERATE_SIZE_ESTIMATE)
        _TRANSFORMER_CACHE_SIZE += transformer.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'{_types.class_and_id_string(transformer)} '
                            f'Size = {transformer.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(transformer.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is entering CPU side memory, {_types.fullname(transformer_cache_size)}() '
                            f'is now {transformer_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(transformer_cache_size())})')


def pipeline_off_cpu_update_cache_info(
        pipeline: diffusers.DiffusionPipeline):
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


def text_encoder_off_cpu_update_cache_info(text_encoder):
    """
    Update CPU side cache size information when a Text Encoder module is moved to a device that is not the CPU

    :param text_encoder: the Text Encoder
    """
    global _TEXT_ENCODER_CACHE_SIZE

    if hasattr(text_encoder, 'DGENERATE_SIZE_ESTIMATE'):
        _TEXT_ENCODER_CACHE_SIZE -= text_encoder.DGENERATE_SIZE_ESTIMATE

        if _TEXT_ENCODER_CACHE_SIZE < 0:
            _TEXT_ENCODER_CACHE_SIZE = 0

        _messages.debug_log(f'Cached TextEncoder {_types.class_and_id_string(text_encoder)} '
                            f'Size = {text_encoder.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(text_encoder.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(text_encoder_cache_size)}() '
                            f'is now {text_encoder_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(text_encoder_cache_size())})')


def image_encoder_off_cpu_update_cache_info(image_encoder):
    """
    Update CPU side cache size information when an Image Encoder module is moved to a device that is not the CPU

    :param image_encoder: the Image Encoder
    """
    global _IMAGE_ENCODER_CACHE_SIZE

    if hasattr(image_encoder, 'DGENERATE_SIZE_ESTIMATE'):
        _IMAGE_ENCODER_CACHE_SIZE -= image_encoder.DGENERATE_SIZE_ESTIMATE

        if _IMAGE_ENCODER_CACHE_SIZE < 0:
            _IMAGE_ENCODER_CACHE_SIZE = 0

        _messages.debug_log(f'Cached ImageEncoder {_types.class_and_id_string(image_encoder)} '
                            f'Size = {image_encoder.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(image_encoder.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(image_encoder_cache_size)}() '
                            f'is now {image_encoder_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(image_encoder_cache_size())})')


def controlnet_off_cpu_update_cache_info(
        controlnet: diffusers.models.ControlNetModel |
                    diffusers.pipelines.controlnet.MultiControlNetModel |
                    diffusers.SD3ControlNetModel |
                    diffusers.SD3MultiControlNetModel):
    """
    Update CPU side cache size information when a ControlNet module is moved to a device that is not the CPU

    :param controlnet: the control net, or multi control net
    """
    global _CONTROLNET_CACHE_SIZE

    if isinstance(controlnet, (diffusers.pipelines.controlnet.MultiControlNetModel,
                               diffusers.SD3MultiControlNetModel)):

        for controlnet in controlnet.nets:
            _CONTROLNET_CACHE_SIZE -= controlnet.DGENERATE_SIZE_ESTIMATE

            if _CONTROLNET_CACHE_SIZE < 0:
                _CONTROLNET_CACHE_SIZE = 0

            _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} Size = '
                                f'{controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiControlNetModel" is leaving CPU side memory, '
                                f'{_types.fullname(controlnet_cache_size)}() is now '
                                f'{controlnet_cache_size()} Bytes '
                                f'({_memory.bytes_best_human_unit(controlnet_cache_size())})')

    elif isinstance(controlnet, (diffusers.models.ControlNetModel,
                                 diffusers.SD3ControlNetModel)):

        _CONTROLNET_CACHE_SIZE -= controlnet.DGENERATE_SIZE_ESTIMATE

        if _CONTROLNET_CACHE_SIZE < 0:
            _CONTROLNET_CACHE_SIZE = 0

        _messages.debug_log(f'Cached ControlNetModel {_types.class_and_id_string(controlnet)} '
                            f'Size = {controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(controlnet_cache_size)}() '
                            f'is now {controlnet_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(controlnet_cache_size())})')


def adapter_off_cpu_update_cache_info(adapter: diffusers.T2IAdapter | diffusers.MultiAdapter):
    """
    Update CPU side cache size information when a T2IAdapter module is moved to a device that is not the CPU

    :param adapter: the adapter, or multi adapter
    """
    global _ADAPTER_CACHE_SIZE

    if isinstance(adapter, diffusers.MultiAdapter):

        for adapter in adapter.adapters:
            _ADAPTER_CACHE_SIZE -= adapter.DGENERATE_SIZE_ESTIMATE

            if _ADAPTER_CACHE_SIZE < 0:
                _ADAPTER_CACHE_SIZE = 0

            _messages.debug_log(f'Cached T2IAdapter {_types.class_and_id_string(adapter)} Size = '
                                f'{adapter.DGENERATE_SIZE_ESTIMATE} Bytes '
                                f'({_memory.bytes_best_human_unit(adapter.DGENERATE_SIZE_ESTIMATE)}) '
                                f'from "MultiAdapter" is leaving CPU side memory, '
                                f'{_types.fullname(adapter_cache_size)}() is now '
                                f'{adapter_cache_size()} Bytes '
                                f'({_memory.bytes_best_human_unit(adapter_cache_size())})')

    elif isinstance(adapter, diffusers.T2IAdapter):

        _ADAPTER_CACHE_SIZE -= adapter.DGENERATE_SIZE_ESTIMATE

        if _ADAPTER_CACHE_SIZE < 0:
            _ADAPTER_CACHE_SIZE = 0

        _messages.debug_log(f'Cached T2IAdapter {_types.class_and_id_string(adapter)} '
                            f'Size = {adapter.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(adapter.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(adapter_cache_size)}() '
                            f'is now {adapter_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(adapter_cache_size())})')


def transformer_off_cpu_update_cache_info(transformer):
    """
    Update CPU side cache size information when a transformer module is moved to a device that is not the CPU

    :param transformer: the transformer
    """

    global _TRANSFORMER_CACHE_SIZE

    if hasattr(transformer, 'DGENERATE_SIZE_ESTIMATE'):
        _TRANSFORMER_CACHE_SIZE -= transformer.DGENERATE_SIZE_ESTIMATE

        if _TRANSFORMER_CACHE_SIZE < 0:
            _TRANSFORMER_CACHE_SIZE = 0

        _messages.debug_log(f'Cached Transformer {_types.class_and_id_string(transformer)} '
                            f'Size = {transformer.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'({_memory.bytes_best_human_unit(transformer.DGENERATE_SIZE_ESTIMATE)}) '
                            f'is leaving CPU side memory, {_types.fullname(transformer_cache_size)}() '
                            f'is now {transformer_cache_size()} Bytes '
                            f'({_memory.bytes_best_human_unit(transformer_cache_size())})')


__all__ = _types.module_all()
