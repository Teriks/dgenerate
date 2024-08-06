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

import diffusers
import huggingface_hub

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_flax_controlnet_uri_parser = _textprocessing.ConceptUriParser('ControlNet',
                                                               ['scale', 'revision', 'subfolder', 'dtype',
                                                                'from_torch'])


class FlaxControlNetUri:
    """
    Representation of ``--control-nets`` uri when ``--model-type`` flax*
    """

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    @property
    def dtype(self) -> _enums.DataType | None:
        """
        Model dtype (precision)
        """
        return self._dtype

    @property
    def scale(self) -> float:
        """
        ControlNet guidance scale
        """
        return self._scale

    @property
    def from_torch(self) -> bool:
        """
        Load from a model format meant for torch?
        """
        return self._from_torch

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: _enums.DataType | str | None = None,
                 scale: float = 1.0,
                 from_torch: bool = False):
        """
        :param model: model path
        :param revision: model revision (branch)
        :param subfolder: model subfolder
        :param dtype: data type (precision)
        :param scale: control net scale value
        :param from_torch: load from a model designed for torch?

        :raises InvalidControlNetUriError: If the ``model`` path represents a singular file (not supported),
            or if ``dtype`` is passed an invalid string
        """

        if _hfutil.is_single_file_model_load(model):
            raise _exceptions.InvalidControlNetUriError('Flax --control-nets do not support single file loads.')

        self._model = model
        self._revision = revision
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidControlNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._scale = scale
        self._from_torch = from_torch

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> tuple[diffusers.FlaxControlNetModel, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxControlNetModel` from this URI.

        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :raises ModelNotFoundError: If the model could not be found.

        :return: tuple (:py:class:`diffusers.FlaxControlNetModel`, flax_controlnet_params)
        """
        try:
            return self._load(dtype_fallback, use_auth_token, local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.ControlNetUriLoadError(
                f'error loading controlnet "{self.model}": {e}')

    @_memoize(_cache._FLAX_CONTROLNET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(
                  args, {'self': lambda o: _d_memoize.struct_hasher(
                      o, exclude={'scale'})}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Flax ControlNet", key, hit[0]),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Flax ControlNet", key, new[0]))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> tuple[diffusers.FlaxControlNetModel, typing.Any]:

        estimated_memory_usage = _hfutil.estimate_model_memory_use(
            repo_id=self.model,
            revision=self.revision,
            subfolder=self.subfolder,
            flax=not self.from_torch,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only
        )

        _cache.enforce_controlnet_cache_constraints(
            new_controlnet_size=estimated_memory_usage)

        flax_dtype = _enums.get_flax_dtype(
            dtype_fallback if self.dtype is None else self.dtype)

        new_net: diffusers.FlaxControlNetModel = \
            diffusers.FlaxControlNetModel.from_pretrained(
                self.model,
                revision=self.revision,
                subfolder=self.subfolder,
                dtype=flax_dtype,
                from_pt=self.from_torch,
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.controlnet_create_update_cache_info(controlnet=new_net[0],
                                                   estimated_size=estimated_memory_usage)

        return new_net

    @staticmethod
    def parse(uri: _types.Uri) -> 'FlaxControlNetUri':
        """
        Parse a ``--model-type`` flax* ``--control-nets`` uri specification and return an object representing its constituents

        :param uri: string with ``--control-nets`` uri syntax

        :raise InvalidControlNetUriError:

        :return: :py:class:`.FlaxControlNetPath`
        """
        try:
            r = _flax_controlnet_uri_parser.parse(uri)

            dtype = r.args.get('dtype')
            scale = r.args.get('scale', 1.0)
            from_torch = r.args.get('from_torch')

            if from_torch is not None:
                try:
                    from_torch = _types.parse_bool(from_torch)
                except ValueError:
                    raise _exceptions.InvalidControlNetUriError(
                        f'Flax ControlNet from_torch must be undefined or boolean (true or false), received: {from_torch}')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidControlNetUriError(
                    f'Flax ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidControlNetUriError(
                    f'Flax ControlNet scale must be a floating point number, received {scale}')

            return FlaxControlNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                subfolder=r.args.get('subfolder', None),
                scale=scale,
                dtype=dtype,
                from_torch=from_torch)

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidControlNetUriError(e)
