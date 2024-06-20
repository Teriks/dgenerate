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

import huggingface_hub
import transformers.models.clip

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

_flax_text_encoder_uri_parser = _textprocessing.ConceptUriParser(
    'TextEncoder', ['model', 'revision', 'subfolder', 'dtype'])

try:
    import flax

    _have_flax = True
except ImportError:
    _have_flax = False


class FlaxTextEncoderUri:
    """
    Representation of ``--text-encoder*`` uri when ``--model-type`` flax*
    """

    @property
    def encoder(self) -> str:
        """
        Encoder class name such as "FlaxCLIPTextModel"
        """
        return self._encoder

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

    _encoders = {
        'FlaxCLIPTextModel': getattr(transformers.models.clip, 'FlaxCLIPTextModel', None)
    }

    @staticmethod
    def supported_encoder_names() -> list[str]:
        return list(FlaxTextEncoderUri._encoders.keys())

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None):
        """
        :param encoder: encoder class name, for example ``FlaxCLIPTextModel``
        :param model: model path
        :param revision: model revision (branch name)
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidTextEncoderUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

        if not _have_flax:
            raise _exceptions.InvalidTextEncoderUriError(
                'FlaxTextEncoderUri cannot be used without flax installed.')

        if encoder not in self._encoders:
            raise _exceptions.InvalidTextEncoderUriError(
                f'Unknown TextEncoder encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load:
            raise _exceptions.InvalidTextEncoderUriError(
                f'{encoder} is not capable of loading from a single file, '
                f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            if subfolder is not None:
                raise _exceptions.InvalidTextEncoderUriError(
                    'Single file TextEncoder loads do not support the subfolder option.')

        self._encoder = encoder
        self._model = model
        self._revision = revision
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidTextEncoderUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False) -> tuple['transformers.models.clip.FlaxCLIPTextModel', typing.Any]:
        """
        Load a flax Text Encoder of type :py:class:`transformers.models.clip.FlaxCLIPTextModel`,
        :py:class:`transformers.models.clip.FlaxCLIPTextModelWithProjection`, or
        :py:class:`transformers.models.t5.FlaxT5EncoderModel` from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?

        :raises ModelNotFoundError: If the model could not be found.

        :return: (:py:class:`transformers.models.clip.FlaxCLIPTextModel`, ``params``)
        """

        try:
            return self._load(dtype_fallback,
                              use_auth_token,
                              local_files_only,
                              sequential_cpu_offload_member,
                              model_cpu_offload_member)

        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.TextEncoderUriLoadError(
                f'error loading text encoder "{self.model}": {e}')

    @_memoize(_cache._FLAX_TEXT_ENCODER_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Flax TextEncoder", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Flax TextEncoder", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> tuple['transformers.models.clip.FlaxCLIPTextModel', typing.Any]:

        if self.dtype is None:
            flax_dtype = _enums.get_flax_dtype(dtype_fallback)
        else:
            flax_dtype = _enums.get_flax_dtype(self.dtype)

        encoder = self._encoders[self.encoder]

        model_path = _hfutil.download_non_hf_model(self.model)

        single_file_load_path = _hfutil.is_single_file_model_load(model_path)

        if single_file_load_path:
            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use)

            text_encoder = encoder.from_single_file(model_path,
                                                    token=use_auth_token,
                                                    revision=self.revision,
                                                    dtype=flax_dtype,
                                                    local_files_only=local_files_only)

        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use)

            text_encoder = encoder.from_pretrained(
                model_path,
                revision=self.revision,
                dtype=flax_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax TextEncoder Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.text_encoder_create_update_cache_info(
            text_encoder=text_encoder,
            estimated_size=estimated_memory_use)

        return text_encoder, text_encoder.params

    @staticmethod
    def parse(uri: _types.Uri) -> 'FlaxTextEncoderUri':
        """
        Parse a ``--model-type`` flax* ``--text-encoder*`` uri and return an object representing its constituents

        :param uri: string with ``--text-encoder*`` uri syntax

        :raise InvalidTextEncoderUriError:

        :return: :py:class:`.FlaxTextEncoderUri`
        """
        try:
            r = _flax_text_encoder_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidTextEncoderUriError(
                    'model argument for flax TextEncoder specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'Flax TextEncoder "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return FlaxTextEncoderUri(encoder=r.concept,
                                      model=model,
                                      revision=r.args.get('revision', None),
                                      dtype=dtype,
                                      subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTextEncoderUriError(e)
