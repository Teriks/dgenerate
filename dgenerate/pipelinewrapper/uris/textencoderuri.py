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
import optimum.quanto
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
import dgenerate.pipelinewrapper.quanto as _quanto

_text_encoder_uri_parser = _textprocessing.ConceptUriParser(
    'TextEncoder', ['model', 'revision', 'variant', 'subfolder', 'dtype', 'quantize'])


class TextEncoderUri:
    """
    Representation of ``--text-encoders*`` uri when ``--model-type`` torch*
    """

    @property
    def encoder(self) -> str:
        """
        Encoder class name such as "CLIPTextModel"
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
    def variant(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._variant

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
    def quantize(self) -> _types.OptionalString:
        """
        ``optimum.quanto`` Quantization dtype
        """
        return self._quantize

    _encoders = {
        'CLIPTextModel': transformers.models.clip.CLIPTextModel,
        'CLIPTextModelWithProjection': transformers.models.clip.CLIPTextModelWithProjection,
        'T5EncoderModel': transformers.models.t5.T5EncoderModel
    }

    @staticmethod
    def supported_encoder_names() -> list[str]:
        return list(TextEncoderUri._encoders.keys())

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None,
                 quantize: _types.OptionalString = None):
        """
        :param encoder: encoder class name, for example ``CLIPTextModel``
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param quantize: Quantize to a specific data type optimum-quanto,
            must be a supported dtype name that exists in ``optimum.quanto.qtypes``,
            such as qint8 or qfloat8

        :raises InvalidTextEncoderUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

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
        self._variant = variant
        self._subfolder = subfolder

        if quantize and quantize.lower() not in optimum.quanto.qtypes:
            raise _exceptions.InvalidTextEncoderUriError(
                f'Unknown quantize argument value "{quantize}", '
                f'must be one of: {_textprocessing.oxford_comma(optimum.quanto.qtypes.keys(), "or")} '
            )

        self._quantize = quantize.lower() if quantize else None

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidTextEncoderUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel]:
        """
        Load a torch Text Encoder of type :py:class:`transformers.models.clip.CLIPTextModel`,
        :py:class:`transformers.models.clip.CLIPTextModelWithProjection`, or
        :py:class:`transformers.models.t5.T5EncoderModel` from this URI

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`transformers.models.clip.CLIPTextModel`,
            :py:class:`transformers.models.clip.CLIPTextModelWithProjection`, or
            :py:class:`transformers.models.t5.T5EncoderModel`
        """

        try:
            return self._load(variant_fallback,
                              dtype_fallback,
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

    @_memoize(_cache._TEXT_ENCODER_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch TextEncoder", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch TextEncoder", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel]:

        if sequential_cpu_offload_member and model_cpu_offload_member:
            # these are used for cache differentiation only
            raise ValueError('sequential_cpu_offload_member and model_cpu_offload_member cannot both be True.')

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        if self.variant is None:
            variant = variant_fallback
        else:
            variant = self.variant

        encoder = self._encoders[self.encoder]

        model_path = _hfutil.download_non_hf_model(self.model)

        single_file_load_path = _hfutil.is_single_file_model_load(model_path)

        if single_file_load_path:
            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use)

            text_encoder = encoder.from_single_file(model_path,
                                                    token=use_auth_token,
                                                    revision=self.revision,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=local_files_only)

        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use)

            text_encoder = encoder.from_pretrained(
                model_path,
                revision=self.revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch TextEncoder Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.text_encoder_create_update_cache_info(
            text_encoder=text_encoder,
            estimated_size=estimated_memory_use)

        if self._quantize is not None:
            _quanto.quantize_freeze(text_encoder, weights=optimum.quanto.qtypes[self._quantize])

        return text_encoder

    @staticmethod
    def parse(uri: _types.Uri) -> 'TextEncoderUri':
        """
        Parse a ``--model-type`` torch* ``--text-encoders*`` uri and return an object representing its constituents

        :param uri: string with ``--text-encoders*`` uri syntax

        :raise InvalidTextEncoderUriError:

        :return: :py:class:`.TorchTextEncoderUri`
        """
        try:
            r = _text_encoder_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidTextEncoderUriError(
                    'model argument for torch TextEncoder specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'Torch TextEncoder "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return TextEncoderUri(encoder=r.concept,
                                  model=model,
                                  revision=r.args.get('revision', None),
                                  variant=r.args.get('variant', None),
                                  dtype=dtype,
                                  subfolder=r.args.get('subfolder', None),
                                  quantize=r.args.get('quantize', False))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTextEncoderUriError(e)
