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

import huggingface_hub
import transformers

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_image_encoder_uri_parser = _textprocessing.ConceptUriParser(
    'ImageEncoder', ['revision', 'variant', 'subfolder', 'dtype'])


class ImageEncoderUri:
    """
    Representation of ``--image-encoder`` uri when ``--model-type`` torch*
    """

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path, or blob link
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidImageEncoderUriError: If ``model`` points to a single file,
            single file loads are not supported. Or if ``dtype`` is passed an
            invalid data type string.
        """

        if _util.is_single_file_model_load(model):
            raise _exceptions.InvalidImageEncoderUriError(
                'Loading an Image Encoder from a single file is not supported.')

        self._model = model
        self._revision = revision
        self._variant = variant

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidImageEncoderUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False):
        """
        Load an Image Encoder Model of type :py:class:`transformers.CLIPVisionModelWithProjection`

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

        :return: :py:class:`transformers.CLIPVisionModelWithProjection`
        """
        try:
            return self._load(
                dtype_fallback,
                use_auth_token,
                local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _util.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.ImageEncoderUriLoadError(
                f'error loading Image Encoder "{self.model}": {e}')

    @_memoize(_cache._IMAGE_ENCODER_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("ImageEncoder", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("ImageEncoder", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False):

        if sequential_cpu_offload_member and model_cpu_offload_member:
            # these are used for cache differentiation only
            raise ValueError('sequential_cpu_offload_member and model_cpu_offload_member cannot both be True.')

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        path = self.model

        estimated_memory_use = _util.estimate_model_memory_use(
            repo_id=path,
            revision=self.revision,
            variant=self.variant,
            subfolder=self.subfolder,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token
        )

        _cache.enforce_image_encoder_cache_constraints(new_image_encoder_size=estimated_memory_use)

        if self.subfolder:

            extra_args = {'subfolder': self.subfolder}
        else:
            # flux null reference bug
            extra_args = dict()

        image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(
            path,
            revision=self.revision,
            variant=self.variant,
            torch_dtype=torch_dtype,
            token=use_auth_token,
            local_files_only=local_files_only,
            **extra_args)

        _messages.debug_log('Estimated Image Encoder Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.image_encoder_create_update_cache_info(image_encoder=image_encoder,
                                                      estimated_size=estimated_memory_use)

        return image_encoder

    @staticmethod
    def parse(uri: _types.Uri) -> 'ImageEncoderUri':
        """
        Parse a ``--model-type`` torch* ``--image-encoder`` uri and return an object representing its constituents

        :param uri: string with ``--image-encoder`` uri syntax

        :raise InvalidImageEncoderUriError:

        :return: :py:class:`.ImageEncoderUri`
        """
        try:
            r = _image_encoder_uri_parser.parse(uri)

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidImageEncoderUriError(
                    f'Image Encoder "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return ImageEncoderUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                dtype=dtype,
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidImageEncoderUriError(e)
