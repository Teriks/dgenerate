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
import contextlib
import logging
import typing

import diffusers.pipelines.kolors
import huggingface_hub
import torch
import transformers.models.clip
import diffusers.loaders
import accelerate

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris import util as _util
import dgenerate.extras.DistillT5.models.T5_encoder as _distill_t5_encoder

DistillT5EncoderModel = _distill_t5_encoder.T5EncoderWithProjection

_text_encoder_uri_parser = _textprocessing.ConceptUriParser(
    'TextEncoder', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer',
        'mode'
    ]
)

_text_encoder_cache = _d_memoize.create_object_cache(
    'text_encoder', cache_type=_memory.SizedConstrainedObjectCache
)


@contextlib.contextmanager
def _suppress_accelerate_warnings():
    """Context manager to temporarily suppress Accelerate warnings."""
    # Get the logger
    logger = logging.getLogger("accelerate.utils.modeling")
    # Store original level
    original_level = logger.level
    # Set level to ERROR (suppressing warnings)
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        # Restore original level
        logger.setLevel(original_level)


def _load_clip_l_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        local_files_only: bool = False,
        quantization_config=None
):
    model_id = "openai/clip-vit-large-patch14"
    
    try:

        with _suppress_accelerate_warnings():
            config = transformers.CLIPTextConfig.from_pretrained(
                model_id, local_files_only=local_files_only,
                quantization_config=quantization_config
            )

            config.torch_dtype = dtype

            # Load with automatic device mapping for efficient memory usage
            text_encoder = model_class(config)
        
        blob_link = _pipelinewrapper_util.HFBlobLink.parse(model_path)
        
        if blob_link is not None:
            model_path = huggingface_hub.hf_hub_download(
                repo_id=blob_link.repo_id,
                revision=blob_link.revision,
                subfolder=blob_link.subfolder,
                filename=blob_link.weight_name,
                local_files_only=local_files_only
            )

        _messages.log(
            f'Loading monolithic clip-l state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = accelerate.load_checkpoint_and_dispatch(
                text_encoder,
                checkpoint=model_path,
                device_map="auto",
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        return text_encoder
    
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(f"Failed to load CLIP-L model: {e}") from e


def _load_t5_xxl_from_single_file(
        model_class: transformers.models.t5.T5EncoderModel | DistillT5EncoderModel,
        model_path: str,
        dtype: torch.dtype,
        local_files_only: bool = False,
        quantization_config=None
):
    model_id = "google/t5-v1_1-xxl"

    try:
        # Following Flux examples, load T5-XXL in 8-bit to save memory
        _messages.debug_log("Loading T5-XXL model with 8-bit quantization and auto device mapping")

        with _suppress_accelerate_warnings():
            config = transformers.T5Config.from_pretrained(
                model_id, local_files_only=local_files_only,
                quantization_config=quantization_config
            )

            config.torch_dtype = dtype

            text_encoder = model_class(config)

        blob_link = _pipelinewrapper_util.HFBlobLink.parse(model_path)
        
        if blob_link is not None:
            model_path = huggingface_hub.hf_hub_download(
                repo_id=blob_link.repo_id,
                revision=blob_link.revision,
                subfolder=blob_link.subfolder,
                filename=blob_link.weight_name,
                local_files_only=local_files_only
            )
        
        # Load state dict and update weights
        _messages.log(
            f'Loading monolithic t5-xxl state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        with _suppress_accelerate_warnings():
            text_encoder = accelerate.load_checkpoint_and_dispatch(
                text_encoder,
                checkpoint=model_path,
                device_map="auto",
                dtype=dtype,
                no_split_module_classes=["T5Block"]
            )

        return text_encoder
    
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(f"Failed to load T5-XXL model: {e}") from e


def _load_clip_g_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        local_files_only: bool = False,
        quantization_config=None
):
    model_id = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    try:
        with _suppress_accelerate_warnings():
            config = transformers.CLIPTextConfig.from_pretrained(
                model_id, local_files_only=local_files_only,
                quantization_config=quantization_config
            )

            config.torch_dtype = dtype

            # Load with automatic device mapping for efficient memory usage
            text_encoder = model_class(config)
        
        blob_link = _pipelinewrapper_util.HFBlobLink.parse(model_path)
        
        if blob_link is not None:
            model_path = huggingface_hub.hf_hub_download(
                repo_id=blob_link.repo_id,
                revision=blob_link.revision,
                subfolder=blob_link.subfolder,
                filename=blob_link.weight_name,
                local_files_only=local_files_only
            )

        _messages.log(
            f'Loading monolithic clip-g state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = accelerate.load_checkpoint_and_dispatch(
                text_encoder,
                checkpoint=model_path,
                device_map="auto",
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        return text_encoder
    
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(f"Failed to load CLIP-G model: {e}") from e


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
    def quantizer(self) -> _types.OptionalUri:
        """
        --quantizer URI override
        """
        return self._quantizer

    @property
    def mode(self) -> _types.OptionalString:
        """
        Model loading mode for single file checkpoints, for example 'clip-l', 'clip-g', or 't5-xxl'

        The default behavior is to extract the sub model from an 
        assumed to be combined checkpoint, which is not compatible
        with quantization.
        """
        return self._mode

    _encoders = {
        'CLIPTextModel': transformers.models.clip.CLIPTextModel,
        'CLIPTextModelWithProjection': transformers.models.clip.CLIPTextModelWithProjection,
        'T5EncoderModel': transformers.models.t5.T5EncoderModel,
        'DistillT5EncoderModel': DistillT5EncoderModel,
        'ChatGLMModel': diffusers.pipelines.kolors.ChatGLMModel
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
                 quantizer: _types.OptionalUri = None,
                 mode: _types.OptionalString = None):
        """
        :param encoder: encoder class name, for example ``CLIPTextModel``
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param mode: model loading mode, for example ``clip-l`` for single file ''clip-l'' checkpoints.

        :raises InvalidTextEncoderUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

        if encoder not in self._encoders:
            raise _exceptions.InvalidTextEncoderUriError(
                f'Unknown TextEncoder encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        mode = mode.lower() if mode is not None else mode

        if mode not in (None, 'clip-l', 't5-xxl', 'clip-g'):
            raise _exceptions.InvalidTextEncoderUriError(
                f'Unknown TextEncoder load mode "{mode}", must be None, "clip-l", "clip-g", or "t5-xxl"')

        if _pipelinewrapper_util.is_single_file_model_load(model):
            if quantizer and mode not in ('clip-l', 't5-xxl', 'clip-g'):
                raise _exceptions.InvalidTextEncoderUriError(
                    'specifying a Text Encoder quantizer URI is only supported for Hugging Face '
                    'repository loads from a repo slug or disk path, single file loads are not supported.')
                
        # Validate special modes don't use variant or revision
        if mode in ('clip-l', 't5-xxl', 'clip-g'):
            if variant is not None:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'TextEncoder cannot use variant with mode "{mode}", these are incompatible options')
            if revision is not None:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'TextEncoder cannot use revision with mode "{mode}", these are incompatible options')
            if subfolder is not None:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'TextEncoder cannot use subfolder with mode "{mode}", these are incompatible options')

        self._encoder = encoder
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer
        self._mode = mode

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidTextEncoderUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             original_config: _types.OptionalPath = None,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             missing_ok: bool = False
             ) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel,
                DistillT5EncoderModel,
                diffusers.pipelines.kolors.ChatGLMModel, None]:
        """
        Load a torch Text Encoder of type :py:class:`transformers.models.clip.CLIPTextModel`,
        :py:class:`transformers.models.clip.CLIPTextModelWithProjection`,
        :py:class:`transformers.models.t5.T5EncoderModel`, or
        :py:class:`diffusers.pipelines.kolors.ChatGLMModel` from this URI

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link
        :param no_cache: If ``True``, force the returned object not to be cached by the memoize decorator.
        :param missing_ok: If ``True``, when a VAE is not found inside a single file checkpoint as a sub model,
            just return ``None`` instead of throwing an error.

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`transformers.models.clip.CLIPTextModel`,
            :py:class:`transformers.models.clip.CLIPTextModelWithProjection`,
            :py:class:`transformers.models.t5.T5EncoderModel`, or
            :py:class:`diffusers.pipelines.kolors.ChatGLMModel`
        """
        try:
            args = locals()
            args.pop('self')
            return self._load(**args)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _pipelinewrapper_util.ModelNotFoundError(e) from e
        except Exception as e:
            raise _exceptions.TextEncoderUriLoadError(
                f'error loading text encoder "{self.model}": {e}') from e

    @staticmethod
    def _enforce_cache_size(new_text_encoder_size):
        _text_encoder_cache.enforce_cpu_mem_constraints(
            _constants.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
            size_var='text_encoder_size',
            new_object_size=new_text_encoder_size)

    @_memoize(_text_encoder_cache,
              exceptions={'local_files_only', 'missing_ok'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch TextEncoder", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch TextEncoder", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              original_config: _types.OptionalPath = None,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              missing_ok: bool = False
              ) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel,
                DistillT5EncoderModel,
                diffusers.pipelines.kolors.ChatGLMModel, None]:

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        if self.variant is None:
            variant = variant_fallback
        else:
            variant = self.variant

        if self.encoder == 'ChatGLMModel':
            encoder_library = 'diffusers.pipelines.kolors'
        else:
            encoder_library = 'transformers'

        encoder = self._encoders[self.encoder]

        if self.mode == 'clip-l' and encoder not in (transformers.CLIPTextModel, transformers.CLIPTextModelWithProjection):
            raise _exceptions.TextEncoderUriLoadError(
                f'Encoder "{self.encoder}" does not support loading with mode "clip-l".')

        if self.mode == 'clip-g' and encoder not in (transformers.CLIPTextModel, transformers.CLIPTextModelWithProjection):
            raise _exceptions.TextEncoderUriLoadError(
                f'Encoder "{self.encoder}" does not support loading with mode "clip-g".')

        if self.mode == 't5-xxl' and encoder not in (transformers.T5EncoderModel, DistillT5EncoderModel):
            raise _exceptions.TextEncoderUriLoadError(
                f'Encoder "{self.encoder}" does not support loading with mode "t5-xxl".')

        model_path = _pipelinewrapper_util.download_non_hf_model(self.model)

        if self.quantizer:
            quant_config = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidTextEncoderUriError
            ).parse(self.quantizer).to_config(torch_dtype)
        else:
            quant_config = None

        if _pipelinewrapper_util.is_single_file_model_load(model_path):
            if self.mode in ('clip-l', 't5-xxl', 'clip-g'):
                # Ensure these modes are only used with safetensors files
                if not model_path.endswith('.safetensors'):
                    raise _exceptions.TextEncoderUriLoadError(
                        f'TextEncoder mode "{self.mode}" only supports loading '
                        f'from safetensors files, but got: "{model_path}"')
                    
            if self.mode == 'clip-l':
                estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                    repo_id="openai/clip-vit-large-patch14",
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token
                )

                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_l_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    local_files_only=local_files_only,
                    quantization_config=quant_config
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-g':
                estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                    repo_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token
                )

                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_g_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    local_files_only=local_files_only,
                    quantization_config=quant_config
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 't5-xxl':
                estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                    repo_id="google/t5-v1_1-xxl",
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token
                )

                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_t5_xxl_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    local_files_only=local_files_only,
                    quantization_config=quant_config
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            else:
                try:
                    original_config = _pipelinewrapper_util.download_non_hf_config(
                        original_config) if original_config else None
                except _pipelinewrapper_util.NonHFConfigDownloadError as e:
                    raise _exceptions.TextEncoderUriLoadError(
                        f'original config file "{original_config}" for Text Encoder could not be downloaded: {e}'
                    ) from e

                estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                    repo_id=model_path,
                    revision=self.revision,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token
                )

                self._enforce_cache_size(estimated_memory_use)

                try:
                    text_encoder = _pipelinewrapper_util.single_file_load_sub_module(
                        path=model_path,
                        class_name=self.encoder,
                        library_name=encoder_library,
                        name=self.subfolder if self.subfolder else 'text_encoder',
                        use_auth_token=use_auth_token,
                        original_config=original_config,
                        local_files_only=local_files_only,
                        revision=self.revision,
                        dtype=torch_dtype
                    )
                except FileNotFoundError as e:
                    # cannot find configs
                    raise _pipelinewrapper_util.ModelNotFoundError(e) from e
                except diffusers.loaders.single_file.SingleFileComponentError as e:
                    if missing_ok:
                        # noinspection PyTypeChecker
                        return None, _d_memoize.CachedObjectMetadata(
                            size=0,
                            skip=True
                        )
                    raise _exceptions.TextEncoderUriLoadError(
                        f'Failed to load Text Encoder from single file checkpoint {model_path}, '
                        f'make sure the file contains a Text Encoders.') from e

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)

        else:
            if original_config:
                raise _exceptions.TextEncoderUriLoadError(
                    'specifying original_config file for Text Encoder '
                    'is only supported for single file loads.')

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

            text_encoder = encoder.from_pretrained(
                model_path,
                revision=self.revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=use_auth_token,
                local_files_only=local_files_only,
                quantization_config=quant_config
            )

        _messages.debug_log('Estimated Torch TextEncoder Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _util._patch_module_to_for_sized_cache(_text_encoder_cache, text_encoder)

        # noinspection PyTypeChecker
        return text_encoder, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_use,
            skip=self.quantizer or no_cache
        )

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
                                  quantizer=r.args.get('quantizer', False),
                                  mode=r.args.get('mode', None)
                                  )
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTextEncoderUriError(e) from e
