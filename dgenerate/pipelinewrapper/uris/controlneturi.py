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
import enum

import diffusers

import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper import constants as _constants
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris import util as _util

_controlnet_uri_parser = _textprocessing.ConceptUriParser(
    'ControlNet', ['scale', 'start', 'end', 'mode', 'revision', 'variant', 'subfolder', 'dtype', 'quantizer'])

_controlnet_cache = _d_memoize.create_object_cache(
    'controlnet', cache_type=_memory.SizedConstrainedObjectCache
)


class FluxControlNetUnionUriModes(enum.IntEnum):
    """
    Represents controlnet modes associated with the Flux Union controlnet.
    """
    CANNY = 0
    TILE = 1
    DEPTH = 2
    BLUR = 3
    POSE = 4
    GRAY = 5
    LQ = 6


class SDXLControlNetUnionUriModes(enum.IntEnum):
    """
    Represents controlnet modes associated with the SDXL Union controlnet.
    """
    OPENPOSE = 0
    DEPTH = 1
    HED = 2
    PIDI = 2
    SCRIBBLE = 2
    TED = 2
    CANNY = 3
    LINEART = 3
    ANIME_LINEART = 3
    MLSD = 3
    NORMAL = 4
    SEGMENT = 5


class ControlNetUri:
    """
    Representation of ``--control-nets`` URI.
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['Control Net']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--control-nets')

    # Arguments that should be hidden from schema
    # generation, because they are not parsed from the URI
    HIDE_ARGS = {'model-type'}

    OPTION_ARGS = {
        'dtype': ['float16', 'bfloat16', 'float32']
    }

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

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
    def scale(self) -> float:
        """
        ControlNet guidance scale
        """
        return self._scale

    @property
    def start(self) -> float:
        """
        ControlNet guidance start point, fraction of inference / timesteps.
        """
        return self._start

    @property
    def end(self) -> float:
        """
        ControlNet guidance end point, fraction of inference / timesteps.
        """
        return self._end

    @property
    def mode(self) -> int | None:
        """
        Union ControlNet mode.
        """
        return self._mode

    @property
    def model_type(self) -> _enums.ModelType:
        """
        Model type the ControlNet model is expected to attach to.
        """
        return self._model_type

    @property
    def quantizer(self) -> _types.OptionalUri:
        """
        --quantizer URI override
        """
        return self._quantizer

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString,
                 variant: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: _enums.DataType | str | None = None,
                 scale: float = 1.0,
                 start: float = 0.0,
                 end: float = 1.0,
                 mode: int | str | FluxControlNetUnionUriModes | SDXLControlNetUnionUriModes | None = None,
                 quantizer: _types.OptionalUri = None,
                 model_type: _enums.ModelType = _enums.ModelType.SD):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param scale: controlnet scale
        :param start: controlnet guidance start value
        :param end: controlnet guidance end value
        :param mode: Flux / SDXL Union controlnet mode.
        :param quantizer: --quantizer URI override
        :param model_type: Model type this ControlNet will be attached to.

        :raises InvalidControlNetUriError: If ``dtype`` is passed an invalid data type string,
            or if ``model`` points to a single file and ``quantizer`` is specified (not supported).
        """

        if _hfhub.is_single_file_model_load(model):
            if quantizer:
                raise _exceptions.InvalidControlNetUriError(
                    'specifying a ControlNet quantizer URI is only supported for Hugging Face '
                    'repository loads from a repo slug or disk path, single file loads are not supported.')

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer
        self._model_type = model_type

        if isinstance(mode, str):
            if _enums.model_type_is_sdxl(model_type):
                self._mode = ControlNetUri._sdxl_mode_int_from_str(mode)
            elif _enums.model_type_is_flux(model_type):
                self._mode = ControlNetUri._flux_mode_int_from_str(mode)
            else:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "mode" argument not supported '
                    f'for model type: {_enums.get_model_type_string(model_type)}.'
                )
        else:
            self._mode = int(mode) if mode is not None else None

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidControlNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._scale = scale
        self._start = start
        self._end = end

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             device_map: str | None = None,
             model_class:
             type[diffusers.ControlNetModel] |
             type[diffusers.ControlNetUnionModel] |
             type[diffusers.SD3ControlNetModel] |
             type[diffusers.FluxControlNetModel] | None = None) -> \
            diffusers.ControlNetModel | \
            diffusers.ControlNetUnionModel | \
            diffusers.SD3ControlNetModel | \
            diffusers.FluxControlNetModel:
        """
        Load a :py:class:`diffusers.ControlNetModel` from this URI.


        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :param no_cache: If True, force the returned object not to be cached by the memoize decorator.

        :param device_map: device placement strategy for quantized models, defaults to ``None``

        :param model_class: What class of controlnet model should be loaded?
            if ``None`` is specified, load based off :py:attr:`ControlNetUri.model_type`
            and provided URI arguments.
        
        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.ControlNetModel`, :py:class:`diffusers.SD3ControlNetModel`, or :py:class:`diffusers.FluxControlNetModel`
        """
        def cache_all(e):
            raise _exceptions.ControlNetUriLoadError(
                f'error loading controlnet "{self.model}": {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):

            if model_class is None:
                if _enums.model_type_is_flux(self.model_type):
                    model_class = diffusers.FluxControlNetModel
                elif _enums.model_type_is_sd3(self.model_type):
                    model_class = diffusers.SD3ControlNetModel
                elif _enums.model_type_is_sdxl(self.model_type) and self.mode is not None:
                    model_class = diffusers.ControlNetUnionModel
                else:
                    model_class = diffusers.ControlNetModel

            return self._load(dtype_fallback,
                              use_auth_token,
                              local_files_only,
                              no_cache,
                              device_map,
                              model_class)


    @staticmethod
    def _enforce_cache_size(new_controlnet_size):
        _controlnet_cache.enforce_cpu_mem_constraints(
            _constants.CONTROLNET_CACHE_MEMORY_CONSTRAINTS,
            size_var='controlnet_size',
            new_object_size=new_controlnet_size)

    @_memoize(_controlnet_cache,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(
                  args, {'self': lambda o: _d_memoize.property_hasher(
                      o, exclude={'scale', 'start', 'end'})}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch ControlNet", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch ControlNet", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              device_map: str | None = None,
              model_class:
              type[diffusers.ControlNetModel] |
              type[diffusers.ControlNetUnionModel] |
              type[diffusers.SD3ControlNetModel] |
              type[diffusers.FluxControlNetModel] = diffusers.ControlNetModel) -> \
            diffusers.ControlNetModel | \
            diffusers.ControlNetUnionModel | \
            diffusers.SD3ControlNetModel | \
            diffusers.FluxControlNetModel:

        if model_class not in {diffusers.FluxControlNetModel, diffusers.ControlNetUnionModel}:
            if self.mode is not None:
                raise ValueError(
                    f'The "mode" argument of ControlNet "{self.model}" is invalid to use '
                    'with non Flux / SDXL ControlNet Union models.'
                )

        model_path = _hfhub.download_non_hf_slug_model(self.model)

        single_file_load_path = _hfhub.is_single_file_model_load(model_path)

        torch_dtype = _enums.get_torch_dtype(
            dtype_fallback if self.dtype is None else self.dtype)

        if self.quantizer:
            quant_config = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidControlNetUriError
            ).parse(self.quantizer).to_config(torch_dtype)
        else:
            quant_config = None

        if single_file_load_path:

            estimated_memory_usage = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            self._enforce_cache_size(estimated_memory_usage)

            new_net = model_class.from_single_file(
                model_path,
                revision=self.revision,
                torch_dtype=torch_dtype,
                token=use_auth_token,
                local_files_only=local_files_only)
        else:

            estimated_memory_usage = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            self._enforce_cache_size(estimated_memory_usage)

            new_net = model_class.from_pretrained(
                model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                torch_dtype=torch_dtype,
                token=use_auth_token,
                local_files_only=local_files_only,
                quantization_config=quant_config,
                device_map=device_map)

        _messages.debug_log('Estimated Torch ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _util._patch_module_to_for_sized_cache(_controlnet_cache, new_net)

        # noinspection PyTypeChecker
        return new_net, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_usage,
            skip=self.quantizer or no_cache
        )

    @staticmethod
    def parse(uri: _types.Uri,
              model_type=_enums.ModelType.SD) -> 'ControlNetUri':
        """
        Parse a ``--control-nets`` uri specification and return an object representing its constituents

        :param uri: string with ``--control-nets`` uri syntax
        :param model_type: model type that the ControlNet will be attached to.

        :raise InvalidControlNetUriError:

        :return: :py:class:`.TorchControlNetUri`
        """
        try:
            r = _controlnet_uri_parser.parse(uri)

            dtype = r.args.get('dtype')
            scale = r.args.get('scale', 1.0)
            start = r.args.get('start', 0.0)
            end = r.args.get('end', 1.0)
            mode = r.args.get('mode', None)

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "scale" must be a floating point number, received: {scale}')

            try:
                start = float(start)
            except ValueError:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "start" must be a floating point number, received: {start}')

            try:
                end = float(end)
            except ValueError:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "end" must be a floating point number, received: {end}')

            if start > end:
                raise _exceptions.InvalidControlNetUriError(
                    f'Torch ControlNet "start" must be less than or equal to "end".')

            if mode is not None:
                if _enums.model_type_is_sdxl(model_type):
                    mode = ControlNetUri._sdxl_mode_int_from_str(mode)
                elif _enums.model_type_is_flux(model_type):
                    mode = ControlNetUri._flux_mode_int_from_str(mode)
                else:
                    raise _exceptions.InvalidControlNetUriError(
                        f'Torch ControlNet "mode" argument not supported '
                        f'for model type: {_enums.get_model_type_string(model_type)}.'
                    )

            return ControlNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                subfolder=r.args.get('subfolder', None),
                dtype=dtype,
                scale=scale,
                start=start,
                end=end,
                mode=mode,
                quantizer=r.args.get('quantizer', None),
                model_type=model_type
            )

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidControlNetUriError(e) from e

    @staticmethod
    def _sdxl_mode_int_from_str(mode):
        modes = _textprocessing.oxford_comma(
            [n.name.lower() for n in SDXLControlNetUnionUriModes], "or")
        try:
            try:
                mode = int(mode)
            except ValueError:
                mode = SDXLControlNetUnionUriModes[mode.upper()].value

        except KeyError:
            raise _exceptions.InvalidControlNetUriError(
                f'Torch SDXL Union ControlNet "mode" must be an integer, '
                f'or one of: {modes}. received: {mode}')
        if mode >= len(SDXLControlNetUnionUriModes) or mode < 0:
            raise _exceptions.InvalidControlNetUriError(
                f'Torch SDXL Union ControlNet "mode" must be less than '
                f'{len(SDXLControlNetUnionUriModes)} and greater than zero, '
                f'mode number {mode} does not exist.')
        return mode

    @staticmethod
    def _flux_mode_int_from_str(mode):
        modes = _textprocessing.oxford_comma(
            [n.name.lower() for n in FluxControlNetUnionUriModes], "or")
        try:
            try:
                mode = int(mode)
            except ValueError:
                mode = FluxControlNetUnionUriModes[mode.upper()].value

        except KeyError:
            raise _exceptions.InvalidControlNetUriError(
                f'Torch Flux Union ControlNet "mode" must be an integer, '
                f'or one of: {modes}. received: {mode}')
        if mode >= len(FluxControlNetUnionUriModes) or mode < 0:
            raise _exceptions.InvalidControlNetUriError(
                f'Torch Flux Union ControlNet "mode" must be less than '
                f'{len(FluxControlNetUnionUriModes)} and greater than zero, '
                f'mode number {mode} does not exist.')
        return mode
