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

import os.path
import typing

import diffusers

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_textual_inversion_uri_parser = _textprocessing.ConceptUriParser(
    'Textual Inversion', ['token', 'revision', 'subfolder', 'weight-name'])


def _load_textual_inversion_state_dict(pretrained_model_name_or_path, **kwargs):
    from diffusers.utils.hub_utils import _get_model_file
    from diffusers.models.modeling_utils import load_state_dict

    text_inversion_name = "learned_embeds.bin"
    text_inversion_name_safe = "learned_embeds.safetensors"

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "text_inversion",
        "framework": "pytorch",
    }

    # 3.1. Load textual inversion file
    state_dict = None
    model_file = None

    # Let's first try to load .safetensors weights
    if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
    ):
        try:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=weight_name or text_inversion_name_safe,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = load_state_dict(model_file)
        except Exception as e:
            if not allow_pickle:
                raise e

            model_file = None

    if model_file is None:
        model_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=weight_name or text_inversion_name,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        state_dict = load_state_dict(model_file)
    return model_file, state_dict


class TextualInversionUri:
    """
    Representation of ``--textual-inversions`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['Textual Inversion']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--textual-inversions')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path
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
    def weight_name(self) -> _types.OptionalName:
        """
        Model weight-name
        """
        return self._weight_name

    @property
    def token(self) -> _types.OptionalString:
        """
        Prompt keyword
        """
        return self._token

    def __init__(self,
                 model: str,
                 token: str | None = None,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None):
        self._token = token
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                         uris: typing.Iterable[typing.Union["TextualInversionUri", str]],
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load Textual Inversion weights on to a pipeline using on or more URIs

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param uris: Iterable of :py:class:`TextualInversionUri` or ``str``
            Textual Inversion URIs to load
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        :raises dgenerate.pipelinewrapper.uris.exceptions.InvalidTextualInversionUriError: On URI parsing errors.
        :raises dgenerate.pipelinewrapper.uris.exceptions.TextualInversionUriLoadError: On loading errors.
        """
        def cache_all(e):
            if isinstance(e, _exceptions.InvalidTextualInversionUriError):
                raise e
            else:
                raise _exceptions.TextualInversionUriLoadError(
                    f'error loading Textual Inversions: {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            TextualInversionUri._load_on_pipeline(
                uris=uris,
                pipeline=pipeline,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only)

    @staticmethod
    def _load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                          uris: typing.Iterable[typing.Union["TextualInversionUri", str]],
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_textual_inversion'):

            for textual_inversion_uri in uris:
                if not isinstance(textual_inversion_uri, TextualInversionUri):
                    textual_inversion_uri = TextualInversionUri.parse(textual_inversion_uri)

                model_path = _hfhub.download_non_hf_slug_model(textual_inversion_uri.model)

                is_sdxl = pipeline.__class__.__name__.startswith('StableDiffusionXL')
                is_flux = pipeline.__class__.__name__.startswith('Flux')

                if is_sdxl or is_flux:
                    filename, dicts = _load_textual_inversion_state_dict(
                        model_path,
                        revision=textual_inversion_uri.revision,
                        subfolder=textual_inversion_uri.subfolder,
                        weight_name=textual_inversion_uri.weight_name,
                        local_files_only=local_files_only,
                        token=use_auth_token
                    )

                    if is_sdxl:
                        if 'clip_l' not in dicts or 'clip_g' not in dicts:
                            raise RuntimeError(
                                'clip_l or clip_g not found in SDXL textual '
                                f'inversion model "{textual_inversion_uri.model}" state dict, '
                                'unsupported model format.')
                    else:
                        if 'clip_l' not in dicts:
                            raise RuntimeError(
                                'clip_l not found in Flux textual '
                                f'inversion model "{textual_inversion_uri.model}" state dict, '
                                'unsupported model format.')

                    # token is the file name (no extension) with spaces
                    # replaced by underscores when the user does not provide
                    # a prompt token
                    token = os.path.splitext(
                        os.path.basename(filename))[0].replace(' ', '_') \
                        if textual_inversion_uri.token is None else textual_inversion_uri.token

                    pipeline.load_textual_inversion(dicts['clip_l'],
                                                    token=token,
                                                    text_encoder=pipeline.text_encoder,
                                                    tokenizer=pipeline.tokenizer,
                                                    hf_token=use_auth_token)

                    if is_sdxl:
                        pipeline.load_textual_inversion(dicts['clip_g'],
                                                        token=token,
                                                        text_encoder=pipeline.text_encoder_2,
                                                        tokenizer=pipeline.tokenizer_2,
                                                        hf_token=use_auth_token)

                    if is_flux and 't5' in dicts:
                        pipeline.load_textual_inversion(dicts['t5'],
                                                        token=token,
                                                        text_encoder=pipeline.text_encoder_2,
                                                        tokenizer=pipeline.tokenizer_2,
                                                        hf_token=use_auth_token)
                else:
                    pipeline.load_textual_inversion(model_path,
                                                    token=textual_inversion_uri.token,
                                                    revision=textual_inversion_uri.revision,
                                                    subfolder=textual_inversion_uri.subfolder,
                                                    weight_name=textual_inversion_uri.weight_name,
                                                    local_files_only=local_files_only,
                                                    hf_token=use_auth_token)

                _messages.debug_log(f'Added Textual Inversion: "{textual_inversion_uri}" '
                                    f'to pipeline: "{pipeline.__class__.__name__}"')
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading textual inversions.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'TextualInversionUri':
        """
        Parse a ``--textual-inversions`` uri and return an object representing its constituents

        :param uri: string with ``--textual-inversions`` uri syntax

        :raise InvalidTextualInversionUriError:

        :return: :py:class:`.TextualInversionPath`
        """
        try:
            r = _textual_inversion_uri_parser.parse(uri)

            return TextualInversionUri(model=r.concept,
                                       token=r.args.get('token', None),
                                       weight_name=r.args.get('weight-name', None),
                                       revision=r.args.get('revision', None),
                                       subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTextualInversionUriError(e) from e
