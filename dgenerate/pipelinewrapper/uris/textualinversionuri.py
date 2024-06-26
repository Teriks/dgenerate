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

import diffusers
import huggingface_hub
import safetensors
import safetensors.torch
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_textual_inversion_uri_parser = _textprocessing.ConceptUriParser('Textual Inversion',
                                                                 ['token', 'revision', 'subfolder', 'weight-name'])


def _load_textual_inversion_state_dict(pretrained_model_name_or_path, **kwargs):
    from diffusers.utils.hub_utils import _get_model_file

    text_inversion_name = "learned_embeds.bin"
    text_inversion_name_safe = "learned_embeds.safetensors"

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
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
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = safetensors.torch.load_file(model_file, device="cpu")
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
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        state_dict = torch.load(model_file, map_location="cpu")
    return model_file, state_dict


class TextualInversionUri:
    """
    Representation of ``--textual-inversions`` uri
    """

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

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load Textual Inversion weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        """
        try:
            self._load_on_pipeline(pipeline=pipeline,
                                   use_auth_token=use_auth_token,
                                   local_files_only=local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.TextualInversionUriLoadError(
                f'error loading textual inversion "{self.model}": {e}')

    def _load_on_pipeline(self,
                          pipeline: diffusers.DiffusionPipeline,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_textual_inversion'):
            debug_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}

            _messages.debug_log('pipeline.load_textual_inversion(' +
                                str(_types.get_public_attributes(self) | debug_args) + ')')

            model_path = _hfutil.download_non_hf_model(self.model)

            # this is tricky because there is stupidly a positional argument named 'token'
            # as well as an accepted kwargs value with the key 'token'

            old_token = os.environ.get('HF_TOKEN', None)
            if use_auth_token is not None:
                os.environ['HF_TOKEN'] = use_auth_token

            try:
                if pipeline.__class__.__name__.startswith('StableDiffusionXL'):
                    filename, dicts = _load_textual_inversion_state_dict(
                        model_path,
                        revision=self.revision,
                        subfolder=self.subfolder,
                        weight_name=self.weight_name,
                        local_files_only=local_files_only
                    )

                    if 'clip_l' not in dicts or 'clip_g' not in dicts:
                        raise RuntimeError(
                            'clip_l or clip_g not found in SDXL textual '
                            f'inversion model "{self.model}" state dict, '
                            'unsupported model format.')

                    # token is the file name (no extension) with spaces
                    # replaced by underscores when the user does not provide
                    # a prompt token
                    token = os.path.splitext(
                        os.path.basename(filename))[0].replace(' ', '_') \
                        if self.token is None else self.token

                    pipeline.load_textual_inversion(dicts['clip_l'],
                                                    token=token,
                                                    text_encoder=pipeline.text_encoder,
                                                    tokenizer=pipeline.tokenizer)

                    pipeline.load_textual_inversion(dicts['clip_g'],
                                                    token=token,
                                                    text_encoder=pipeline.text_encoder_2,
                                                    tokenizer=pipeline.tokenizer_2)
                else:
                    pipeline.load_textual_inversion(model_path,
                                                    token=self.token,
                                                    revision=self.revision,
                                                    subfolder=self.subfolder,
                                                    weight_name=self.weight_name,
                                                    local_files_only=local_files_only)
            finally:
                if old_token is not None:
                    os.environ['HF_TOKEN'] = old_token

            _messages.debug_log(f'Added Textual Inversion: "{self}" to pipeline: "{pipeline.__class__.__name__}"')
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
            raise _exceptions.InvalidTextualInversionUriError(e)
