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
import collections.abc
import glob
import json
import os
import pathlib
import re
from contextlib import contextmanager

import diffusers.loaders.single_file as _single_file
import diffusers.pipelines
import huggingface_hub
import huggingface_hub.errors
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.uris
import dgenerate.types as _types
import dgenerate.webcache as _webcache


class InvalidDeviceOrdinalException(Exception):
    """
    GPU in device specification (cuda:N) does not exist
    """
    pass


def default_device() -> str:
    """
    Return a string representing the systems default accelerator device.

    Possible Values:

        * ``"cuda"``
        * ``"mps"``
        * ``"cpu"``

    :return: ``"cuda"``, ``"mps"``, etc.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def is_valid_device_string(device, raise_ordinal=True):
    """
    Is a device string valid? including the device ordinal specified?

    Other than cuda, "mps" (MacOS metal performance shaders) is experimentally supported.

    :param device: device string, such as ``cpu``, or ``cuda``, or ``cuda:N``
    :param raise_ordinal: Raise :py:exc:`.InvalidDeviceOrdinalException` if
        a specified CUDA device ordinal is found to not exist?

    :raises InvalidDeviceOrdinalException: If ``raise_ordinal=True`` and a the
        device ordinal specified in a CUDA device string does not exist.

    :return: ``True`` or ``False``
    """

    match = re.match(r'^(?:cpu|cuda(?::([0-9]+))?)$', device)

    if match:
        if match.lastindex:
            ordinal = int(match[1])
            valid_ordinal = ordinal < torch.cuda.device_count()
            if raise_ordinal and not valid_ordinal:
                raise InvalidDeviceOrdinalException(f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
            return valid_ordinal
        return True

    if device == 'mps' and hasattr(torch.backends, 'mps') \
            and torch.backends.mps.is_available():
        return True

    return False


@contextmanager
def _disable_tqdm():
    huggingface_hub.utils.enable_progress_bars()
    try:
        yield
    finally:
        huggingface_hub.utils.enable_progress_bars()


def fetch_model_index_dict(
        path: str,
        revision=None,
        use_auth_token=None,
        local_files_only=False
) -> dict:
    """
    Fetch ``model_index.json` as a dict for a given model path, this may be a Hugging Face repo
    or a path to a single combined checkpoint file.

    :param path: Hugging Face repo, or directory, or file path
    :param revision: repo revision
    :param use_auth_token: Use this HF auth token?
    :param local_files_only: Only look through the cache?

    :return: config dict
    """
    if is_single_file_model_load(path):
        checkpoint = _single_file.load_single_file_checkpoint(
            path,
            token=use_auth_token,
            local_files_only=local_files_only,
            revision=revision,
        )

        config = _single_file.fetch_diffusers_config(checkpoint)
        default_pretrained_model_config_name = config["pretrained_model_name_or_path"]
    else:
        default_pretrained_model_config_name = path

    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]

    with _disable_tqdm():
        # the mischief I am doing here does not really integrate well with
        # the diffusers tqdm bars for single file loads, so disable them
        # for this helper function

        try:
            cached_model_config_path = huggingface_hub.snapshot_download(
                default_pretrained_model_config_name,
                revision=revision,
                local_files_only=True,
                token=use_auth_token,
                allow_patterns=allow_patterns
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            if local_files_only:
                raise FileNotFoundError(
                    f'offline mode is active, but a config file is needed from Hugging Face '
                    f'hub to utilize: {path}')
            else:
                try:
                    cached_model_config_path = huggingface_hub.snapshot_download(
                        default_pretrained_model_config_name,
                        revision=revision,
                        local_files_only=False,
                        token=use_auth_token,
                        allow_patterns=allow_patterns
                    )
                except huggingface_hub.errors.LocalEntryNotFoundError:
                    raise FileNotFoundError(
                        f'could not find the config file on Hugging Face hub '
                        f'for: {path}')
    with open(os.path.join(cached_model_config_path, 'model_index.json')) as config_file:
        return json.load(config_file)


def single_file_load_sub_module(
        path: str,
        class_name: str,
        library_name: str,
        name: str,
        use_auth_token: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        dtype: torch.dtype | None = None,
        config: _types.OptionalPath = None,
        original_config: _types.OptionalPath = None
) -> torch.nn.Module:
    """
    Load a submodule (vae, unet, textencoder, etc..) by name out of an in one checkpoint file.

    :param path: checkpoint file path
    :param class_name: submodule class name
    :param library_name: python module where the class exists
    :param name: submodule name, i.e. vae, unet, text_encoder, text_encoder_2, etc.
    :param use_auth_token: Hugging Face auth token for downloading configs?
    :param local_files_only: Do not attempt to download files, only use cache?
    :param revision: Repo revision for detected config repo
    :param dtype: torch dtype for the module
    :param config: Path to model configuration, Hugging Face repo, or Hugging Face blob link, or `.json` file on disk.
    :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.

    :return: The module.
    """

    checkpoint = _single_file.load_single_file_checkpoint(
        path,
        token=use_auth_token,
        local_files_only=local_files_only,
        revision=revision,
    )

    if config is None:
        config = _single_file.fetch_diffusers_config(checkpoint)

        default_pretrained_model_config_name = config["pretrained_model_name_or_path"]
    else:
        default_pretrained_model_config_name = config

    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]
    with _disable_tqdm():
        # the mischief I am doing here does not really integrate well with
        # the diffusers tqdm bars for single file loads, so disable them
        # for this helper function

        try:
            cached_model_config_path = huggingface_hub.snapshot_download(
                default_pretrained_model_config_name,
                revision=revision,
                local_files_only=True,
                token=use_auth_token,
                allow_patterns=allow_patterns
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            if local_files_only:
                raise FileNotFoundError(
                    f'offline mode is active, but a config file is needed from Hugging Face '
                    f'hub to utilize the {class_name} in: {path}')
            else:
                try:
                    cached_model_config_path = huggingface_hub.snapshot_download(
                        default_pretrained_model_config_name,
                        revision=revision,
                        local_files_only=False,
                        token=use_auth_token,
                        allow_patterns=allow_patterns
                    )
                except huggingface_hub.errors.LocalEntryNotFoundError:
                    raise FileNotFoundError(
                        f'could not find the config file on Hugging Face hub '
                        f'for {class_name} in: {path}')

    model = _single_file.load_single_file_sub_model(
        library_name=library_name,
        checkpoint=checkpoint,
        class_name=class_name,
        name=name,
        pipelines=diffusers.pipelines,
        cached_model_config_path=cached_model_config_path,
        is_pipeline_module=False,
        torch_dtype=dtype,
        original_config=original_config,
        local_files_only=local_files_only)

    return model


def get_quantizer_uri_class(uri, exception=ValueError):
    """
    Get the URI parser class needed for a particular quantizer URI
    :param uri: The URI
    :param exception: Exception type to raise on unsupported quantization backend.
    :return: Class from :py:mod:`dgenerate.pipelinewrapper.uris`
    """
    concept = uri.split(';')[0].strip()
    if concept in {'bnb', 'bitsandbytes'}:
        if not diffusers.utils.is_bitsandbytes_available():
            raise exception(
                f'Cannot load quantization backend bitsandbytes, '
                f'as bitsandbytes is not installed.')
        return dgenerate.pipelinewrapper.uris.BNBQuantizerUri
    elif concept == 'torchao':
        if not diffusers.utils.is_torchao_available():
            raise exception(
                f'Cannot load quantization backend torchao, '
                f'as torchao is not installed.')
        return dgenerate.pipelinewrapper.uris.TorchAOQuantizerUri
    else:
        raise exception(f'Unknown quantization backend: {concept}')


def estimate_memory_usage(module: torch.nn.Module) -> str:
    """
    Estimate the static memory use of a torch module.

    :param module: the module

    :return: static memory use in bytes
    """

    dtype = next(module.parameters()).dtype
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }
    bytes_per_param = dtype_sizes.get(dtype, 4)
    num_params = sum(p.numel() for p in module.parameters())
    return num_params * bytes_per_param


class ModelNotFoundError(Exception):
    """Raised when a specified model can not be located either locally or remotely"""
    pass


class NonHFModelDownloadError(Exception):
    pass


class NonHFConfigDownloadError(Exception):
    pass


class HFBlobLink:
    """
    Represents the constituents of a huggingface blob link.
    """

    repo_id: str
    revision: str
    subfolder: str
    weight_name: str

    def __init__(self,
                 repo_id: str,
                 revision: str,
                 subfolder: str,
                 weight_name: str):
        self.repo_id = repo_id
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def __str__(self):
        return str(_types.get_public_attributes(self))

    def __repr__(self):
        return str(self)

    __REGEX = re.compile(
        r'(https|http)://(?:www\.)?huggingface\.co/'
        r'(?P<repo_id>.+)/blob/(?P<revision>.+?)/'
        r'(?:(?P<subfolder>.+)/)?(?P<weight_name>.+)')

    @staticmethod
    def parse(blob_link):
        """
        Attempt to parse a huggingface blob link out of a string.

        If the string does not contain a blob link, return None.

        :param blob_link: supposed blob link string
        :return: :py:class:`.HFBlobLink` or None
        """

        match = HFBlobLink.__REGEX.match(blob_link)

        if match:
            result = HFBlobLink(match.group('repo_id'),
                                match.group('revision'),
                                match.group('subfolder'),
                                match.group('weight_name'))

            _messages.debug_log(
                f'Parsed huggingface Blob Link: {blob_link} -> {result}')
            return result

        return None


def variant_match(filename: str, variant: str | None = None):
    """
    Match a model filename against a huggingface variant specifier such as "fp16"

    If no variant is specified and the filename contains a variant, this function will
    return ``False``.

    Examples:

        * ``variant_match('file.fp16-0001-of-0002.bin', variant=None) -> False``
        * ``variant_match('file.bin', variant=None) -> True``

    If a variant is specified and the filename contains a matching
    variant specifier, this function will return ``True``.

    Examples:

        * ``variant_match('file.fp16.bin', variant='fp16') -> True``
        * ``variant_match('file.8bit.bin', variant='fp16') -> False``

    If the file has shard information such as in the filename ``file.fp16-0001-of-0002.bin``,
    variant will be matched against the variant part (in this example filename, variant 'fp16'),
    and considered a match if **variant** is 'fp16'.

    Examples:

        * ``variant_match('file.fp16-0001-of-0002.bin', variant='fp16') -> True``
        * ``variant_match('file.8bit-0001-of-0002.bin', variant='fp16') -> False``

    :param filename: The filename
    :param variant: the variant string
    :return: ``True`` or ``False``
    """
    base, ext = os.path.splitext(filename)
    _, file_variant = os.path.splitext(base)

    if not variant:
        if file_variant:
            return False
        else:
            return True

    pattern = r'\.' + variant + r'(-[0-9]+-of-[0-9]+)?'
    return re.match(pattern, file_variant) is not None


def _hf_try_to_load_from_cache(repo_id: str,
                               filename: str,
                               cache_dir: str | pathlib.Path | None = None,
                               revision: str | None = None,
                               repo_type: str | None = None):
    try:
        return huggingface_hub.try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            repo_type=repo_type
        )
    except huggingface_hub.utils.HFValidationError as e:
        raise ModelNotFoundError(e)


def download_non_hf_model(model_path):
    """
    Check for a non hugging face link or reference to a model that is possibly downloadable as a single file.

    If this is applicable, download it to the web cache and return its path.
    If the file already exists in the web cache simply return a path to it.

    If this is not applicable, return the path unchanged.

    :param model_path: proposed model path
    :return: path to downloaded file or unchanged model path.
    """
    if _webcache.is_downloadable_url(model_path) and \
            HFBlobLink.parse(model_path) is None:
        _, model_path = _webcache.create_web_cache_file(
            model_path,
            mime_acceptable_desc='not text',
            mimetype_is_supported=lambda m: m is not None and not m.startswith('text/'),
            unknown_mimetype_exception=NonHFModelDownloadError)
    return model_path


def download_non_hf_config(path):
    """
    Check for a non hugging face link or reference to a config file
    that is possibly downloadable as a single file.

    If this is applicable, download it to the web cache and return its path.
    If the file already exists in the web cache simply return a path to it.

    If this is not applicable, return the path unchanged.

    :param path: proposed model path
    :return: path to downloaded file or unchanged model path.
    """
    if _webcache.is_downloadable_url(path) and \
            HFBlobLink.parse(path) is None:
        _, path = _webcache.create_web_cache_file(
            path,
            mime_acceptable_desc='text / yaml / json',
            mimetype_is_supported=lambda m:
            m.startswith('text/') or m.startswith('application/'),
            unknown_mimetype_exception=NonHFConfigDownloadError)
    return path


def fetch_model_files_with_size(repo_id: str,
                                revision: str | None = 'main',
                                variant: str | None = None,
                                subfolder: str | None = None,
                                weight_name: str | None = None,
                                use_auth_token: str | None = None,
                                extensions: collections.abc.Iterable | None = None,
                                local_files_only: bool = False,
                                sentencepiece: bool = False,
                                watermarker: bool = False) -> collections.abc.Iterator[tuple[str, int]]:
    """
    Attempt to fetch model files with their size that are relevant for the type of model being loaded.

    Either from huggingface disk cache or through the huggingface API if not on disk and ``local_files_only`` is ``False``.

    This function also works on blob links, paths to folders, or singular files on disk.



    :param repo_id: huggingface repo_id, or path to folder or file on disk
    :param revision: repo revision, IE: branch
    :param variant: files variant, IE: fp16
    :param subfolder: subfolder in the repo where the models exist
    :param weight_name: look for a specific model file name
    :param use_auth_token: optional huggingface auth token
    :param extensions: if specified, only search for extensions in this iterable
    :param local_files_only: utilize the huggingface API if necessary?
        if this is ``True``, and it is necessary to fetch info from the API, this function
        will simply yield nothing
    :param sentencepiece: Forcibly include tokenizer/spiece.model for models with a unet?
    :param watermarker: Forcibly include watermarker/diffusion_pytorch_model.bin for models with a unet?

    :return: an iterator over (filename, file size bytes)
    """

    __args_debug = locals()
    _messages.debug_log(
        f'{_types.fullname(fetch_model_files_with_size)}({__args_debug})')

    blob_link = HFBlobLink.parse(repo_id)

    if blob_link is not None:
        repo_id = blob_link.repo_id
        revision = blob_link.revision
        subfolder = blob_link.subfolder
        weight_name = blob_link.weight_name

    def post_discover_check(found):
        if not isinstance(found, str):
            return None
        if found:
            path_with_one_parent = os.path.join(
                pathlib.Path(found).parents[0].name, pathlib.Path(found).name)

            if sentencepiece:
                if path_with_one_parent == os.path.join('tokenizer', 'spiece.model'):
                    return found

            if watermarker:
                if path_with_one_parent == os.path.join('watermarker', 'diffusion_pytorch_model.bin'):
                    return found

            if weight_name and weight_name != os.path.basename(found):
                # Weight name specified but no match
                return None

            if not variant_match(filename=found,
                                 variant=variant):
                # Variant missmatch
                return None

            if extensions and os.path.splitext(found)[1] not in extensions:
                # Not a file extension we are interested in
                return None
        return found

    def enumerate_directory(d):
        files = glob.iglob(os.path.join(d,
                                        '**' if not subfolder else subfolder,
                                        '*' if not weight_name else weight_name),
                           recursive=True)
        for file in files:
            if os.path.isfile(file) and post_discover_check(file):
                yield os.path.join(subfolder if subfolder else '',
                                   os.path.relpath(file, d)), os.path.getsize(file)

    def enumerate_file(f):
        yield os.path.join(subfolder if subfolder else '',
                           os.path.basename(f)), os.path.getsize(f)

    def find_diffuser_weights(search_dir):
        found = _hf_try_to_load_from_cache(
            repo_id=repo_id,
            revision=revision,
            filename=os.path.join(search_dir,
                                  f'diffusion_pytorch_model{variant_part}safetensors')
        )

        if not isinstance(found, str):
            found = _hf_try_to_load_from_cache(
                repo_id=repo_id,
                revision=revision,
                filename=os.path.join(search_dir,
                                      f'diffusion_pytorch_model{variant_part}bin')
            )
        return post_discover_check(found)

    if os.path.isfile(repo_id):
        yield from enumerate_file(repo_id)
    elif os.path.isdir(repo_id):
        yield from enumerate_directory(repo_id)
    else:
        if variant:
            variant_part = f'.{variant}.'
        else:
            variant_part = '.'

        unet = find_diffuser_weights(
            os.path.join(subfolder, 'unet') if subfolder else 'unet')

        transformer = find_diffuser_weights(
            os.path.join(subfolder, 'transformer') if subfolder else 'transformer')

        prior = find_diffuser_weights(
            os.path.join(subfolder, 'prior') if subfolder else 'prior')

        decoder = find_diffuser_weights(
            os.path.join(subfolder, 'decoder') if subfolder else 'decoder')

        lora_search_dir = subfolder if subfolder else ''

        lora = _hf_try_to_load_from_cache(
            repo_id=repo_id,
            revision=revision,
            filename=os.path.join(lora_search_dir,
                                  f'pytorch_lora_weights{variant_part}safetensors')
        )

        if not isinstance(lora, str):
            lora = _hf_try_to_load_from_cache(
                repo_id=repo_id,
                revision=revision,
                filename=os.path.join(lora_search_dir, f'pytorch_lora_weights{variant_part}bin')
            )

        lora = post_discover_check(lora)

        top_level_weights = None
        other = None

        if not isinstance(lora, str):

            top_level_weights = find_diffuser_weights(
                subfolder if subfolder else '')

            if not top_level_weights:
                other_search_name = 'config.json' if not weight_name else weight_name

                other_name = other_search_name if not subfolder \
                    else os.path.join(subfolder, other_search_name)

                other = _hf_try_to_load_from_cache(
                    repo_id=repo_id,
                    revision=revision,
                    filename=other_name
                )

                other = post_discover_check(other)

        def yield_with_check(itr):
            for f, s in itr:
                if post_discover_check(f) is not None:
                    yield f, s

        if unet:
            # 2 directories up from the cached file
            # found in the unet folder, IE, top level
            yield from yield_with_check(
                enumerate_directory(str(pathlib.Path(unet).parents[1])))
        elif transformer:
            yield from yield_with_check(
                enumerate_directory(str(pathlib.Path(transformer).parents[1])))
        elif prior:
            yield from yield_with_check(
                enumerate_directory(str(pathlib.Path(prior).parents[1])))
        elif decoder:
            yield from yield_with_check(
                enumerate_directory(str(pathlib.Path(decoder).parents[1])))
        elif lora:
            # One file probably a lora
            yield from yield_with_check(
                enumerate_file(lora))
        elif top_level_weights:
            # One file
            yield from yield_with_check(
                enumerate_file(top_level_weights))
        elif other:
            # Everything in the folder where config.json
            # or was found, or whatever weight_name was found
            yield from yield_with_check(
                enumerate_directory(os.path.dirname(other)))
        elif not local_files_only:
            # Nothing matching what we are expecting in the
            # huggingface cache, ask the API

            _messages.debug_log('Fetching Model File Info with huggingface API call:',
                                'api.list_files_info(' +
                                str({'repo_id': repo_id, 'revision': revision, 'paths': subfolder}) + ')')

            api = huggingface_hub.HfApi(token=use_auth_token)

            try:
                info_entries = list(
                    i for i in api.list_repo_tree(
                        repo_id,
                        revision=revision,
                        path_in_repo=subfolder,
                        recursive=True) if isinstance(i, huggingface_hub.hf_api.RepoFile))
            except (huggingface_hub.utils.HFValidationError,
                    huggingface_hub.utils.HfHubHTTPError) as e:
                raise ModelNotFoundError(e)

            def detect_unet(path):
                try:
                    return str(pathlib.Path(path).parents[-2]) == 'unet'
                except IndexError:
                    return False

            have_unet = any(detect_unet(info.rfilename) for info in info_entries)

            try:
                for info in info_entries:
                    if have_unet and not weight_name and not os.path.dirname(info.rfilename):
                        # Ignore top level directory when a unet
                        # folder is detected unless weight_name is specified
                        continue

                    normalized_filename = post_discover_check(
                        os.path.join(*os.path.split(info.rfilename)))

                    if normalized_filename:
                        yield normalized_filename, info.size
            except Exception as e:
                _messages.debug_log('huggingface API error: ', e)
        else:
            yield from ()


def estimate_model_memory_use(repo_id: str,
                              revision: str | None = 'main',
                              variant: str | None = None,
                              subfolder: str | None = None,
                              weight_name: str | None = None,
                              safety_checker: bool = False,
                              include_unet: bool = True,
                              include_vae: bool = True,
                              include_text_encoder: bool = True,
                              include_text_encoder_2: bool = True,
                              include_text_encoder_3: bool = True,
                              safetensors: bool = True,
                              sentencepiece: bool = False,
                              watermarker: bool = False,
                              use_auth_token: str | None = None,
                              local_files_only: bool = False) -> int:
    """
    Attempt to estimate the CPU side memory consumption of a model before it is loaded into memory.

    Either from huggingface disk cache or through the huggingface API if not on disk and ``local_files_only`` is ``False``.

    This function also works on blob links, paths to folders, or singular files on disk.


    :param repo_id: huggingface repo_id, or path to folder or file on disk
    :param revision: repo revision, IE: branch
    :param variant: files variant, IE: fp16
    :param subfolder: subfolder in the repo where the models exist
    :param weight_name: look for a specific model file name
    :param safety_checker: include the safety checker model if it exists?
    :param include_unet: include the unet model if it exists?
    :param include_vae: include the vae model if it exists?
    :param include_text_encoder: include the text encoder model if it exists?
    :param include_text_encoder_2: include the second text encoder model if it exists?
    :param include_text_encoder_3: include the third text encoder model if it exists?
    :param safetensors: Use safetensors if available?
    :param sentencepiece: Forcibly include tokenizer/spiece.model for models with a unet?
    :param watermarker: Forcibly include watermarker/diffusion_pytorch_model.bin for models with a unet?
    :param use_auth_token: optional huggingface auth token
    :param local_files_only: should we only look for files cached on disk and never hit the API?
    :return: estimated size in bytes
    """
    __debug_args = locals()

    _messages.debug_log(
        f'{_types.fullname(estimate_model_memory_use)}({__debug_args})')

    if safetensors:
        try:
            import safetensors
        except ImportError:
            safetensors = None

    directories = {}

    for file, size in fetch_model_files_with_size(repo_id,
                                                  revision=revision,
                                                  variant=variant,
                                                  subfolder=subfolder,
                                                  weight_name=weight_name,
                                                  use_auth_token=use_auth_token,
                                                  local_files_only=local_files_only,
                                                  extensions={'.safetensors',
                                                              '.bin'},
                                                  sentencepiece=sentencepiece,
                                                  watermarker=watermarker):
        d = os.path.dirname(file)
        d = '.' if not d else d

        file = os.path.relpath(file, d)

        entry = directories.get(d)

        if entry:
            entry[file] = size
        else:
            directories[d] = {file: size}

    if safetensors:
        extensions = {'.safetensors'}
    else:
        extensions = {'.bin'}

    def estimate(forced_only=False):
        size_sum = 0
        if 'unet' in directories \
                or 'prior' in directories \
                or 'decoder' in directories \
                or 'transformer' in directories:

            important_directories = set()

            if 'decoder' in directories:
                important_directories.add('vqgan')

            if 'prior' in directories:
                important_directories.add('image_encoder')

            if include_unet:
                important_directories.add('unet')
                important_directories.add('prior')
                important_directories.add('decoder')
                important_directories.add('transformer')

            if not forced_only and include_vae:
                important_directories.add('vae')

            if not forced_only and include_text_encoder:
                important_directories.add('text_encoder')

            if not forced_only and include_text_encoder_2:
                important_directories.add('text_encoder_2')

            if not forced_only and include_text_encoder_3:
                important_directories.add('text_encoder_3')

            if forced_only and sentencepiece:
                important_directories.add('tokenizer')

            if forced_only and watermarker:
                important_directories.add('watermarker')

            for directory, files in directories.items():
                if directory not in important_directories:
                    continue

                for file, size in files.items():
                    base, ext = os.path.splitext(file)

                    forced_include = sentencepiece and (
                            directory == 'tokenizer' and os.path.basename(file) == 'spiece.model')
                    forced_include |= watermarker and (
                            directory == 'watermarker' and os.path.basename(file) == 'diffusion_pytorch_model.bin')
                    forced_include |= safety_checker and directory == 'safety_checker'

                    if forced_only and not forced_include:
                        continue

                    if not forced_only and ext not in extensions:
                        continue

                    size_sum += size

                    _messages.debug_log(
                        'Estimate Considering:',
                        os.path.join(directory, file) + f', Size: {size} Bytes')

            return size_sum
        else:
            directory = '.' if not subfolder else subfolder
            directory_entry = directories.get(directory)

            if not directory_entry:
                return 0

            for file, size in directory_entry.items():
                _, ext = os.path.splitext(file)
                if ext not in extensions:
                    continue
                _messages.debug_log('Estimate Considering:',
                                    os.path.join(directory if directory != '.' else '', file) +
                                    f', Size: {size} Bytes')
                size_sum += size
        return size_sum

    e = estimate()
    if e == 0:
        if not weight_name and safetensors:
            extensions = {'.bin'}
            e = estimate()

    if sentencepiece or watermarker:
        e += estimate(forced_only=True)

    _messages.debug_log(
        f'{_types.fullname(estimate_model_memory_use)}() = {e} Bytes')

    return e


# noinspection HttpUrlsUsage
def is_single_file_model_load(path):
    """
    Should we use :py:meth:`diffusers.loaders.FromSingleFileMixin.from_single_file` on this path?

    :param path: The path
    :return: ``True`` or ``False``
    """
    path, ext = os.path.splitext(path)

    if path.startswith('http://') or path.startswith('https://'):
        return True

    if os.path.isfile(path):
        return True

    if not ext:
        return False

    if ext in {'.pt', '.pth', '.bin', '.ckpt', '.safetensors'}:
        return True

    return False


__all__ = _types.module_all()
