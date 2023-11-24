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
import glob
import os
import pathlib
import re
import typing

import huggingface_hub

import dgenerate.messages as _messages
import dgenerate.types as _types


class ModelNotFoundError(Exception):
    """Raised when a specified model can not be located either locally or remotely"""
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
                 repo_id,
                 revision,
                 subfolder,
                 weight_name):
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


def variant_match(filename: str, variant: typing.Optional[str] = None):
    """
    Match a model filename against a huggingface variant specifier such as "fp16"

    If no variant is specified and the filename contains a variant, this function will
    return false.

    Examples:

        * ``variant_match('file.fp16-0001-of-0002.bin', variant=None) -> False``
        * ``variant_match('file.bin', variant=None) -> True``

    If a variant is specified and the filename contains a matching
    variant specifier, this function will return True.

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
                               cache_dir: typing.Union[str, pathlib.Path, None] = None,
                               revision: typing.Optional[str] = None,
                               repo_type: typing.Optional[str] = None):
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


def fetch_model_files_with_size(repo_id: str,
                                revision: typing.Optional[str] = 'main',
                                variant: typing.Optional[str] = None,
                                subfolder: typing.Optional[str] = None,
                                weight_name: typing.Optional[str] = None,
                                use_auth_token: typing.Optional[str] = None,
                                extensions: typing.Optional[typing.Union[set, list]] = None,
                                local_files_only: bool = False,
                                flax: bool = False,
                                sentencepiece: bool = False,
                                watermarker: bool = False) -> typing.Iterator[tuple[str, int]]:
    """
    Attempt to fetch model files with their size that are relevant for the type of model being loaded.

    Either from huggingface disk cache or through the huggingface API if not on disk and local_files_only is False.

    This function also works on blob links, paths to folders, or singular files on disk.



    :param repo_id: huggingface repo_id, or path to folder or file on disk
    :param revision: repo revision, IE: branch
    :param variant: files variant, IE: fp16
    :param subfolder: subfolder in the repo where the models exist
    :param weight_name: look for a specific model file name
    :param use_auth_token: optional huggingface auth token
    :param extensions: if specified, only search for extensions in this set, or list
    :param local_files_only: utilize the huggingface API if necessary?
        if this is True, and it is necessary to fetch info from the API, this function
        will simply yield nothing
    :param flax: if False, only look for torch diffusion weights.
        If True, only look for flax diffusion weights.
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
        if flax:
            found = _hf_try_to_load_from_cache(
                repo_id=repo_id,
                revision=revision,
                filename=os.path.join(search_dir,
                                      f'diffusion_flax_model{variant_part}msgpack')
            )
        else:
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
                info_entries = list(api.list_files_info(repo_id,
                                                        revision=revision,
                                                        paths=subfolder))
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
                              revision: typing.Optional[str] = 'main',
                              variant: typing.Optional[str] = None,
                              subfolder: typing.Optional[str] = None,
                              weight_name: typing.Optional[str] = None,
                              safety_checker: bool = False,
                              include_vae: bool = True,
                              include_text_encoder: bool = True,
                              include_text_encoder_2: bool = True,
                              safetensors: bool = True,
                              flax: bool = False,
                              sentencepiece: bool = False,
                              watermarker: bool = False,
                              use_auth_token: typing.Optional[str] = None,
                              local_files_only: bool = False) -> int:
    """
    Attempt to estimate the CPU side memory consumption of a model before it is loaded into memory.

    Either from huggingface disk cache or through the huggingface API if not on disk and local_files_only is False.

    This function also works on blob links, paths to folders, or singular files on disk.

    :param repo_id: huggingface repo_id, or path to folder or file on disk
    :param revision: repo revision, IE: branch
    :param variant: files variant, IE: fp16
    :param subfolder: subfolder in the repo where the models exist
    :param weight_name: look for a specific model file name
    :param safety_checker: include the safety checker model if it exists?
    :param include_vae: include the vae model if it exists?
    :param include_text_encoder: include the text encoder model if it exists?
    :param include_text_encoder_2: include the second text encoder model if it exists?
    :param safetensors: Use safetensors if available?
    :param flax: Only look for msgpack files?
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
                                                  extensions={'.msgpack',
                                                              '.safetensors',
                                                              '.bin'},
                                                  flax=flax,
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

    if flax:
        extensions = {'.msgpack'}
    else:
        if safetensors:
            extensions = {'.safetensors'}
        else:
            extensions = {'.bin'}

    def estimate(forced_only=False):
        size_sum = 0
        if 'unet' in directories:

            important_directories = {
                'unet'
            }

            if not forced_only and include_vae:
                important_directories.add('vae')

            if not forced_only and include_text_encoder:
                important_directories.add('text_encoder')

            if not forced_only and include_text_encoder_2:
                important_directories.add('text_encoder_2')

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
        if not flax and not weight_name and safetensors:
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
    :return: true or false
    """
    path, ext = os.path.splitext(path)

    if path.startswith('http://') or path.startswith('https://'):
        return True

    if os.path.isdir(path):
        return True

    if not ext:
        return False

    if ext in {'.pt', '.pth', '.bin', '.msgpack', '.ckpt', '.safetensors'}:
        return True

    return False


__all__ = _types.module_all()
