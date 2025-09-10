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
import importlib.util
import itertools
import os
import pathlib
import platform as _platform
import typing

import PIL.Image
import fake_useragent
import pyrfc6266
import requests
import torch
import tqdm

import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.image as _image
import dgenerate.memory
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.renderloop as _renderloop
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.webcache as _webcache


def _format_prompt_single(prompt):
    pos = prompt.positive
    neg = prompt.negative

    if pos is None:
        raise _batchprocessor.BatchProcessError('Attempt to format a prompt with no positive prompt value.')

    if pos and neg:
        return _textprocessing.shell_quote(f"{pos}; {neg}")
    return _textprocessing.shell_quote(pos)


def format_prompt(
        prompts: _prompt.Prompt | collections.abc.Iterable[_prompt.Prompt]) -> str:
    """
    Format a prompt object, or a list of prompt objects, into quoted string(s)
    """
    if isinstance(prompts, _prompt.Prompt):
        return _format_prompt_single(prompts)
    return ' '.join(_format_prompt_single(p) for p in prompts)


def format_size(size: collections.abc.Iterable[int]) -> str:
    """
    Join an iterable of integers into a string seperated by the character 'x', for example (512, 512) -> "512x512"
    """
    return _textprocessing.format_size(size)


def quote(
        strings: str | collections.abc.Iterable[typing.Any],
        double: bool = False,
        quotes: bool = True
) -> str:
    """
    Shell quote a string or iterable of strings.

    The "double" argument allows you to change the outer quote character to double quotes.

    The "quotes" argument determines whether to ddd quotes. If ``False``, only add the
    proper escape sequences and no surrounding quotes. This can be useful for templating
    extra string content into an existing string.
    """
    if isinstance(strings, str):
        return _textprocessing.shell_quote(
            strings,
            double=double,
            quotes=quotes
        )
    return ' '.join(_textprocessing.shell_quote(str(s)) for s in strings)


def unquote(
        strings: str | collections.abc.Iterable[typing.Any],
        expand: bool = False,
        glob_hidden: bool = False,
        glob_recursive: bool = False
) -> list:
    """
    Un-Shell quote a string or iterable of strings (shell parse)

    The "expand" argument can be used to indicate that you wish to expand
    shell globs and the home directory operator.

    The "glob_hidden" argument can be used to indicate that hidden files
    should be included in globs when expand is True.

    The "glob_recursive" argument can be used to indicate that globbing
    should be recursive when expand is True.
    """
    if isinstance(strings, str):
        return _textprocessing.shell_parse(
            strings,
            expand_home=expand,
            expand_glob=expand,
            expand_vars=False,
            glob_hidden=glob_hidden,
            glob_recursive=glob_recursive
        )
    return list(
        itertools.chain.from_iterable(
            _textprocessing.shell_parse(
                str(s),
                expand_home=expand,
                expand_glob=expand,
                expand_vars=False,
                glob_hidden=glob_hidden,
                glob_recursive=glob_recursive) for s in strings))


def last(iterable: list | collections.abc.Iterable[typing.Any]) -> typing.Any:
    """
    Return the last element in an iterable collection.
    """
    if isinstance(iterable, list):
        return iterable[-1]
    try:
        *_, last_item = iterable
    except ValueError:
        raise _batchprocessor.BatchProcessError(
            'Usage of template function "last" on an empty iterable.')
    return last_item


def first(iterable: collections.abc.Iterable[typing.Any]) -> typing.Any:
    """
    Return the first element in an iterable collection.
    """
    try:
        v = next(iter(iterable))
    except StopIteration:
        raise _batchprocessor.BatchProcessError(
            'Usage of template function "first" on an empty iterable.')
    return v


def gen_seeds(n: int) -> list[str]:
    """
    Generate N random integer seeds (as strings) and return a list of them.
    """
    return [str(s) for s in _renderloop.gen_seeds(int(n))]


def cwd() -> str:
    """
    Return the current working directory as a string.
    """
    return pathlib.Path.cwd().as_posix()


def format_model_type(model_type: _pipelinewrapper.ModelType) -> str:
    """
    Return the string representation of a ModelType enum.
    This can be used to get command line compatible --model-type
    string from the last_model_type template variable.
    """
    return _pipelinewrapper.get_model_type_string(model_type)


def format_dtype(dtype: _pipelinewrapper.DataType) -> str:
    """
    Return the string representation of a DataType enum.
    This can be used to get command line compatible --dtype
    string from the last_dtype template variable.
    """
    return _pipelinewrapper.get_data_type_string(dtype)


def download(url: str,
             output: str | None = None,
             overwrite: bool = False,
             text: bool = False) -> str:
    """
    Download a file from a URL to the web cache or a specified path,
    and return the file path to the downloaded file.

    NOWRAP!
    \\set my_variable {{ download('https://modelhost.com/model.safetensors' }}

    NOWRAP!
    \\set my_variable {{ download('https://modelhost.com/model.safetensors', output='model.safetensors') }}

    NOWRAP!
    \\set my_variable {{ download('https://modelhost.com/model.safetensors', output='directory/' }}

    NOWRAP!
    \\setp my_variable download('https://modelhost.com/model.safetensors')

    When an "output" path is specified, if the file already exists it
    will be reused by default (simple caching behavior), this can be disabled
    with the argument "overwrite=True" indicating that the file should
    always be downloaded.

    "overwrite=True" can also be used to overwrite cached
    files in the dgenerate web cache.

    An error will be raised by default if a text mimetype is encountered,
    this can be overridden with "text=True"

    Be weary that if you have a long-running loop in your config using
    a top level jinja template, which refers to your template variable,
    cache expiry may invalidate the file stored in your variable.

    You can rectify this by using the template function inside your loop.
    """

    def mimetype_supported(mimetype):
        if text:
            return True
        return mimetype is None or not mimetype.startswith('text/')

    if output:
        cache_key = f'download pointer: {url}, output: {os.path.abspath(output)}'

        if not overwrite:
            with _webcache.cache as web_cache:
                cache_pointer = _webcache.cache.get(cache_key)
                if cache_pointer is not None:
                    if not os.path.exists(cache_pointer.path):
                        del web_cache[cache_key]
                    else:
                        with open(cache_pointer.path, 'rt', encoding='utf8') as pointer_file:
                            downloaded_file = pointer_file.read().strip()
                        if os.path.exists(downloaded_file):
                            _messages.log(
                                f'Downloaded file already exists, using: '
                                f'{os.path.relpath(downloaded_file)}', underline=True)
                            return pathlib.Path(downloaded_file).as_posix()
                        else:
                            del web_cache[cache_key]

        try:
            with requests.get(
                _webcache._append_tokens_to_url(url),
                headers={'User-Agent': fake_useragent.UserAgent().chrome},
                          stream=True,
                          timeout=5
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get('content-type', 'unknown')

                if not mimetype_supported(content_type):
                    raise _batchprocessor.BatchProcessError(
                        f'Encountered text/* mimetype at "{url}" '
                        'without specifying the -t/--text argument.')

                if output.endswith('/') or output.endswith('\\'):
                    os.makedirs(output, exist_ok=True)
                    output = os.path.join(
                        output, pyrfc6266.requests_response_to_filename(response))

                total_size = int(response.headers.get('content-length', 0))

                if not overwrite and os.path.exists(output):
                    _messages.log(f'Downloaded file already exists, using: '
                                  f'{os.path.normpath(output)}',
                                  underline=True)
                    _webcache.cache.add(
                        cache_key,
                        os.path.abspath(output).encode('utf8'))
                    return pathlib.Path(output).absolute().as_posix()

                _messages.log(f'Downloading: "{url}"\n'
                              f'Destination: "{output}"',
                              underline=True)

                chunk_size = _memory.calculate_chunk_size(total_size)
                current_dl = output + '.unfinished'

                with open(current_dl, 'wb') as file:
                    if chunk_size != total_size:
                        with tqdm.tqdm(total=total_size if total_size != 0 else None,
                                       unit='iB',
                                       unit_scale=True) as progress_bar:
                            for chunk in response.iter_content(
                                    chunk_size=chunk_size):
                                if chunk:
                                    progress_bar.update(len(chunk))
                                    file.write(chunk)
                                    file.flush()
                            downloaded_size = progress_bar.n
                    else:
                        content = response.content
                        downloaded_size = len(content)
                        file.write(content)
                        file.flush()

                if total_size != 0 and downloaded_size != total_size:
                    raise _batchprocessor.BatchProcessError(
                        'Download failure, something went wrong '
                        f'downloading "{url}".', )

                os.replace(current_dl, output)

                file_path = os.path.abspath(output)

                _webcache.cache.add(
                    cache_key,
                    file_path.encode('utf8'))

        except requests.RequestException as e:
            raise _batchprocessor.BatchProcessError(
                f'Failed to download "{url}": {e}') from e
    else:
        class _MimeExcept(Exception):
            pass

        try:
            _, file_path = _webcache.create_web_cache_file(
                url,
                mime_acceptable_desc=None if text else 'not text',
                mimetype_is_supported=mimetype_supported,
                unknown_mimetype_exception=_MimeExcept,
                overwrite=overwrite)
        except _MimeExcept as e:
            raise _batchprocessor.BatchProcessError(
                f'Encountered text/* mimetype at "{url}" '
                'without specifying the -t/--text argument.') from e
        except requests.RequestException as e:
            raise _batchprocessor.BatchProcessError(f'Failed to download "{url}": {e}') from e

    return pathlib.Path(file_path).as_posix()


def align_size(size: str | tuple, align: int, format_size: bool = True) -> str | tuple:
    """
    Align a string dimension such as "700x700", or a tuple dimension such as (700, 700) to a
    specific alignment value ("align") and format the result to a string dimension recognized by dgenerate.

    This function expects a string with the format WIDTHxHEIGHT, or just WIDTH, or a tuple of dimensions.

    It returns a string in the same format with the dimension aligned to
    the specified amount, unless "format_size" is False, in which case it will
    return a tuple.
    """
    if align < 1:
        raise _batchprocessor.BatchProcessError(
            'Argument "align" of align_size may not be less than 1.')

    if isinstance(size, str):
        aligned = _image.align_by(_textprocessing.parse_dimensions(size), align)
    elif isinstance(size, tuple):
        aligned = _image.align_by(size, align)
    else:
        raise _batchprocessor.BatchProcessError(
            'Unsupported type passed to align_size.')

    if not format_size:
        return aligned

    return _textprocessing.format_size(aligned)


def pow2_size(size: str | tuple, format_size: bool = True) -> str | tuple:
    """
    Round a string dimension such as "700x700", or a tuple dimension such as (700, 700) to
    the nearest power of 2 and format the result to a string dimension recognized by dgenerate.

    This function expects a string with the format WIDTHxHEIGHT, or just WIDTH, or a tuple of dimensions.

    It returns a string in the same format with the dimension rounded to
    the nearest power of 2, unless "format_size" is False, in which case it will
    return a tuple.
    """
    if isinstance(size, str):
        aligned = _image.nearest_power_of_two(_textprocessing.parse_dimensions(size))
    elif isinstance(size, tuple):
        aligned = _image.nearest_power_of_two(size)
    else:
        raise _batchprocessor.BatchProcessError(
            'Unsupported type passed to pow2_size.')

    if not format_size:
        return aligned

    return _textprocessing.format_size(aligned)


def image_size(file: str, format_size: bool = True) -> str | tuple[int, int]:
    """
    Return the width and height of an image file on disk.

    If "format_size" is False, return a tuple instead of a WIDTHxHEIGHT string.
    """

    with PIL.Image.open(file) as img:
        if not format_size:
            return img.width, img.height

        return _textprocessing.format_size((img.width, img.height))


def size_is_aligned(size: str | tuple, align: int) -> bool:
    """
    Check if a string dimension such as "700x700", or a tuple dimension such as (700, 700)
    is aligned to a specific ("align") value. Returns True or False.

    This function expects a string with the format WIDTHxHEIGHT, or just WIDTH, or a tuple of dimensions.
    """
    if align < 1:
        raise _batchprocessor.BatchProcessError(
            'Argument "align" of size_is_aligned may not be less than 1.')

    if isinstance(size, str):
        aligned = _image.is_aligned(_textprocessing.parse_dimensions(size), align)
    elif isinstance(size, tuple):
        aligned = _image.is_aligned(size, align)
    else:
        raise _batchprocessor.BatchProcessError(
            'Unsupported type passed to size_is_aligned.')

    return aligned


def size_is_pow2(size: str | tuple) -> bool:
    """
    Check if a string dimension such as "700x700", or a tuple dimension such as (700, 700)
    is a power of 2 dimension. Returns True or False.

    This function expects a string with the format WIDTHxHEIGHT, or just WIDTH, or a tuple of dimensions.
    """

    if isinstance(size, str):
        aligned = _image.is_power_of_two(_textprocessing.parse_dimensions(size))
    elif isinstance(size, tuple):
        aligned = _image.is_power_of_two(size)
    else:
        raise _batchprocessor.BatchProcessError(
            'Unsupported type passed to size_is_pow2.')

    return aligned


def have_feature(feature_name: str) -> bool:
    """
    Return a boolean value indicating if dgenerate has a specific feature available.

    Currently accepted values are:

    NOWRAP!
    "ncnn": Do we have ncnn installed?
    "gpt4all": Do we have gpt4all installed?
    "bitsandbytes": Do we have bitsandbytes installed?
    "flash-attn": Do we have flash-attn installed?
    "xformers": Do we have xformers installed?
    "triton": Do we have triton installed?
    """

    known_flags = [
        'ncnn',
        'gpt4all',
        'bitsandbytes',
        'flash-attn',
        'xformers',
        'triton',
    ]

    if feature_name not in known_flags:
        raise _batchprocessor.BatchProcessError(
            f'Feature "{feature_name}" is not a known feature flag, '
            f'acceptable values are: {_textprocessing.oxford_comma(known_flags, "or")}')

    return importlib.util.find_spec(feature_name) is not None


def platform() -> str:
    """
    Return platform.system()

    Returns the system/OS name, such as 'Linux', 'Darwin', 'Java', 'Windows'.

    An empty string is returned if the value cannot be determined.
    """

    return _platform.system()


def frange(start, stop=None, step=0.1):
    """
    Like range, but for floating point numbers.

    The default step value is 0.1
    """

    if stop is None:
        stop = start
        start = 0.0
    current = start
    while current < stop:
        yield round(current, 10)
        current += step


def default_device() -> str:
    """
    Return the name of the default accelerator device on the system.
    """
    return dgenerate.default_device()


def have_cuda() -> bool:
    """
    Check if CUDA backend is available.
    """
    return _torchutil.is_cuda_available()


def have_xpu() -> bool:
    """
    Check if XPU backend is available.
    """
    return _torchutil.is_xpu_available()


def have_mps() -> bool:
    """
    Check if MPS backend is available.
    """
    return _torchutil.is_mps_available()


def total_memory(device: str | None = None, unit: str = 'b'):
    """
    Get the total ram that a specific device possesses.

    This will always return 0 for "mps".

    The "device" argument specifies the device, if none is
    specified, the systems default accelerator will be used,
    if a GPU is installed, it will be the first GPU.

    The "unit" argument specifies the unit you want returned,
    must be one of (case insensitive): b (bytes), kb (kilobytes),
    mb (megabytes), gb (gigabytes), kib (kibibytes),
    mib (mebibytes), gib (gibibytes)
    """

    if device is None:
        device = dgenerate.default_device()

    device = torch.device(device)

    if device.type == 'cpu':
        return _memory.get_total_memory(unit)
    else:
        return _memory.get_gpu_total_memory(device, unit)


def import_module(module_name: str) -> typing.Any:
    """
    Import a Python module by name and return the module object.

    If the module cannot be imported, an error will be raised.

    See also the directive: \\import
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f'Failed to import python module "{module_name}": {e}') from e


def csv(iterable: typing.Iterable):
    """
    Convert an iterable into a CSV formatted string.
    """

    return ','.join(str(item) for item in iterable)
