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

import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.filecache as _filecache
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

    An interrupted download is kept beside the destination with an
    ``.unfinished`` suffix and continued on the next call.

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

        class _UseExisting(Exception):
            def __init__(self, path):
                self.path = path

        state = {}

        def resolve_path(response):
            nonlocal output
            content_type = response.headers.get('content-type', 'unknown')

            if not mimetype_supported(content_type):
                raise _batchprocessor.BatchProcessError(
                    f'Encountered text/* mimetype at "{url}" '
                    'without specifying the -t/--text argument.')

            if output.endswith('/') or output.endswith('\\'):
                os.makedirs(output, exist_ok=True)
                output = os.path.join(
                    output, pyrfc6266.requests_response_to_filename(response))

            if not overwrite and os.path.exists(output):
                raise _UseExisting(output)

            _messages.log(f'Downloading: "{url}"\n'
                          f'Destination: "{output}"',
                          underline=True)
            state['output'] = output
            return output + '.unfinished'

        try:
            try:
                _filecache.download_url_to_file(
                    _webcache._append_tokens_to_url(url),
                    headers={'User-Agent': fake_useragent.UserAgent().chrome},
                    resolve_path=resolve_path)
            except _UseExisting as existing:
                _messages.log(f'Downloaded file already exists, using: '
                              f'{os.path.normpath(existing.path)}',
                              underline=True)
                _webcache.cache.add(
                    cache_key,
                    os.path.abspath(existing.path).encode('utf8'))
                return pathlib.Path(existing.path).absolute().as_posix()

            os.replace(state['output'] + '.unfinished', state['output'])

            file_path = os.path.abspath(state['output'])

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


def image_width(file: str) -> int:
    """
    Return the width of an image file on disk as an integer.

    Useful for arithmetic in ``\\setp``, for example:
    ``\\setp w image_width("input.png") * 2``
    """
    with PIL.Image.open(file) as img:
        return img.width


def image_height(file: str) -> int:
    """
    Return the height of an image file on disk as an integer.

    Useful for arithmetic in ``\\setp``, for example:
    ``\\setp h image_height("input.png") * 2``
    """
    with PIL.Image.open(file) as img:
        return img.height


def _parse_scale_factors(scale: float | int | str | collections.abc.Sequence) -> tuple[float, float]:
    """
    Parse a uniform or per-axis scale into ``(scale_width, scale_height)``.
    """
    if isinstance(scale, bool):
        raise _batchprocessor.BatchProcessError(
            'Argument "scale" of scale_size must be a number, '
            '"WxH" string, or a sequence of one or two numbers.')

    if isinstance(scale, (int, float)):
        value = float(scale)
        return value, value

    if isinstance(scale, str):
        parts = [p.strip() for p in scale.lower().split('x')]
        try:
            if len(parts) == 1 and parts[0]:
                value = float(parts[0])
                return value, value
            if len(parts) == 2 and parts[0] and parts[1]:
                return float(parts[0]), float(parts[1])
        except ValueError as e:
            raise _batchprocessor.BatchProcessError(
                'Argument "scale" of scale_size string must be a number '
                'or WIDTHxHEIGHT factors such as "2x1.5".') from e
        raise _batchprocessor.BatchProcessError(
            'Argument "scale" of scale_size string must be a number '
            'or WIDTHxHEIGHT factors such as "2x1.5".')

    if isinstance(scale, collections.abc.Sequence):
        try:
            values = [float(v) for v in scale]
        except (TypeError, ValueError) as e:
            raise _batchprocessor.BatchProcessError(
                'Argument "scale" of scale_size sequence must contain numbers.') from e
        if len(values) == 1:
            return values[0], values[0]
        if len(values) == 2:
            return values[0], values[1]
        raise _batchprocessor.BatchProcessError(
            'Argument "scale" of scale_size sequence must have 1 or 2 values.')

    raise _batchprocessor.BatchProcessError(
        'Argument "scale" of scale_size must be a number, '
        '"WxH" string, or a sequence of one or two numbers.')


def scale_size(
        size: str | tuple,
        scale: float | int | str | collections.abc.Sequence = 1,
        format_size: bool = True) -> str | tuple:
    """
    Scale a dimension or an image file's dimensions by a factor.

    "size" may be a WIDTHxHEIGHT string such as "512x768", a tuple such as
    (512, 768), or a path to an image file on disk. If a string cannot be
    parsed as a dimension, it is treated as an image file path.

    "scale" may be:

    * a single number applied to both width and height
    * a ``(scale_width, scale_height)`` sequence for independent axes
    * a ``"WxH"`` string of scale factors such as ``"2x1.5"``

    Results are rounded to the nearest integer and clamped to a minimum of 1.

    Returns a WIDTHxHEIGHT string unless "format_size" is False, in which
    case a tuple of integers is returned.

    Examples: scale_size("512x512", 2) -> "1024x1024",
    scale_size((512, 768), 1.5) -> "768x1152",
    scale_size("512x768", (2, 1)) -> "1024x768",
    scale_size("512x768", "2x1.5") -> "1024x1152",
    scale_size("photo.png", 2) -> scaled dimensions of photo.png,
    ``\\setp out_size scale_size("input.png", (2, 1))``
    """
    scale_w, scale_h = _parse_scale_factors(scale)

    if isinstance(size, tuple):
        dims = size
    elif isinstance(size, str):
        try:
            dims = _textprocessing.parse_image_size(size)
        except ValueError:
            with PIL.Image.open(size) as img:
                dims = (img.width, img.height)
    else:
        raise _batchprocessor.BatchProcessError(
            'Unsupported type passed to scale_size.')

    try:
        if len(dims) == 1:
            dims = (dims[0], dims[0])
        if len(dims) != 2:
            raise ValueError('expected 1 or 2 dimensions')
        factors = (scale_w, scale_h)
        scaled = tuple(
            max(1, int(round(int(d) * float(f))))
            for d, f in zip(dims, factors))
    except (TypeError, ValueError) as e:
        raise _batchprocessor.BatchProcessError(
            f'Invalid dimensions passed to scale_size: {e}') from e

    if not format_size:
        return scaled

    return _textprocessing.format_size(scaled)


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
    "xllamacpp": Do we have xllamacpp installed?
    "bitsandbytes": Do we have bitsandbytes installed?
    "sdnq": Do we have sdnq installed?
    "flash-attn": Do we have flash-attn installed?
    "triton": Do we have triton installed?
    """

    known_flags = [
        'ncnn',
        'xllamacpp',
        'bitsandbytes',
        'sdnq',
        'flash-attn',
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
