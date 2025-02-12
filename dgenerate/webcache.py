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

import os
import pathlib
import sys
import typing
import urllib.parse
import tqdm

import dgenerate.filecache as _filecache
import dgenerate.textprocessing as _textprocessing

__doc__ = """
Single point of access to the global dgenerate web cache.
"""


def get_web_cache_directory() -> str:
    """
    Get the default web cache directory or the value of the environmental variable ``DGENERATE_WEB_CACHE``

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_WEB_CACHE')

    if user_cache_path is not None:
        path = user_cache_path
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'web'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


try:
    cache = _filecache.WebFileCache(
        os.path.join(get_web_cache_directory(), 'cache.db'),
        get_web_cache_directory(),
        expiry_delta=_textprocessing.parse_timedelta(
            os.environ.get('DGENERATE_WEB_CACHE_EXPIRY_DELTA', 'hours=12').strip('"\'')))
except _textprocessing.TimeDeltaParseError as e:
    print(f'DGENERATE_WEB_CACHE_EXPIRY_DELTA environmental variable not parseable: {e}',
          file=sys.stderr, flush=True)
    sys.exit(1)


def _append_tokens_to_url(url: str) -> str:
    # append API tokens from environment

    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.netloc == "civitai.com" and parsed_url.path.startswith("/api/download"):
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if 'CIVIT_AI_TOKEN' in os.environ:
            token_val = os.environ.get('CIVIT_AI_TOKEN')
            if token_val and not 'token' in query_params:
                query_params['token'] = token_val

        new_query = urllib.parse.urlencode(query_params, doseq=True)
        new_url = urllib.parse.urlunparse(parsed_url._replace(query=new_query))
        return new_url

    return url


def create_web_cache_file(url,
                          mime_acceptable_desc: str | None = None,
                          mimetype_is_supported: typing.Callable[[str], bool] | None = None,
                          unknown_mimetype_exception=ValueError,
                          overwrite: bool = False,
                          tqdm_pbar=tqdm.tqdm) \
        -> tuple[str, str]:
    """
    Download a file from a url and add it to dgenerates temporary web cache that is
    available to all concurrent dgenerate processes.

    If the file exists in the cache already, return information for the existing file.

    Append API tokens if applicable, such as `CIVIT_AI_TOKEN` from your environment.

    :raise requests.RequestException: Can raise any exception
        raised by ``requests.get`` for request related errors.

    :param url: The url
    :param mime_acceptable_desc: A description of acceptable mimetypes for use in exceptions.
    :param mimetype_is_supported: A function that determines if a mimetype is supported for downloading.
    :param unknown_mimetype_exception: The exception type to raise when an unknown mimetype is encountered.
    :param overwrite: Always overwrite any previously cached file?
    :param tqdm_pbar: tqdm progress bar type, if set to `None` no progress bar will be used. Defaults to `tqdm.tqdm`
    :return: tuple(mimetype_str, filepath)
    """

    cached_file = cache.download(
        _append_tokens_to_url(url),
        mime_acceptable_desc,
        mimetype_is_supported,
        unknown_mimetype_exception,
        overwrite,
        tqdm_pbar)

    return cached_file.metadata['mime-type'], cached_file.path


def request_mimetype(url) -> str:
    """
    Request the mimetype of a file at a URL, if the file exists in the cache, a known mimetype
    is returned without connecting to the internet. Otherwise, connect to the internet
    to retrieve the mimetype, this action does not update the cache.

    :param url: The url

    :return: mimetype string
    """

    return cache.request_mimetype(url)


def is_downloadable_url(string) -> bool:
    """
    Does a string represent a URL that can be downloaded into the web cache?

    :param string: the string
    :return: ``True`` or ``False``
    """
    return cache.is_downloadable_url(string)
