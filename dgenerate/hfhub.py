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
import os
import re
import typing
import huggingface_hub
import dgenerate.exceptions
import dgenerate.messages as _messages
import dgenerate.types as _types
import dgenerate.webcache as _webcache

__doc__ = """Hugging Face Hub utilities for supporting Hugging Face downloads."""


class NonHFDownloadError(Exception):
    """Raised when a non-Hugging Face download fails."""
    pass


class NonHFModelDownloadError(NonHFDownloadError):
    """Raised when a non-Hugging Face model download fails."""
    pass


class NonHFConfigDownloadError(NonHFDownloadError):
    """Raised when a non-Hugging Face config download fails."""
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

def is_single_file_model_load(path):
    """
    Should we use :py:meth:`from_single_file` on this path?

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


def webcache_or_hf_blob_download(url: str,
                                 mime_acceptable_desc: str | None = None,
                                 mimetype_is_supported: typing.Callable[[str], bool] | None = None,
                                 unknown_mimetype_exception: type[Exception] = NonHFDownloadError,
                                 local_files_only: bool = False) -> str:
    """
    Download to ``dgenerate`` web cache or Hugging Face cache, depending on the model path.

    If model path is a Hugging Face blob link, it will be downloaded to the Hugging Face cache.

    If not, it will be downloaded to the ``dgenerate`` web cache.

    TQDM progress bar is used for any download that occurs, TQDM progress bars will differ
    somewhat in appearance depending on whether the file is downloaded to the web cache or Hugging Face cache.


    :param url: The url
    :param mime_acceptable_desc: A description of acceptable mimetypes for use in exceptions. (dgenerate webcache)
    :param mimetype_is_supported: A function that determines if a mimetype is supported for downloading. (dgenerate webcache)
    :param unknown_mimetype_exception: The exception type to raise when an unknown mimetype is encountered. (dgenerate webcache)
    :param local_files_only: If ``True``, do not attempt to download files, only check cache.

    :raise NonHFDownloadError: If the download mimetype unsupported.

    :raise dgenerate.webcache.WebFileCacheOfflineModeException: If ``local_files_only`` is ``True`` and a
        download is required for a non Hugging Face blob link. This will occur if the file in question
        is not found in the dgenerate web cache. This can also occur if the :py:mod:`dgenerate.webcache``
        module is in global offline mode.

    :raise huggingface_hub.errors.HFValidationError: If the Hugging Face blob link is invalid.

    :raise huggingface_hub.errors.HfHubHTTPError: If the Hugging Face blob link is valid
        but the file could not be downloaded. This can also occur if ``local_files_only`` is ``True``
        and the file is not found in the cache.

    :raise huggingface_hub.errors.OfflineModeIsEnabled: If global offline mode is enabled
        for ``huggingface_hub`` and the file is not found in the cache.

    :return: filepath
    """
    blob_link = HFBlobLink.parse(url)

    if blob_link is None:
        _, model_path = _webcache.create_web_cache_file(
            url,
            mime_acceptable_desc=mime_acceptable_desc,
            mimetype_is_supported=mimetype_is_supported,
            unknown_mimetype_exception=unknown_mimetype_exception,
            local_files_only=local_files_only
        )
    else:
        model_path = huggingface_hub.hf_hub_download(
            repo_id=blob_link.repo_id,
            revision=blob_link.revision,
            subfolder=blob_link.subfolder,
            filename=blob_link.weight_name,
            local_files_only=local_files_only
        )
    return model_path


def download_non_hf_slug_model(model_path: str, local_files_only: bool = False):
    """
    Check for a non hugging face slug or reference to a model that is possibly downloadable as a single file.

    If this is applicable, download it to the web cache and return its path.
    If the file already exists in the web cache simply return a path to it.

    Hugging Face blob links are also supported, in which case the file will be downloaded to the huggingface cache.

    If this is not applicable, return the path unchanged.

    TQDM progress bar is used for any download that occurs.

    :raise NonHFModelDownloadError: If the download mimetype is ``None`` or ``text/*``.

    :raise dgenerate.webcache.WebFileCacheOfflineModeException: If ``local_files_only`` is ``True`` and a
        download is required for a non Hugging Face blob link. This will occur if the file in question
        is not found in the dgenerate web cache. This can also occur if the :py:mod:`dgenerate.webcache``
        module is in global offline mode.

    :raise huggingface_hub.errors.HfHubHTTPError: If the Hugging Face blob link is valid
        but the file could not be downloaded. This can also occur if ``local_files_only`` is ``True``
        and the file is not found in the cache.

    :raise huggingface_hub.errors.OfflineModeIsEnabled: If global offline mode is enabled
        for ``huggingface_hub`` and the file is not found in the cache.


    :param model_path: proposed model path
    :param local_files_only: If ``True``, do not attempt to download files, only check cache.
    :return: path to downloaded file or unchanged model path.
    """
    if _webcache.is_downloadable_url(model_path):
        return webcache_or_hf_blob_download(
            model_path,
            mime_acceptable_desc='not text',
            mimetype_is_supported=lambda m: m is not None and not m.startswith('text/'),
            unknown_mimetype_exception=NonHFModelDownloadError,
            local_files_only=local_files_only
        )

    return model_path


def download_non_hf_slug_config(path: str, local_files_only: bool = False):
    """
    Check for a non hugging face slug or reference to a config
    file that is possibly downloadable as a single file.

    If this is applicable, download it to the web cache and return its path.
    If the file already exists in the web cache simply return a path to it.

    Hugging Face blob links are also supported, in which case the file will
    be downloaded to the huggingface cache.

    If this is not applicable, return the path unchanged.

    TQDM progress bar is used for any download that occurs.

    :raise NonHFConfigDownloadError: If the download mimetype is not ``text/*`` or ``application/*``.

    :raise dgenerate.webcache.WebFileCacheOfflineModeException: If ``local_files_only`` is ``True`` and a
        download is required for a non Hugging Face blob link. This will occur if the file in question
        is not found in the dgenerate web cache. This can also occur if the :py:mod:`dgenerate.webcache``
        module is in global offline mode.

    :raise huggingface_hub.errors.HFValidationError: If the Hugging Face blob link is invalid.

    :raise huggingface_hub.errors.HfHubHTTPError: If the Hugging Face blob link is valid
        but the file could not be downloaded. This can also occur if ``local_files_only`` is ``True``
        and the file is not found in the cache.

    :raise huggingface_hub.errors.OfflineModeIsEnabled: If global offline mode is enabled
        for ``huggingface_hub`` and the file is not found in the cache.

    :param path: proposed model path
    :param local_files_only: If ``True``, do not attempt to download files, only check cache.
    :return: path to downloaded file or unchanged model path.

    """
    if _webcache.is_downloadable_url(path):
        return webcache_or_hf_blob_download(
            path,
            mime_acceptable_desc='text / yaml / json',
            mimetype_is_supported=lambda m: m.startswith('text/') or m.startswith('application/'),
            unknown_mimetype_exception=NonHFConfigDownloadError,
            local_files_only=local_files_only
        )

    return path


@contextlib.contextmanager
def with_hf_errors_as_model_not_found(catch_all: typing.Callable[[Exception], None] = None):
    """
    Context manager that catches Hugging Face hub errors
    associated with missing models or invalid model name specification
    and raises a :py:class:`dgenerate.exceptions.ModelNotFoundError` exception.

    :param catch_all: Optional callable to catch and handle all other exceptions.

    :raise dgenerate.exceptions.ModelNotFoundError: If a Hugging Face hub error occurs
    """

    if catch_all is None:
        try:
            yield
        except (huggingface_hub.errors.HFValidationError,
                huggingface_hub.errors.HfHubHTTPError,
                huggingface_hub.errors.OfflineModeIsEnabled) as e:
            raise dgenerate.exceptions.ModelNotFoundError(e) from e
    else:
        try:
            yield
        except (huggingface_hub.errors.HFValidationError,
                huggingface_hub.errors.HfHubHTTPError,
                huggingface_hub.errors.OfflineModeIsEnabled) as e:
            raise dgenerate.exceptions.ModelNotFoundError(e) from e
        except Exception as e:
            catch_all(e)


@contextlib.contextmanager
def with_hf_errors_as_config_not_found(catch_all: typing.Callable[[Exception], None] = None):
    """
    Context manager that catches Hugging Face hub errors
    associated with missing models or invalid model name specification
    and raises a :py:class:`dgenerate.exceptions.ConfigNotFoundError` exception.

    :param catch_all: Optional callable to catch and handle all other exceptions.

    :raise dgenerate.exceptions.ConfigNotFoundError: If a Hugging Face hub error occurs
    """

    if catch_all is None:
        try:
            yield
        except (huggingface_hub.errors.HFValidationError,
                huggingface_hub.errors.HfHubHTTPError,
                huggingface_hub.errors.OfflineModeIsEnabled) as e:
            raise dgenerate.exceptions.ConfigNotFoundError(e) from e
    else:
        try:
            yield
        except (huggingface_hub.errors.HFValidationError,
                huggingface_hub.errors.HfHubHTTPError,
                huggingface_hub.errors.OfflineModeIsEnabled) as e:
            raise dgenerate.exceptions.ConfigNotFoundError(e) from e
        except Exception as e:
            catch_all(e)


def is_offline_mode() -> bool:
    """
    Check if the global offline mode for ``huggingface_hub`` is enabled.

    :return: `True` if offline mode is enabled, `False` otherwise.
    """
    return huggingface_hub.constants.HF_HUB_OFFLINE


def enable_offline_mode():
    """
    Enable global offline mode for ``huggingface_hub``.

    This will prevent any network requests from being made, and will only use files
    that are already in the hub cache.
    """
    huggingface_hub.constants.HF_HUB_OFFLINE = True


def disable_offline_mode():
    """
    Disable global offline mode for ``huggingface_hub``.

    This will allow network requests to the hub to be made again.
    """
    huggingface_hub.constants.HF_HUB_OFFLINE = False


@contextlib.contextmanager
def offline_mode_context(enabled=True):
    """
    Context manager to temporarily enable or disable global offline mode for ``huggingface_hub``.

    :param enabled: If `True`, enables offline mode. If `False`, disables it.
    """
    original_mode = huggingface_hub.constants.HF_HUB_OFFLINE
    huggingface_hub.constants.HF_HUB_OFFLINE = enabled
    try:
        yield
    finally:
        huggingface_hub.constants.HF_HUB_OFFLINE = original_mode



__all__ = _types.module_all()
