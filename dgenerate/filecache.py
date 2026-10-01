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

import datetime
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
import typing
import urllib.parse
import uuid

import fake_useragent
import filelock
import pyrfc6266
import requests
import tqdm

import dgenerate.memory as _memory
import dgenerate.messages as _messages

__doc__ = """
On disk file cache implementation and primitives.
"""

# (connect, read) in seconds. A single number would also limit silence between
# socket reads. Large CDN downloads stall for longer than a few seconds.
_DOWNLOAD_TIMEOUT = (10, 60)
_DOWNLOAD_ATTEMPTS = 5
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_DOWNLOAD_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _download_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or 'download'


def _download_error_text(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    message = text[0] if text else exc.__class__.__name__
    message = re.sub(r'\?[^)\s]*', '', message)
    if len(message) > 180:
        message = message[:177] + '...'
    return message


def _format_byte_size(num: int) -> str:
    value = float(num)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            if unit == 'B':
                return f'{int(value)} {unit}'
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{int(num)} B'


def _header_int(value) -> int:
    if value is None or value == '':
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_content_range(header) -> tuple[int | None, int | None]:
    """Return ``(start, total)`` from a Content-Range header."""
    if not header:
        return None, None
    try:
        _unit, spec = str(header).split(' ', 1)
        range_part, total_part = spec.split('/', 1)
        total = None if total_part.strip() == '*' else int(total_part)
        if range_part.strip() == '*':
            return None, total
        start_text, _end = range_part.split('-', 1)
        return int(start_text), total
    except (ValueError, AttributeError):
        return None, None


def _download_chunk_size(total: int) -> int:
    chunk_size = _memory.calculate_chunk_size(total)
    if chunk_size <= 0:
        return 1024 * 1024
    return chunk_size


def download_url_to_file(
        url: str,
        path: str | None = None,
        *,
        headers: dict | None = None,
        on_response: typing.Callable[[requests.Response], None] | None = None,
        resolve_path: typing.Callable[[requests.Response], str] | None = None,
        tqdm_pbar=tqdm.tqdm,
        timeout: tuple[float, float] = _DOWNLOAD_TIMEOUT,
        attempts: int = _DOWNLOAD_ATTEMPTS,
        resume: bool = True) -> str:
    """
    Stream ``url`` to ``path``.

    ``timeout`` is ``(connect_seconds, read_seconds)``. The read timeout is the
    maximum silence between socket reads, not a limit on the whole transfer.

    An existing partial file is continued with an HTTP Range request. Connection
    failures and short reads are retried until ``attempts`` is exhausted. The
    partial file is left on disk so a later call can continue it.

    :param url: File URL.
    :param path: Destination path. Required when ``resolve_path`` is not given.
    :param headers: Extra request headers. A Range header is added when resuming.
    :param on_response: Called once with the response that will be written,
        before any bytes are written. Raise to reject the response.
    :param resolve_path: Called with the first response when ``path`` is omitted.
        Return the destination path.
    :param tqdm_pbar: tqdm progress bar type. ``None`` disables the bar.
    :param timeout: ``(connect, read)`` timeouts in seconds.
    :param attempts: How many times to try the transfer.
    :param resume: Continue a partial destination file when the server allows it.
    :return: Destination path.
    """
    if path is None and resolve_path is None:
        raise ValueError('path or resolve_path is required.')

    if path is not None and resume and os.path.isfile(path):
        downloaded = os.path.getsize(path)
    else:
        downloaded = 0

    if downloaded:
        _messages.log(
            f'Resuming download from {_format_byte_size(downloaded)} '
            f'({_download_host(url)}).')

    progress = None
    headers_accepted = False
    host = _download_host(url)

    try:
        attempt = 1
        while attempt <= attempts:
            req_headers = dict(headers or {})
            if downloaded and path is not None:
                req_headers['Range'] = f'bytes={downloaded}-'
            try:
                with requests.get(
                        url,
                        headers=req_headers,
                        stream=True,
                        timeout=timeout) as response:
                    if response.status_code == 416:
                        _start, total = _parse_content_range(
                            response.headers.get('Content-Range'))
                        if total is None and downloaded:
                            try:
                                head = requests.head(
                                    url,
                                    headers=dict(headers or {}),
                                    timeout=timeout,
                                    allow_redirects=True)
                                total = _header_int(head.headers.get('Content-Length'))
                            except requests.RequestException:
                                total = None
                        if downloaded and total is not None and downloaded == total:
                            return path
                        if (downloaded and total is not None and downloaded > total
                                and path and os.path.isfile(path)):
                            os.remove(path)
                            downloaded = 0
                            if attempt >= attempts:
                                response.raise_for_status()
                            raise requests.ConnectionError(
                                f'Partial download from {host} was longer than the remote file.')
                        response.raise_for_status()

                    if response.status_code in _RETRYABLE_STATUS:
                        if attempt >= attempts:
                            response.raise_for_status()
                        raise requests.ConnectionError(f'HTTP {response.status_code} from {host}')

                    response.raise_for_status()

                    if path is None:
                        path = resolve_path(response)
                        if resume and response.status_code == 200 and os.path.isfile(path):
                            existing = os.path.getsize(path)
                            if existing:
                                downloaded = existing
                                _messages.log(
                                    f'Resuming download from {_format_byte_size(downloaded)} '
                                    f'({host}).')
                                continue

                    if on_response is not None and not headers_accepted:
                        on_response(response)
                        headers_accepted = True

                    if response.status_code == 206:
                        start, ranged_total = _parse_content_range(
                            response.headers.get('Content-Range'))
                        if start is not None and start != downloaded:
                            if start == 0:
                                downloaded = 0
                            else:
                                raise requests.ConnectionError(
                                    f'Unexpected resume offset {start} from {host}, '
                                    f'expected {downloaded}.')
                        total = ranged_total or 0
                        if not total:
                            remaining = _header_int(response.headers.get('Content-Length'))
                            total = downloaded + remaining if remaining else 0
                    else:
                        downloaded = 0
                        total = _header_int(response.headers.get('Content-Length'))

                    content_encoding = response.headers.get('Content-Encoding')
                    verify_length = (
                        not content_encoding or content_encoding.lower() == 'identity')
                    expected = total if verify_length else 0
                    chunk_size = _download_chunk_size(total)

                    if tqdm_pbar is not None and progress is None:
                        progress = tqdm_pbar(
                            total=total if total else None,
                            initial=downloaded,
                            unit='iB',
                            unit_scale=True)
                    elif progress is not None:
                        progress.total = total if total else None
                        progress.n = downloaded
                        progress.refresh()

                    with open(path, 'ab' if downloaded else 'wb') as handle:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            handle.flush()
                            downloaded += len(chunk)
                            if progress is not None:
                                progress.update(len(chunk))

                    if expected and downloaded != expected:
                        raise requests.ConnectionError(
                            f'Download from {host} ended early '
                            f'({_format_byte_size(downloaded)} of '
                            f'{_format_byte_size(expected)}).')
                    return path
            except _RETRYABLE_DOWNLOAD_ERRORS as exc:
                if path and os.path.isfile(path) and resume:
                    downloaded = os.path.getsize(path)
                else:
                    downloaded = 0
                if attempt >= attempts:
                    raise
                _messages.log(
                    f'Download from {host} interrupted ({_download_error_text(exc)}). '
                    + (f'Resuming from {_format_byte_size(downloaded)} '
                       if downloaded else 'Retrying ')
                    + f'(attempt {attempt + 1} of {attempts}).')
                time.sleep(min(2 ** (attempt - 1), 8))
                attempt += 1
        raise requests.ConnectionError(f'Download from {host} failed.')
    finally:
        if progress is not None:
            progress.close()


class WebFileCacheOfflineModeException(Exception):
    """
    Exception raised when the web cache is in offline mode and a file is not found in the cache.
    """
    pass


class KeyValueStore:
    """
    A key-value store using SQLite3 for storage.
    """

    def __init__(self, db_path: str):
        """
        Initialize the key-value store.

        :param db_path: The path to the SQLite3 database file.
        """
        db_dir = pathlib.Path(db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.file_lock = filelock.FileLock(db_path + ".lock")
        self._lock_counter = 0

    def __enter__(self):
        """
        Enter a context managed by this key-value store.

        :return: This key-value store.
        """
        if self._lock_counter == 0:
            self.file_lock.acquire()
            try:
                self.connection = sqlite3.connect(self.db_path)
                self.cursor = self.connection.cursor()
                self.cursor.execute(
                    "CREATE TABLE IF NOT EXISTS store (key TEXT PRIMARY KEY, value TEXT, creation_date TIMESTAMP)")
            except Exception:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                    self.cursor = None
                self.file_lock.release()
                raise

        self._lock_counter += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit a context managed by this key-value store.
        """
        self._lock_counter -= 1
        if self._lock_counter == 0:
            self.connection.commit()
            self.connection.close()
            self.file_lock.release()

    def get(self, key: str, default=None):
        """
        Get the value associated with a key.

        :param key: The key to get the value for.
        :param default: The default value to return if the key is not found.
        :return: The value associated with the key, or the default value if the key is not found.
        """
        with self:
            self.cursor.execute("SELECT value FROM store WHERE key=?", (key,))
            result = self.cursor.fetchone()
            if result is None:
                return default
            return result[0]

    def __getitem__(self, key: str) -> str:
        """
        Get the value associated with a key.

        :param key: The key to get the value for.
        :return: The value associated with the key.
        :raises KeyError: If the key is not found.
        """
        with self:
            self.cursor.execute("SELECT value FROM store WHERE key=?", (key,))
            result = self.cursor.fetchone()
            if result is None:
                raise KeyError(key)
            return result[0]

    def __setitem__(self, key: str, value: str):
        """
        Set the value for a key.

        :param key: The key to set the value for.
        :param value: The value to set.
        """
        with self:
            creation_date = datetime.datetime.now()
            self.cursor.execute("REPLACE INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                                (key, value, creation_date))

    def __delitem__(self, key: str):
        """
        Delete a key and its associated value.

        :param key: The key to delete.
        :raises KeyError: If the key is not found.
        """
        with self:
            if key not in self:
                raise KeyError(key)
            self.cursor.execute("DELETE FROM store WHERE key=?", (key,))

    def __contains__(self, key: str) -> bool:
        """
        Check if a key is in the store.

        :param key: The key to check.
        :return: ``True`` if the key is in the store, ``False`` otherwise.
        """
        with self:
            self.cursor.execute("SELECT 1 FROM store WHERE key=?", (key,))
            return self.cursor.fetchone() is not None

    def __iter__(self) -> typing.Iterator[str]:
        """
        Iterate over the keys and values in the store.

        :return: An iterator over the keys and values in the store.
        """
        with self:
            self.cursor.execute("SELECT key, value FROM store")
            for row in self.cursor:
                yield row

    def keys(self) -> typing.Iterator[str]:
        """
        Get all keys in the store.

        :return: An iterator over the keys in the store.
        """
        with self:
            self.cursor.execute("SELECT key FROM store")
            for row in self.cursor:
                yield row[0]

    def items(self) -> typing.Iterator[str]:
        """
        Get all values in the store.

        :return: An iterator over the values in the store.
        """
        with self:
            self.cursor.execute("SELECT value FROM store")
            for row in self.cursor:
                yield row[0]

    def delete_older_than(self, timedelta: datetime.timedelta) -> list[tuple[str, str]]:
        """
        Delete all keys and their associated values that were created more than a certain time ago.

        :param timedelta: The age of the keys to delete.
        :return: The keys and values that were deleted.
        """
        with self:
            try:
                cutoff_date = datetime.datetime.now() - timedelta
            except OverflowError:
                cutoff_date = datetime.datetime.min

            self.cursor.execute("SELECT key, value FROM store WHERE creation_date < ?", (cutoff_date,))
            deleted_rows = self.cursor.fetchall()
            self.cursor.execute("DELETE FROM store WHERE creation_date < ?", (cutoff_date,))
            return deleted_rows


class CachedFile:
    """Represents the path of a file in a :py:class:`.FileCache`"""

    path: str
    """
    The path to the file on disk.
    """

    metadata: dict[str, str]
    """
    Optional metadata for the file stored in the database.
    """

    def __init__(self, data_dict):
        """
        :param data_dict: file data dict parsed from the cache database.
        """
        self.path = data_dict['path']
        self.metadata = data_dict['metadata']


class FileCache:
    """
    A cache system that stores files and their metadata.
    """

    def __init__(self, db_path: str, cache_dir: str):
        """
        Initializes the :py:class:`.FileCache` object with a key-value store located
        at ``db_path`` and a cache directory at ``cache_dir``. If the cache directory
        doesn't exist, it creates it.

        :param db_path: The path to the key-value store database.
        :param cache_dir: The directory where the cache files are stored.
        """
        self.kv_store = KeyValueStore(db_path)
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def __iter__(self) -> typing.Iterator[CachedFile]:
        """
        Allows iteration over the key-value pairs in the key-value store,
        yielding each key and its corresponding :py:class:`.CachedFile` object.
        """
        with self.kv_store:
            for k, v in self.kv_store:
                yield k, CachedFile(json.loads(v))

    def __delitem__(self, key):
        """
        Deletes the item with the specified key from the key-value store.

        This also deletes the associated file in the cache.
        """
        with self.kv_store:
            if key not in self.kv_store:
                raise KeyError(key)

            file = CachedFile(json.loads(self.kv_store[key]))

            try:
                os.unlink(file.path)
            except OSError:
                pass

            del self.kv_store[key]

    def items(self) -> typing.Iterator[CachedFile]:
        """
        Yields all items in the key-value store as :py:class:`.CachedFile` objects.
        """
        with self.kv_store:
            for k, v in self.kv_store:
                yield CachedFile(json.loads(v))

    def keys(self) -> typing.Iterator[str]:
        """
        Yields all keys in the key-value store.
        """
        with self.kv_store:
            for k in self.kv_store.keys():
                yield k

    def _generate_unique_filename(self, ext):
        """
        Generates a unique filename with the specified extension in the cache directory.
        """
        if ext is None or not ext.strip():
            ext = ''
        else:
            ext = '.' + ext.lstrip('.')
        while True:
            file_path = os.path.join(self.cache_dir, str(uuid.uuid4())) + ext
            if not os.path.exists(file_path):
                break
        return file_path

    def __enter__(self):
        """
        Allows the :py:class:`.FileCache` object to be used in a with statement,
        ensuring that the key-value store is properly opened.
        """
        self.kv_store.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures that the key-value store is properly closed after being used in a with statement.
        """
        self.kv_store.__exit__(exc_type, exc_val, exc_tb)

    def delete_older_than(self, timedelta: datetime.timedelta) -> typing.Iterator[CachedFile]:
        """
        Deletes items from the key-value store that are older than the specified timedelta,
        yielding each key and its corresponding :py:class:`.CachedFile` object.
        """
        for key, value in self.kv_store.delete_older_than(timedelta):
            yield key, CachedFile(json.loads(value))

    def add(self,
            key: str,
            file_data: bytes | typing.Iterable[bytes],
            metadata: typing.Dict[str, str] = None,
            ext: str | None = None) \
            -> CachedFile:
        """
        Adds a file to the cache. If a file with the same key already exists, it overwrites the existing file.
        Otherwise, it creates a new file with a unique filename.

        :param key: The key associated with the file.
        :param file_data: The data of the file in bytes, or an iterable of binary chunks.
        :param metadata: The metadata of the file.
        :param ext: The extension of the file.
        :return: A :py:class:`.CachedFile` object representing the added file.
        """
        with self.kv_store as kv:
            if key in kv:
                file_path = json.loads(kv.get(key))['path']
            else:
                file_path = self._generate_unique_filename(ext)

        if isinstance(file_data, bytes):
            with open(file_path, 'wb') as f:
                f.write(file_data)
                f.flush()
        else:
            with open(file_path, 'wb') as f:
                iterable = iter(file_data)
                for chunk in iterable:
                    f.write(chunk)
                    f.flush()

        with self.kv_store as kv:
            entry_data = {'path': file_path,
                          'metadata': metadata}
            kv[key] = json.dumps(entry_data)

        return CachedFile(entry_data)

    def get(self, key) -> CachedFile | None:
        """
        Retrieves the :py:class:`.CachedFile` object for the specified key
        from the  key-value store, or returns None if the key does not exist.

        :param key: The key associated with the file.
        :return: A :py:class:`.CachedFile` object representing the file, or ``None`` if the key does not exist.
        """
        with self.kv_store as kv:
            if key in kv:
                return CachedFile(json.loads(kv[key]))
            else:
                return None


class WebFileCache(FileCache):
    """
    A cache system that stores files and their metadata downloaded from the web.
    """

    def __init__(self,
                 db_path: str,
                 cache_dir: str,
                 expiry_delta: datetime.timedelta = datetime.timedelta(hours=12)):
        """
        Initializes the :py:class:`.WebFileCache` object with a key-value store
        located at ``db_path``, a cache directory at ``cache_dir``, and an expiry delta.
        If the cache directory doesn't exist, it creates it. It also attempts to clear old files.

        :param db_path: The path to the key-value store database.
        :param cache_dir: The directory where the cache files are stored.
        :param expiry_delta: The time delta for file expiry.
        """
        super().__init__(db_path, cache_dir)
        self.expiry_delta = expiry_delta
        self._local_files_only = False
        try:
            self._clear_old_files()
        except sqlite3.Error:
            self._remove_cache_files_except_locks()

    @property
    def local_files_only(self) -> bool:
        """
        Get the local_files_only mode status.

        :return: ``True`` if local_files_only mode is enabled, ``False`` otherwise.
        """
        return self._local_files_only

    @local_files_only.setter
    def local_files_only(self, value: bool):
        """
        Set the local_files_only mode status.

        :param value: ``True`` to enable local_files_only mode, ``False`` to disable it.
        """
        self._local_files_only = value

    def _remove_cache_files_except_locks(self):
        """
        Removes all cache files except for lock files.
        """
        with self.kv_store.file_lock:
            os.unlink(self.kv_store.db_path)
            stack = [self.cache_dir]
            while stack:
                base = stack.pop()
                for entry in os.scandir(base):
                    if entry.is_file():
                        if not entry.name.endswith('.lock'):
                            os.remove(entry.path)
                    elif entry.is_dir():
                        stack.append(entry.path)
                if not os.listdir(base):
                    os.rmdir(base)

    def _partial_download_path(self, url: str) -> str:
        digest = hashlib.sha256(url.encode('utf-8')).hexdigest()
        directory = os.path.join(self.cache_dir, '.partials')
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, digest)

    def _read_partial_state(self, partial_path: str) -> dict:
        meta_path = partial_path + '.meta'
        if not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write_partial_state(self, partial_path: str, state: dict):
        with open(partial_path + '.meta', 'w', encoding='utf-8') as handle:
            json.dump(state, handle)

    def _clear_old_partials(self):
        partial_dir = os.path.join(self.cache_dir, '.partials')
        if not os.path.isdir(partial_dir):
            return
        cutoff = time.time() - self.expiry_delta.total_seconds()
        stale = []
        for entry in os.scandir(partial_dir):
            if not entry.is_file():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    stale.append(entry.path)
            except OSError:
                pass
        for stale_path in stale:
            try:
                os.remove(stale_path)
            except OSError:
                pass

    def _store_downloaded_file(self, key: str, source_path: str,
                               metadata: dict, ext: str | None) -> CachedFile:
        with self.kv_store as kv:
            if key in kv:
                file_path = json.loads(kv.get(key))['path']
            else:
                file_path = self._generate_unique_filename(ext)

        os.replace(source_path, file_path)
        meta_path = source_path + '.meta'
        if os.path.isfile(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass

        with self.kv_store as kv:
            entry_data = {'path': file_path, 'metadata': metadata}
            kv[key] = json.dumps(entry_data)
        return CachedFile(entry_data)

    def _clear_old_files(self):
        """
        Clears files that are older than the expiry delta.
        """
        for key, cached_file in self.delete_older_than(self.expiry_delta):
            try:
                os.unlink(cached_file.path)
            except FileNotFoundError:
                pass
        self._clear_old_partials()

    def request_mimetype(self, url, local_files_only: bool = False) -> str:
        """
        Requests the mimetype of a file at a URL. If the file exists in the cache, a known mimetype
        is returned without connecting to the internet. Otherwise, it connects to the internet
        to retrieve the mimetype. This action does not update the cache.

        :raise HTTPError: On http status errors.
        :raise WebFileCacheOfflineModeException: If local_files_only mode is enabled and the file is not found in the cache.

        :param url: The URL of the file.
        :param local_files_only: If ``True``, do not make a request, only check the cache.
        :return: The mimetype of the file.
        """
        with self:
            exists = self.get(url)
            if exists is not None:
                return exists.metadata['mime-type']

        # Check if we're in offline mode (either from property or parameter)
        if self._local_files_only or local_files_only:
            raise WebFileCacheOfflineModeException(
                f'Web cache is in offline mode, and the '
                f'file for "{url}" was not found in the local cache.'
            )

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}

        with requests.get(url, headers=headers, stream=True,
                          timeout=_DOWNLOAD_TIMEOUT) as req:
            req.raise_for_status()
            mime_type = req.headers['content-type']

        return mime_type

    @staticmethod
    def is_downloadable_url(string) -> bool:
        """
        Does a string represent a URL that can be downloaded by this web cache implementation?

        :param string: the string
        :return: ``True`` or ``False``
        """
        return string.startswith('http://') or string.startswith('https://')

    def download(self, url,
                 mime_acceptable_desc: str | None = None,
                 mimetype_is_supported: typing.Callable[[str], bool] | None = None,
                 unknown_mimetype_exception=ValueError,
                 overwrite: bool = False,
                 tqdm_pbar=tqdm.tqdm,
                 local_files_only: bool = False) -> CachedFile:
        """
        Downloads a file and/or returns a file path from the cache. If the mimetype
        of the file is not supported, it raises an exception.

        Interrupted downloads are kept and continued on the next call. The read
        timeout limits silence between socket reads, not the full transfer.

        :raise requests.RequestException: Can raise any exception
            raised by ``requests.get`` for request related errors.
        :raise WebFileCacheOfflineModeException: If local_files_only mode is enabled and the file is not found in the cache.

        :param url: The URL of the file.
        :param mime_acceptable_desc: A description of acceptable mimetypes for use in exceptions.
        :param mimetype_is_supported: A function that determines if a mimetype is supported for downloading.
        :param unknown_mimetype_exception: The exception type to raise when an unknown mimetype is encountered.
        :param overwrite: Always overwrite any previously cached file?
        :param tqdm_pbar: tqdm progress bar type, if set to `None` no progress bar will be used. Defaults to `tqdm.tqdm`
        :param local_files_only: If ``True``, do not attempt to download files, only check cache.
        :return: The path to the downloaded file.
        """

        self._clear_old_files()

        def _mimetype_is_supported(mimetype):
            if mimetype_is_supported is not None:
                return mimetype_is_supported(mimetype)
            return True

        # Check if we're in offline mode (either from property or parameter)
        is_offline = self._local_files_only or local_files_only

        if not overwrite:
            with self:
                cached_file = self.get(url)
                if cached_file is not None and os.path.exists(cached_file.path):
                    return cached_file

        # If we're in offline mode and the file wasn't found in cache, raise an exception
        if is_offline:
            raise WebFileCacheOfflineModeException(
                f'Web cache is in offline mode, and the '
                f'file for "{url}" was not found in the local cache.'
            )

        partial_path = self._partial_download_path(url)
        state = self._read_partial_state(partial_path)

        def on_response(response):
            mime_type = response.headers.get('content-type', 'unknown')

            if not _mimetype_is_supported(mime_type):
                raise unknown_mimetype_exception(
                    f'Unknown mimetype "{mime_type}" from URL "{url}". '
                    f'Expected: {mime_acceptable_desc}')

            filename = pyrfc6266.requests_response_to_filename(response)
            _, ext = os.path.splitext(filename)
            state['metadata'] = {'mime-type': mime_type}
            state['ext'] = ext
            self._write_partial_state(partial_path, {
                'metadata': state['metadata'],
                'ext': ext,
            })

        if tqdm_pbar is not None:
            _messages.log(f'Downloading: "{url}"', underline=True)

        download_url_to_file(
            url,
            partial_path,
            headers={'User-Agent': fake_useragent.UserAgent().chrome},
            on_response=on_response,
            tqdm_pbar=tqdm_pbar)

        if 'metadata' not in state:
            saved = self._read_partial_state(partial_path)
            state['metadata'] = saved.get('metadata') or {'mime-type': 'unknown'}
            state['ext'] = saved.get('ext')

        return self._store_downloaded_file(
            url, partial_path, state['metadata'], state.get('ext'))
