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
import json
import os
import pathlib
import sqlite3
import typing
import uuid
import re

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

    def _clear_old_files(self):
        """
        Clears files that are older than the expiry delta.
        """
        for key, cached_file in self.delete_older_than(self.expiry_delta):
            try:
                os.unlink(cached_file.path)
            except FileNotFoundError:
                pass

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

        with requests.get(url, headers=headers, stream=True, timeout=5) as req:
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

        with requests.get(url,
                          headers={'User-Agent': fake_useragent.UserAgent().chrome},
                          stream=True,
                          timeout=5) as response:
            response.raise_for_status()

            mime_type = response.headers.get('content-type', 'unknown')

            if not _mimetype_is_supported(mime_type):
                raise unknown_mimetype_exception(
                    f'Unknown mimetype "{mime_type}" from URL "{url}". '
                    f'Expected: {mime_acceptable_desc}')

            metadata = {'mime-type': mime_type}

            filename = pyrfc6266.requests_response_to_filename(response)
            _, ext = os.path.splitext(filename)

            total_size = int(response.headers.get('content-length', 0))

            chunk_size = _memory.calculate_chunk_size(total_size)

            if tqdm_pbar is not None:
                _messages.log(f'Downloading: "{url}"', underline=True)

            if chunk_size != total_size:

                if tqdm_pbar is None:
                    def file_data_generator():
                        for chunk in response.iter_content(
                                chunk_size=chunk_size):
                            yield chunk
                else:
                    def file_data_generator():
                        with tqdm_pbar(total=total_size if total_size != 0 else None,
                                       unit='iB',
                                       unit_scale=True) as progress_bar:
                            for chunk in response.iter_content(
                                    chunk_size=chunk_size):
                                progress_bar.update(len(chunk))
                                yield chunk
            else:
                def file_data_generator():
                    yield response.content

            # Add the downloaded file to the cache
            return self.add(url, file_data_generator(),
                            metadata, ext)
