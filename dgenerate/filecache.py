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
import urllib.parse
import uuid

import fake_useragent
import filelock
import requests


class KeyValueStore:
    def __init__(self, db_path):
        db_dir = pathlib.Path(db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.file_lock = filelock.FileLock(db_path + ".lock")
        self._lock_counter = 0

    def __enter__(self):
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
        self._lock_counter -= 1
        if self._lock_counter == 0:
            self.connection.commit()
            self.connection.close()
            self.file_lock.release()

    def get(self, key, default=None):
        with self:
            self.cursor.execute("SELECT value FROM store WHERE key=?", (key,))
            result = self.cursor.fetchone()
            if result is None:
                return default
            return result[0]

    def __getitem__(self, key):
        with self:
            self.cursor.execute("SELECT value FROM store WHERE key=?", (key,))
            result = self.cursor.fetchone()
            if result is None:
                raise KeyError(key)
            return result[0]

    def __setitem__(self, key, value):
        with self:
            creation_date = datetime.datetime.now()
            self.cursor.execute("REPLACE INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                                (key, value, creation_date))

    def __delitem__(self, key):
        with self:
            if key not in self:
                raise KeyError(key)
            self.cursor.execute("DELETE FROM store WHERE key=?", (key,))

    def __contains__(self, key):
        with self:
            self.cursor.execute("SELECT 1 FROM store WHERE key=?", (key,))
            return self.cursor.fetchone() is not None

    def __iter__(self):
        with self:
            self.cursor.execute("SELECT key, value FROM store")
            for row in self.cursor:
                yield row

    def keys(self):
        with self:
            self.cursor.execute("SELECT key FROM store")
            for row in self.cursor:
                yield row[0]

    def items(self):
        with self:
            self.cursor.execute("SELECT value FROM store")
            for row in self.cursor:
                yield row[0]

    def delete_older_than(self, timedelta):
        with self:
            cutoff_date = datetime.datetime.now() - timedelta
            self.cursor.execute("SELECT key, value FROM store WHERE creation_date < ?", (cutoff_date,))
            deleted_rows = self.cursor.fetchall()
            self.cursor.execute("DELETE FROM store WHERE creation_date < ?", (cutoff_date,))
            return deleted_rows


class CachedFile:
    def __init__(self, data_dict):
        self.path = data_dict['path']
        self.metadata = data_dict['metadata']


class FileCache:
    def __init__(self, db_path, cache_dir):
        self.kv_store = KeyValueStore(db_path)
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def __iter__(self):
        with self.kv_store:
            for k, v in self.kv_store:
                yield k, CachedFile(json.loads(v))

    def __delitem__(self, key):
        with self.kv_store:
            del self.kv_store[key]

    def items(self):
        with self.kv_store:
            for k, v in self.kv_store:
                yield CachedFile(json.loads(v))

    def keys(self):
        with self.kv_store:
            for k in self.kv_store.keys():
                yield k

    def _generate_unique_filename(self, ext):
        if ext is None:
            ext = ''
        else:
            ext = '.' + ext.lstrip('.')
        while True:
            file_path = os.path.join(self.cache_dir, str(uuid.uuid4())) + ext
            if not os.path.exists(file_path):
                break
        return file_path

    def __enter__(self):
        self.kv_store.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kv_store.__exit__(exc_type, exc_val, exc_tb)

    def delete_older_than(self, timedelta):
        for key, value in self.kv_store.delete_older_than(timedelta):
            yield key, CachedFile(json.loads(value))

    def add(self, key, file_data: bytes, metadata: typing.Dict[str, str] = None, ext=None) -> typing.Optional[CachedFile]:

        with self.kv_store as kv:
            if key in kv:
                file_path = pathlib.Path(json.loads(kv.get(key))['path']).name
            else:
                file_path = self._generate_unique_filename(ext)

        with open(file_path, 'wb') as f:
            f.write(file_data)
            f.flush()

        with self.kv_store as kv:
            entry_data = {'path': file_path, 'metadata': metadata}
            kv[key] = json.dumps(entry_data)

        return CachedFile(entry_data)

    def get(self, key) -> typing.Optional[CachedFile]:
        with self.kv_store as kv:
            if key in kv:
                return CachedFile(json.loads(kv[key]))
            else:
                return None


class WebFileCache(FileCache):
    def __init__(self, db_path, cache_dir, expiry_delta=datetime.timedelta(hours=12)):
        super().__init__(db_path, cache_dir)
        self.expiry_delta = expiry_delta
        try:
            self._clear_old_files()
        except sqlite3.Error:
            self._remove_cache_files_except_locks()

    def _remove_cache_files_except_locks(self):
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
        for key, cached_file in self.delete_older_than(self.expiry_delta):
            os.unlink(cached_file.path)

    def request_mimetype(self, url) -> str:
        """
        Request the mimetype of a file at a URL, if the file exists in the cache, a known mimetype
        is returned without connecting to the internet. Otherwise connect to the internet
        to retrieve the mimetype, this action does not update the cache.

        :param url: The url

        :return: mimetype string
        """
        with self:
            exists = self.get(url)
            if exists is not None:
                return exists.metadata['mime-type']

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}

        with requests.get(url, headers=headers, stream=True) as req:
            mime_type = req.headers['content-type']

        return mime_type

    def download(self, url,
                 mime_acceptable_desc: typing.Optional[str] = None,
                 mimetype_is_supported: typing.Optional[typing.Callable[[str], bool]] = None,
                 unknown_mimetype_exception=ValueError):
        self._clear_old_files()

        def _mimetype_is_supported(mimetype):
            if mimetype_is_supported is not None:
                return mimetype_is_supported(mimetype)
            return True

        with self:
            cached_file = self.get(url)
            if cached_file is not None and os.path.exists(cached_file.path):
                return cached_file

        response = requests.get(url, headers={'User-Agent': fake_useragent.UserAgent().chrome}, stream=True)
        response.raise_for_status()

        mime_type = response.headers.get('content-type', 'unknown')

        if not _mimetype_is_supported(mime_type):
            raise unknown_mimetype_exception(
                f'Unknown mimetype "{mime_type}" from URL "{url}". '
                f'Expected: {mime_acceptable_desc}')

        metadata = {'mime-type': mime_type}

        parsed = urllib.parse.urlparse(url)
        path = os.path.splitext(parsed.path)
        if len(path) > 1:
            ext = path[1]
        else:
            ext = ''

        return self.add(url, response.content, metadata, ext)
