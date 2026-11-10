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
import functools
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


def _adapt_datetime(value: datetime.datetime) -> str:
    """
    Store a timestamp the way sqlite used to.

    The space separator matches the adapter sqlite3 is removing. Existing
    ``creation_date`` values compare as text, so a ``T`` separator would
    sort them differently.
    """
    return value.isoformat(' ')


def _adapt_date(value: datetime.date) -> str:
    return value.isoformat()


sqlite3.register_adapter(datetime.datetime, _adapt_datetime)
sqlite3.register_adapter(datetime.date, _adapt_date)

import dgenerate.filetypes as _filetypes
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing

__doc__ = """
On disk file cache implementation and primitives.
"""

# (connect, read) in seconds. A single number would also limit silence between
# socket reads. Large CDN downloads stall for longer than a few seconds.
_DOWNLOAD_TIMEOUT = (10, 60)
_DOWNLOAD_ATTEMPTS = 5

_DESKTOP_CHROME = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)


@functools.lru_cache(maxsize=1)
def _desktop_user_agent() -> fake_useragent.UserAgent:
    """Chrome on desktop OS only. Mobile agents often get 403 from Cloudflare."""
    return fake_useragent.UserAgent(
        browsers=['Chrome'],
        os=['Windows', 'Linux', 'Ubuntu', 'Mac OS X', 'Chrome OS'],
        platforms=['desktop'],
        fallback=_DESKTOP_CHROME,
    )


def user_agent() -> str:
    """A random desktop user agent string."""
    try:
        return _desktop_user_agent().chrome
    except Exception:
        return _DESKTOP_CHROME

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_DOWNLOAD_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _download_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or 'download'


def _url_for_warning(url: str) -> str:
    """Drop credential query parameters before a URL is written to the log."""
    parsed = urllib.parse.urlparse(url)
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {'token', 'access_token', 'auth_token'}
    ]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _basename_filename(name: str) -> str | None:
    filename = os.path.basename(str(name).replace('\\', '/')).strip().strip('"')
    if not filename or filename in {'.', '..'}:
        return None
    return filename


def _content_disposition_filename(response) -> str | None:
    """
    Basename from a ``Content-Disposition`` header.

    ``None`` when the response does not send one.
    """
    if not response.headers.get('content-disposition'):
        return None
    try:
        filename = pyrfc6266.requests_response_to_filename(response)
    except Exception:
        return None
    return _basename_filename(filename)


def _url_path_filename(url: str) -> str | None:
    """
    Last path segment when it is a file name.

    Uses the URL that was requested. A redirect to object storage replaces
    the name with a hash, so the response URL is not consulted. The segment
    must end in a suffix that contains a letter, such as ``.gguf`` or
    ``.7z``. ``/api/download/models/714161`` and ``.../v1.2`` do not qualify.
    """
    parsed = urllib.parse.urlparse(url)
    name = _basename_filename(urllib.parse.unquote(parsed.path).rstrip('/'))
    if name is None:
        return None
    suffix = os.path.splitext(name)[1]
    if not re.fullmatch(r'\.[A-Za-z0-9]{1,16}', suffix):
        return None
    if not any(character.isalpha() for character in suffix):
        return None
    return name


def _download_filename(url: str, response) -> str | None:
    """Published file name: the disposition header, or a file name in the URL path."""
    return _content_disposition_filename(response) or _url_path_filename(url)


def _warn_missing_disposition(url: str, reason: str) -> None:
    _messages.warning(
        f'Could not read the Content-Disposition filename for "{_url_for_warning(url)}": {reason}. '
        f'The name will be requested again the next time this file is used.')


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
                _register_cache_sql(self.connection)
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

    def delete_matching(self, where_sql: str, params: typing.Sequence) -> list[tuple[str, str]]:
        """
        Delete rows matched by a SQL predicate.

        The predicate sees ``key``, ``value``, ``creation_date``, ``eligible``,
        and ``ext``. ``eligible`` is 0 when ``value`` is not a cache entry.
        ``ext`` is the file extension stored in that entry.

        :param where_sql: SQL boolean expression.
        :param params: Values bound to ``where_sql``.
        :return: The keys and values that were deleted.
        """
        extension_sql = _cache_value_extension_sql()
        select_sql = f"""
            WITH keyed AS (
                SELECT key, value, creation_date,
                    CASE
                        WHEN value IS NULL THEN 0
                        WHEN json_valid(value) = 0 THEN 0
                        WHEN json_type(value) != 'object' THEN 0
                        WHEN json_type(value, '$.path') IS NULL THEN 0
                        WHEN json_type(value, '$.metadata') IS NULL THEN 0
                        ELSE 1
                    END AS eligible,
                    {extension_sql} AS ext
                FROM store
            )
            SELECT key, value FROM keyed WHERE {where_sql}
        """
        with self:
            self.cursor.execute(select_sql, params)
            deleted = self.cursor.fetchall()
            if deleted:
                self.cursor.executemany(
                    "DELETE FROM store WHERE key=?",
                    [(key,) for key, _ in deleted])
            return deleted


def _stored_datetime(value) -> datetime.datetime | None:
    if value is None or value == '':
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.datetime.min.time())
    try:
        return datetime.datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _normalize_extension(ext: str | None) -> str:
    if not ext:
        return ''
    text = str(ext).strip().lower()
    if text.startswith('*.'):
        text = text[2:]
    elif text.startswith('.'):
        text = text[1:]
    return text


def _extension_from(path: str | None = None,
                    metadata: dict | None = None,
                    ext: str | None = None) -> str:
    if ext:
        return ext
    if isinstance(metadata, dict):
        filename = metadata.get('filename') or ''
        found = os.path.splitext(str(filename))[1]
        if found:
            return found
    if path:
        return os.path.splitext(str(path))[1]
    return ''


# str.isspace() ordinals. SQLite trim() otherwise removes only the ASCII space,
# which is a smaller set than str.strip().
_SQL_SPACE = '||'.join(
    f'char({code})' for code in (
        0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0,
        0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
    ))


def _sql_lower(value):
    if value is None:
        return None
    return str(value).lower()


def _sql_created_before(created, cutoff) -> int:
    """
    Whether a stored timestamp is older than ``cutoff``.

    ``cutoff`` is null when the file is kept. An unreadable timestamp is kept.
    """
    if cutoff is None:
        return 0
    left = _stored_datetime(created)
    right = _stored_datetime(cutoff)
    if left is None or right is None:
        return 0
    return 1 if left < right else 0


def _register_cache_sql(connection: sqlite3.Connection):
    """Register the functions used by a compiled expiry delete."""
    connection.create_function('dgenerate_lower', 1, _sql_lower)
    connection.create_function('dgenerate_before', 2, _sql_created_before)


def _cache_name_extension_sql(source: str) -> str:
    """
    SQL for ``_normalize_extension(os.path.splitext(source)[1])``.

    The suffix comes from the last path component. A dot in an earlier
    directory does not give an extensionless file an extension. ``source``
    is a SQL expression.
    """
    norm = f"replace(ifnull({source}, ''), char(92), '/')"
    base = f"replace({norm}, rtrim({norm}, replace({norm}, '/', '')), '')"
    trimmed = f"ltrim({base}, '.')"
    suffix = f"replace({trimmed}, rtrim({trimmed}, replace({trimmed}, '.', '')), '')"
    dotted = f"trim('.' || {suffix}, {_SQL_SPACE})"
    folded = f"dgenerate_lower({dotted})"
    return (
        "CASE "
        f"WHEN instr({trimmed}, '.') = 0 THEN '' "
        f"WHEN substr({folded}, 1, 1) = '.' THEN substr({folded}, 2) "
        f"ELSE {folded} END"
    )


def _cache_value_extension_sql() -> str:
    """
    SQL for the extension of a cache ``value``.

    A metadata filename supplies the extension when it has one. Otherwise the
    stored path does. The result is empty when neither has an extension.
    """
    filename = _cache_name_extension_sql("json_extract(value, '$.metadata.filename')")
    path = _cache_name_extension_sql("json_extract(value, '$.path')")
    return (
        "CASE WHEN json_type(value, '$.metadata') = 'object' "
        f"THEN COALESCE(NULLIF({filename}, ''), {path}) "
        f"ELSE {path} END"
    )


def _expiry_cutoff(delta: datetime.timedelta | None,
                   now: datetime.datetime) -> datetime.datetime | None:
    if delta is None or delta == datetime.timedelta.max:
        return None
    try:
        return now - delta
    except OverflowError:
        return None


# Default DGENERATE_WEB_CACHE_EXPIRY_DELTA value.
# Model files are kept. Every other cached file expires after 12 hours.
DEFAULT_WEB_CACHE_EXPIRY = 'models;forever,other;hours=12'

# Default DGENERATE_WEB_CACHE_PARTIAL_EXPIRY_DELTA value.
# An interrupted download of a file kept forever expires after 12 hours.
DEFAULT_WEB_CACHE_PARTIAL_EXPIRY = 'hours=12'


class WebCacheExpiryParseError(_textprocessing.TimeDeltaParseError):
    """
    Raised when a web cache expiry expression cannot be parsed.

    This is a :py:class:`dgenerate.textprocessing.TimeDeltaParseError`, so callers
    that already handle timedelta parse failures handle rule-list failures as well.
    """
    pass


class WebCacheExpiry:
    """
    Expiry policy for :py:class:`WebFileCache`.

    A policy is either one interval for every file, or a list of rules.
    Intervals are parsed by :py:func:`dgenerate.textprocessing.parse_timedelta`,
    so ``forever``, an empty interval, and ``datetime.timedelta`` arguments
    such as ``days=1;hours=6`` are accepted.

    Rule selectors are ``models``, ``images``, ``videos``, ``other``, or a file
    extension such as ``.png``. ``model``, ``image``, and ``video`` are accepted
    as the same selectors. An extension rule overrides a category rule. ``other``
    applies only to files that no other rule matched. A file that matches nothing
    is not expired.

    ``models`` is every model file dgenerate can open:
    ``.safetensors``, ``.gguf``, ``.pt``, ``.pth``, ``.bin``, ``.ckpt``,
    ``.param``, ``.yaml``, and ``.yml``. ``images`` and ``videos`` are the image
    and video formats dgenerate can read.
    """

    _CATEGORIES = {
        'models': 'models',
        'model': 'models',
        'images': 'images',
        'image': 'images',
        'videos': 'videos',
        'video': 'videos',
        'other': 'other',
    }

    def __init__(self,
                 uniform: datetime.timedelta | None = None,
                 extensions: dict[str, datetime.timedelta] | None = None,
                 categories: dict[str, datetime.timedelta] | None = None,
                 other: datetime.timedelta | None = None):
        self.uniform_delta = uniform
        self._extensions = dict(extensions or {})
        self._categories = dict(categories or {})
        self._other = other

    @property
    def is_uniform(self) -> bool:
        """
        ``True`` when one interval applies to every cached file.

        This is the historical ``DGENERATE_WEB_CACHE_EXPIRY_DELTA`` syntax.
        """
        return self.uniform_delta is not None

    @classmethod
    def for_timedelta(cls, delta: datetime.timedelta) -> 'WebCacheExpiry':
        """
        Policy that expires every file after ``delta``.

        :param delta: Expiry interval.
        :return: Uniform policy.
        """
        return cls(uniform=delta)

    @classmethod
    def coerce(cls, value) -> 'WebCacheExpiry':
        """
        Accept a policy, a ``datetime.timedelta``, or an expiry expression.

        :param value: Policy value passed to :py:class:`WebFileCache`.
        :return: Policy.
        :raises TypeError: If ``value`` is not a supported type.
        :raises WebCacheExpiryParseError: If a string cannot be parsed.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, datetime.timedelta):
            return cls.for_timedelta(value)
        if value is None or isinstance(value, str):
            return cls.parse(value)
        raise TypeError(
            'expiry_delta must be a datetime.timedelta, an expiry expression, '
            'or a WebCacheExpiry.')

    @classmethod
    def parse(cls, value: str | None) -> 'WebCacheExpiry':
        """
        Parse web cache expiry syntax.

        These are one interval for every file:

        * ``days=5;hours=6``
        * ``hours=12``
        * ``forever``
        * an empty string

        A comma-separated rule list assigns intervals by file kind:

        ``models;forever,images;days=1,videos;days=2,other;hours=12``

        ``.png;hours=1,.jpg;days=1;hours=6``

        The text after the first semicolon of a rule is parsed by
        :py:func:`dgenerate.textprocessing.parse_timedelta`.

        :param value: Environment value or expression. Surrounding quotes are ignored.
        :return: Policy.
        :raises WebCacheExpiryParseError: On invalid syntax or an invalid interval.
        """
        if value is None:
            return cls.for_timedelta(_textprocessing.parse_timedelta(None))

        text = str(value).strip().strip('"\'')
        if _is_legacy_expiry(text):
            try:
                return cls.for_timedelta(_textprocessing.parse_timedelta(text))
            except _textprocessing.TimeDeltaParseError as e:
                raise WebCacheExpiryParseError(e) from e

        extensions: dict[str, datetime.timedelta] = {}
        categories: dict[str, datetime.timedelta] = {}
        other = None
        for part in (piece.strip() for piece in text.split(',')):
            if not part:
                continue
            selector, separator, spec = part.partition(';')
            selector = selector.strip()
            if not separator or not selector:
                raise WebCacheExpiryParseError(
                    f'Expiry rule "{part}" must be a selector and an interval '
                    f'separated by a semicolon, for example "images;days=1" or '
                    f'".png;hours=1". A single interval for every file has no '
                    f'comma, for example "days=5;hours=6".')
            delta = _parse_rule_interval(selector, spec)
            category = cls._CATEGORIES.get(selector.lower())
            if category == 'other':
                other = delta
                continue
            if category is not None:
                categories.pop(category, None)
                categories[category] = delta
                continue
            extension = _selector_extension(selector)
            extensions.pop(extension, None)
            extensions[extension] = delta

        return cls(extensions=extensions, categories=categories, other=other)

    def delta_for(self,
                  path: str | None = None,
                  metadata: dict | None = None,
                  ext: str | None = None) -> datetime.timedelta | None:
        """
        Expiry interval for a cached file.

        :param path: Cached file path. Its extension is used when ``ext`` and
            a metadata filename are not available.
        :param metadata: Cache metadata. ``filename`` supplies the extension
            when it has one.
        :param ext: Extension, with or without a leading dot.
        :return: Interval, or ``None`` when this file is not expired.
        """
        if self.is_uniform:
            return self.uniform_delta

        key = _normalize_extension(_extension_from(path, metadata, ext))
        if key in self._extensions:
            return self._extensions[key]
        for category, delta in self._categories.items():
            if key and key in _category_extensions(category):
                return delta
        return self._other

    def _expired_where(self, now: datetime.datetime) -> tuple[str, list]:
        """
        SQL predicate and bound values for rows this policy expires.

        Explicit extensions are tested before categories. Categories keep
        their written order. ``forever`` and a file with no rule produce a
        null cutoff, which does not match.
        """
        clauses = []
        params: list = []
        for extension, delta in self._extensions.items():
            clauses.append('WHEN ext = ? THEN ?')
            params.append(extension)
            params.append(_expiry_cutoff(delta, now))
        for category, delta in self._categories.items():
            names = sorted(_category_extensions(category))
            if not names:
                continue
            clauses.append(f"WHEN ext IN ({','.join('?' * len(names))}) THEN ?")
            params.extend(names)
            params.append(_expiry_cutoff(delta, now))
        if clauses:
            case = 'CASE ' + ' '.join(clauses) + ' ELSE ? END'
        else:
            case = 'CASE WHEN 0 THEN NULL ELSE ? END'
        params.append(_expiry_cutoff(self._other, now))
        return f'eligible = 1 AND dgenerate_before(creation_date, {case}) = 1', params


def _is_legacy_expiry(text: str) -> bool:
    if not text or text.lower() == 'forever':
        return True
    if ',' in text:
        return False
    pieces = [piece.strip() for piece in text.split(';') if piece.strip()]
    if not pieces:
        return True
    return all('=' in piece and not piece.startswith('.') for piece in pieces)


def parse_partial_expiry_delta(value: str = DEFAULT_WEB_CACHE_PARTIAL_EXPIRY) -> datetime.timedelta:
    """
    Parse ``DGENERATE_WEB_CACHE_PARTIAL_EXPIRY_DELTA``.

    The value is one interval in the syntax accepted by
    :py:func:`dgenerate.textprocessing.parse_timedelta`, such as ``hours=12``,
    ``days=2``, or ``days=1;hours=6``. ``forever`` and an empty string keep
    the partial. A rule list is not accepted.

    :param value: Environment value. Surrounding quotes are ignored.
        The default is :data:`DEFAULT_WEB_CACHE_PARTIAL_EXPIRY`.
    :return: Expiry interval for an interrupted download whose finished file
        would be kept forever.
    :raises WebCacheExpiryParseError: On a rule list or an invalid interval.
    """
    text = str(value).strip().strip('"\'')
    if not _is_legacy_expiry(text):
        raise WebCacheExpiryParseError(
            'DGENERATE_WEB_CACHE_PARTIAL_EXPIRY_DELTA accepts one interval, '
            'for example "hours=12" or "days=2". A rule list is not accepted.')
    try:
        return _textprocessing.parse_timedelta(text)
    except _textprocessing.TimeDeltaParseError as e:
        raise WebCacheExpiryParseError(e) from e


def _parse_rule_interval(selector: str, spec: str) -> datetime.timedelta:
    try:
        return _textprocessing.parse_timedelta(spec.strip())
    except _textprocessing.TimeDeltaParseError as e:
        raise WebCacheExpiryParseError(
            f'Invalid expiry interval "{spec.strip()}" for "{selector}": {e}') from e


def _selector_extension(selector: str) -> str:
    if not selector.startswith('.') and not selector.startswith('*.'):
        raise WebCacheExpiryParseError(
            f'Unknown expiry selector "{selector}". '
            f'Use models, images, videos, other, or an extension such as .png.')
    extension = _normalize_extension(selector)
    if not extension or any(character in extension for character in '\\/'):
        raise WebCacheExpiryParseError(
            f'Expiry extension "{selector}" is not a file extension.')
    return extension


def _category_extensions(category: str) -> frozenset[str]:
    if category == 'models':
        return _filetypes.model_extension_set()
    if category == 'images':
        return _filetypes.image_read_extension_set()
    if category == 'videos':
        return _filetypes.video_read_extension_set()
    return frozenset()


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
                 expiry_delta: datetime.timedelta | str | WebCacheExpiry = DEFAULT_WEB_CACHE_EXPIRY,
                 partial_expiry_delta: datetime.timedelta | str = DEFAULT_WEB_CACHE_PARTIAL_EXPIRY):
        """
        Initializes the :py:class:`.WebFileCache` object with a key-value store
        located at ``db_path``, a cache directory at ``cache_dir``, and an expiry policy.
        If the cache directory doesn't exist, it creates it. It also attempts to clear old files.

        ``expiry_delta`` accepts a ``datetime.timedelta``, an expiry expression, or a
        :py:class:`WebCacheExpiry`. The default is ``models;forever,other;hours=12``:
        model files are kept, and every other file expires after 12 hours.
        A timedelta or a single interval such as ``days=5;hours=6`` expires every file,
        which is the historical behavior. Intervals in a rule list are parsed by
        :py:func:`dgenerate.textprocessing.parse_timedelta`.

        :param db_path: The path to the key-value store database.
        :param cache_dir: The directory where the cache files are stored.
        :param expiry_delta: Expiry interval or :py:class:`WebCacheExpiry` policy.
            When the policy is a rule list, :py:attr:`expiry_delta` is ``None`` and
            :py:attr:`expiry` holds the policy.
        :param partial_expiry_delta: Interval for an interrupted download whose
            finished file would be kept forever. A string uses the same single
            interval syntax as :py:func:`dgenerate.textprocessing.parse_timedelta`.
            The default is :data:`DEFAULT_WEB_CACHE_PARTIAL_EXPIRY`.
        """
        super().__init__(db_path, cache_dir)
        self.expiry = WebCacheExpiry.coerce(expiry_delta)
        # Historical attribute. Uniform policies keep a timedelta here.
        self.expiry_delta = self.expiry.uniform_delta
        if isinstance(partial_expiry_delta, datetime.timedelta):
            self.partial_expiry_delta = partial_expiry_delta
        else:
            self.partial_expiry_delta = parse_partial_expiry_delta(partial_expiry_delta)
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

    def _partial_expiry_delta(self, partial_path: str) -> datetime.timedelta | None:
        if self.expiry.is_uniform:
            return self.expiry.uniform_delta
        meta = self._read_partial_state(partial_path)
        ext = meta.get('ext') if isinstance(meta, dict) else None
        if not ext and isinstance(meta, dict):
            metadata = meta.get('metadata')
            if isinstance(metadata, dict):
                filename = metadata.get('filename')
                if filename:
                    ext = os.path.splitext(str(filename))[1]
        delta = self.expiry.delta_for(ext=ext)
        # forever keeps a finished file. The partial is not finished.
        if delta == datetime.timedelta.max:
            return self.partial_expiry_delta
        return delta

    def _clear_old_partials(self):
        partial_dir = os.path.join(self.cache_dir, '.partials')
        if not os.path.isdir(partial_dir):
            return
        now = time.time()
        stale = []
        seen = {}
        for entry in os.scandir(partial_dir):
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith('.meta'):
                partial_path = os.path.join(partial_dir, name[:-len('.meta')])
            else:
                partial_path = entry.path
            delta = seen.get(partial_path)
            if partial_path not in seen:
                delta = self._partial_expiry_delta(partial_path)
                seen[partial_path] = delta
            if delta is None:
                continue
            try:
                cutoff = now - delta.total_seconds()
            except OverflowError:
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
        Clears files that are older than their expiry interval.

        A uniform policy uses one interval for every file. A rule list is one
        SQL delete: the extension is read from the stored JSON, and rows that
        match no rule are left in place.
        """
        if self.expiry.is_uniform:
            for key, cached_file in self.delete_older_than(self.expiry.uniform_delta):
                try:
                    os.unlink(cached_file.path)
                except FileNotFoundError:
                    pass
        else:
            where_sql, params = self.expiry._expired_where(datetime.datetime.now())
            for key, value in self.kv_store.delete_matching(where_sql, params):
                try:
                    os.unlink(CachedFile(json.loads(value)).path)
                except FileNotFoundError:
                    pass
        self._clear_old_partials()

    def request_mimetype(self, url, local_files_only: bool = False,
                         extra_headers: dict | None = None) -> str:
        """
        Requests the mimetype of a file at a URL. If the file exists in the cache, a known mimetype
        is returned without connecting to the internet. Otherwise, it connects to the internet
        to retrieve the mimetype. This action does not update the cache.

        :raise HTTPError: On http status errors.
        :raise WebFileCacheOfflineModeException: If local_files_only mode is enabled and the file is not found in the cache.

        :param url: The URL of the file.
        :param local_files_only: If ``True``, do not make a request, only check the cache.
        :param extra_headers: Extra request headers, such as Hugging Face authorization.
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

        headers = {'User-Agent': user_agent()}
        if extra_headers:
            headers.update(extra_headers)

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
                 local_files_only: bool = False,
                 extra_headers: dict | None = None) -> CachedFile:
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
        :param extra_headers: Extra request headers, such as Hugging Face authorization.
            These are not part of the cache key.
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
            cached_file = self.get(url)
            if cached_file is not None and os.path.exists(cached_file.path):
                # The lookup above locks the cache. The name request is a
                # network call and must not hold that lock.
                self._backfill_filename(
                    url, cached_file,
                    local_files_only=is_offline,
                    extra_headers=extra_headers)
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

            state['metadata'] = {'mime-type': mime_type}
            stored_name = _download_filename(url, response)
            if stored_name:
                state['metadata']['filename'] = stored_name
                _, ext = os.path.splitext(stored_name)
            else:
                _warn_missing_disposition(
                    url, 'the response had no Content-Disposition filename')
                try:
                    fallback = pyrfc6266.requests_response_to_filename(response)
                except Exception:
                    fallback = ''
                _, ext = os.path.splitext(str(fallback))
            state['ext'] = ext
            self._write_partial_state(partial_path, {
                'metadata': state['metadata'],
                'ext': ext,
            })

        if tqdm_pbar is not None:
            _messages.log(f'Downloading: "{url}"', underline=True)

        headers = {'User-Agent': user_agent()}
        if extra_headers:
            headers.update(extra_headers)

        download_url_to_file(
            url,
            partial_path,
            headers=headers,
            on_response=on_response,
            tqdm_pbar=tqdm_pbar)

        if 'metadata' not in state:
            saved = self._read_partial_state(partial_path)
            state['metadata'] = saved.get('metadata') or {'mime-type': 'unknown'}
            state['ext'] = saved.get('ext')

        return self._store_downloaded_file(
            url, partial_path, state['metadata'], state.get('ext'))

    def _remember_filename(self, url: str, cached_file: CachedFile, filename: str) -> None:
        metadata = dict(cached_file.metadata or {})
        metadata['filename'] = filename
        cached_file.metadata = metadata
        with self.kv_store as kv:
            raw = kv.get(url)
            if raw is None:
                return
            entry = json.loads(raw)
            entry['metadata'] = metadata
            # Keep creation_date. Replacing the row would refresh the expiry.
            kv.cursor.execute(
                "UPDATE store SET value=? WHERE key=?",
                (json.dumps(entry), url))

    def _fetch_disposition_filename(self, url: str,
                                    extra_headers: dict | None = None) -> tuple[str | None, str | None]:
        """
        Read ``Content-Disposition`` without saving the body.

        :return: ``(filename or None, failure reason or None)``
        """
        headers = {'User-Agent': user_agent()}
        if extra_headers:
            headers.update(extra_headers)
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=_DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                filename = _content_disposition_filename(response)
        except requests.RequestException as error:
            return None, _download_error_text(error)
        if not filename:
            return None, 'the response had no Content-Disposition filename'
        return filename, None

    def _backfill_filename(self, url: str, cached_file: CachedFile,
                           local_files_only: bool = False,
                           extra_headers: dict | None = None) -> None:
        """
        Store a missing ``Content-Disposition`` name on a file already in the cache.

        Nothing is recorded when the name cannot be read, so a later run tries again.
        """
        if (cached_file.metadata or {}).get('filename'):
            return
        from_url = _url_path_filename(url)
        if from_url:
            self._remember_filename(url, cached_file, from_url)
            return
        if self._local_files_only or local_files_only:
            _warn_missing_disposition(url, 'the web cache is offline')
            return
        filename, reason = self._fetch_disposition_filename(url, extra_headers)
        if not filename:
            _warn_missing_disposition(url, reason or 'the response had no Content-Disposition filename')
            return
        self._remember_filename(url, cached_file, filename)

    def filename_for_path(self, path: str) -> str | None:
        """
        ``Content-Disposition`` basename stored for a cached file path.

        The on-disk name is a generated id. This is the name the server sent.
        """
        try:
            wanted = os.path.normcase(os.path.abspath(path))
        except (OSError, ValueError):
            return None
        with self.kv_store as kv:
            kv.cursor.execute("SELECT value FROM store")
            for (raw,) in kv.cursor:
                entry = json.loads(raw)
                stored = entry.get('path')
                if not stored:
                    continue
                try:
                    candidate = os.path.normcase(os.path.abspath(stored))
                except (OSError, ValueError):
                    continue
                if candidate == wanted:
                    return (entry.get('metadata') or {}).get('filename') or None
        return None
