import datetime
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import requests
from requests.structures import CaseInsensitiveDict

from dgenerate.filecache import WebFileCache, download_url_to_file


class _Response:
    def __init__(self, status, headers, chunks, url):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers)
        self._chunks = list(chunks)
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} Error', response=self)

    def iter_content(self, chunk_size=1):
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDownloadUrlToFile(unittest.TestCase):

    def setUp(self):
        self._sleep = patch('dgenerate.filecache.time.sleep').start()
        self.addCleanup(patch.stopall)

    def _headers(self, length, **extra):
        headers = {
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(length),
        }
        headers.update(extra)
        return headers

    def test_read_timeout_resumes_with_range(self):
        payload = b'abcdefghij'
        calls = []
        url = 'https://cdn.example/model.bin?token=secret-token'

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append({'headers': dict(headers or {}), 'timeout': timeout})
            if len(calls) == 1:
                return _Response(
                    200,
                    self._headers(len(payload)),
                    [payload[:4], requests.ConnectionError(
                        'Read timed out. https://cdn.example/model.bin?token=secret-token')],
                    request_url)
            start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
            rest = payload[start:]
            return _Response(
                206,
                self._headers(
                    len(rest),
                    **{'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}'}),
                [rest],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with patch('dgenerate.filecache.requests.get', fake_get), \
                    patch('dgenerate.filecache._messages.log') as log:
                result = download_url_to_file(url, path, tqdm_pbar=None, attempts=3)

            self.assertEqual(result, path)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)

        self.assertEqual(calls[0]['timeout'], (10, 60))
        self.assertNotIn('Range', calls[0]['headers'])
        self.assertEqual(calls[1]['headers']['Range'], 'bytes=4-')
        logged = ' '.join(str(call) for call in log.call_args_list)
        self.assertIn('Resuming', logged)
        self.assertNotIn('secret-token', logged)
        self._sleep.assert_called()

    def test_ignored_range_restarts_without_duplicating(self):
        payload = b'abcdefghij'
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(dict(headers or {}))
            if len(calls) == 1:
                return _Response(
                    200,
                    self._headers(len(payload)),
                    [payload[:4], requests.ConnectionError('reset')],
                    request_url)
            return _Response(200, self._headers(len(payload)), [payload], request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with patch('dgenerate.filecache.requests.get', fake_get):
                download_url_to_file(
                    'https://cdn.example/model.bin', path, tqdm_pbar=None, attempts=3)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
        self.assertEqual(calls[1]['Range'], 'bytes=4-')

    def test_existing_partial_is_resumed(self):
        payload = b'abcdefghij'
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(dict(headers or {}))
            start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
            rest = payload[start:]
            return _Response(
                206,
                self._headers(
                    len(rest),
                    **{'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}'}),
                [rest],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with open(path, 'wb') as handle:
                handle.write(payload[:6])
            with patch('dgenerate.filecache.requests.get', fake_get):
                download_url_to_file(
                    'https://cdn.example/model.bin', path, tqdm_pbar=None)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
        self.assertEqual(calls[0]['Range'], 'bytes=6-')

    def test_unknown_path_resumes_without_burning_the_only_attempt(self):
        payload = b'abcdefghij'
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(dict(headers or {}))
            if 'Range' not in (headers or {}):
                return _Response(200, self._headers(len(payload)), [payload], request_url)
            start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
            rest = payload[start:]
            return _Response(
                206,
                self._headers(
                    len(rest),
                    **{'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}'}),
                [rest],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin.unfinished')
            with open(path, 'wb') as handle:
                handle.write(payload[:4])

            def resolve_path(response):
                return path

            with patch('dgenerate.filecache.requests.get', fake_get):
                download_url_to_file(
                    'https://cdn.example/model.bin',
                    resolve_path=resolve_path,
                    tqdm_pbar=None,
                    attempts=1)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
        self.assertEqual(calls[1]['Range'], 'bytes=4-')

    def test_short_read_is_resumed(self):
        payload = b'abcdefghij'
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(dict(headers or {}))
            if 'Range' not in (headers or {}):
                return _Response(200, self._headers(len(payload)), [payload[:3]], request_url)
            start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
            rest = payload[start:]
            return _Response(
                206,
                self._headers(
                    len(rest),
                    **{'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}'}),
                [rest],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with patch('dgenerate.filecache.requests.get', fake_get):
                download_url_to_file(
                    'https://cdn.example/model.bin', path, tqdm_pbar=None, attempts=3)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
        self.assertEqual(calls[1]['Range'], 'bytes=3-')

    def test_http_404_is_not_retried(self):
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(1)
            return _Response(404, {'Content-Type': 'text/plain'}, [], request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with patch('dgenerate.filecache.requests.get', fake_get):
                with self.assertRaises(requests.HTTPError):
                    download_url_to_file(
                        'https://cdn.example/missing.bin', path, tqdm_pbar=None, attempts=4)
        self.assertEqual(len(calls), 1)
        self._sleep.assert_not_called()

    def test_retries_stop_after_attempt_limit(self):
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            calls.append(1)
            return _Response(
                200,
                self._headers(8),
                [requests.ConnectionError('Read timed out.')],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with patch('dgenerate.filecache.requests.get', fake_get):
                with self.assertRaises(requests.ConnectionError):
                    download_url_to_file(
                        'https://cdn.example/model.bin', path, tqdm_pbar=None, attempts=3)
        self.assertEqual(len(calls), 3)

    def test_completed_partial_accepts_range_not_satisfiable(self):
        payload = b'abcdefghij'

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            return _Response(
                416,
                {'Content-Range': f'bytes */{len(payload)}'},
                [],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model.bin')
            with open(path, 'wb') as handle:
                handle.write(payload)
            with patch('dgenerate.filecache.requests.get', fake_get):
                download_url_to_file(
                    'https://cdn.example/model.bin', path, tqdm_pbar=None)
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)


class TestWebFileCacheDownload(unittest.TestCase):

    def setUp(self):
        patch('dgenerate.filecache.time.sleep').start()
        self.addCleanup(patch.stopall)

    def test_failed_download_is_resumed_on_the_next_call(self):
        payload = b'0123456789abcdef'
        url = 'https://civitai.com/api/download/models/1?token=secret-token'
        phase = {'resume': False}
        calls = []

        def fake_get(request_url, headers=None, stream=None, timeout=None):
            headers = dict(headers or {})
            calls.append(headers)
            disposition = 'attachment; filename="weights.safetensors"'
            if not phase['resume']:
                if 'Range' in headers:
                    start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
                    return _Response(
                        206,
                        {
                            'Content-Type': 'application/octet-stream',
                            'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}',
                            'Content-Disposition': disposition,
                        },
                        [requests.ConnectionError('Read timed out.')],
                        request_url)
                return _Response(
                    200,
                    {
                        'Content-Type': 'application/octet-stream',
                        'Content-Length': str(len(payload)),
                        'Content-Disposition': disposition,
                    },
                    [payload[:5], requests.ConnectionError('Read timed out.')],
                    request_url)
            start = int(headers['Range'].split('=', 1)[1].rstrip('-'))
            rest = payload[start:]
            return _Response(
                206,
                {
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(len(rest)),
                    'Content-Range': f'bytes {start}-{len(payload) - 1}/{len(payload)}',
                    'Content-Disposition': disposition,
                },
                [rest],
                request_url)

        with tempfile.TemporaryDirectory() as directory:
            cache = WebFileCache(
                os.path.join(directory, 'cache.db'),
                directory,
                expiry_delta=datetime.timedelta(hours=12))
            with patch('dgenerate.filecache.requests.get', fake_get):
                with self.assertRaises(requests.ConnectionError):
                    cache.download(url, tqdm_pbar=None)
            self.assertIsNone(cache.get(url))
            partials = [
                path for path in pathlib.Path(directory, '.partials').iterdir()
                if path.is_file() and not path.name.endswith('.meta')]
            self.assertEqual(len(partials), 1)
            self.assertEqual(partials[0].read_bytes(), payload[:5])

            phase['resume'] = True
            calls.clear()
            with patch('dgenerate.filecache.requests.get', fake_get):
                cached = cache.download(url, tqdm_pbar=None)
            self.assertTrue(cached.path.endswith('.safetensors'))
            with open(cached.path, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
            self.assertEqual(calls[0]['Range'], 'bytes=5-')
            self.assertFalse(partials[0].exists())

            calls.clear()
            with patch('dgenerate.filecache.requests.get', fake_get):
                again = cache.download(url, tqdm_pbar=None)
            self.assertEqual(again.path, cached.path)
            self.assertEqual(calls, [])
