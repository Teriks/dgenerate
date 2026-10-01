import json
import os
import tempfile
import unittest
from unittest.mock import patch

import huggingface_hub.constants as hf_constants

from dgenerate._patches.hfhub_download_timeout_patch import apply_hf_download_timeout
from dgenerate.spacycache import _download_whl_file, _get_compatibility


class TestHfDownloadTimeout(unittest.TestCase):

    def test_unset_environment_uses_60_seconds(self):
        old_env = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')
        old_constant = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
        os.environ.pop('HF_HUB_DOWNLOAD_TIMEOUT', None)
        try:
            self.assertEqual(apply_hf_download_timeout(), 60)
            self.assertEqual(hf_constants.HF_HUB_DOWNLOAD_TIMEOUT, 60)
            self.assertEqual(os.environ['HF_HUB_DOWNLOAD_TIMEOUT'], '60')
        finally:
            if old_env is None:
                os.environ.pop('HF_HUB_DOWNLOAD_TIMEOUT', None)
            else:
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = old_env
            hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = old_constant

    def test_environment_value_is_kept(self):
        old_env = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')
        old_constant = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '15'
        try:
            self.assertEqual(apply_hf_download_timeout(), 15)
            self.assertEqual(hf_constants.HF_HUB_DOWNLOAD_TIMEOUT, 15)
        finally:
            if old_env is None:
                os.environ.pop('HF_HUB_DOWNLOAD_TIMEOUT', None)
            else:
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = old_env
            hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = old_constant


class TestSpacyDownloads(unittest.TestCase):

    def test_wheel_download_uses_shared_downloader(self):
        with patch('dgenerate.spacycache._filecache.download_url_to_file') as download:
            _download_whl_file('en_core_web_sm', 'https://example.test/model.whl', 'out.whl')
        download.assert_called_once()
        args, kwargs = download.call_args
        self.assertEqual(args[0], 'https://example.test/model.whl')
        self.assertEqual(args[1], 'out.whl')
        bar = kwargs['tqdm_pbar'](total=1)
        self.assertIn('en_core_web_sm', bar.desc)
        bar.close()

    def test_compatibility_index_uses_shared_downloader(self):
        payload = {'spacy': {'3.8': ['1.0.0']}}

        def fake_download(url, path, tqdm_pbar=None):
            self.assertEqual(url, 'https://example.test/compatibility.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)

        with tempfile.TemporaryDirectory() as directory, \
                patch('dgenerate.spacycache.get_spacy_cache_directory', return_value=directory), \
                patch('dgenerate.spacycache._filecache.download_url_to_file', fake_download), \
                patch('spacy.about.__compatibility__', 'https://example.test/compatibility.json', create=True), \
                patch('spacy.util.get_minor_version', return_value='3.8'):
            version = _get_compatibility(local_files_only=False)
            self.assertEqual(version, ['1.0.0'])
            with open(os.path.join(directory, 'compatibility.json'), encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['3.8'], ['1.0.0'])
