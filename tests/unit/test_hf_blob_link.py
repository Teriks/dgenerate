import unittest
from unittest.mock import patch

import dgenerate.hfhub as hfhub


class TestHFBlobLink(unittest.TestCase):

    def test_blob_and_resolve_parse_the_same_file(self):
        blob = hfhub.HFBlobLink.parse(
            'https://huggingface.co/calcuis/hunyuanimage-gguf/blob/main/hunyuanimage2.1-q4_k_m.gguf')
        resolve = hfhub.HFBlobLink.parse(
            'https://huggingface.co/calcuis/hunyuanimage-gguf/resolve/main/hunyuanimage2.1-q4_k_m.gguf')
        self.assertEqual(blob.repo_id, 'calcuis/hunyuanimage-gguf')
        self.assertEqual(resolve.repo_id, blob.repo_id)
        self.assertEqual(resolve.revision, 'main')
        self.assertIsNone(resolve.subfolder)
        self.assertEqual(resolve.weight_name, 'hunyuanimage2.1-q4_k_m.gguf')

    def test_resolve_keeps_a_subfolder_and_drops_the_query(self):
        parsed = hfhub.HFBlobLink.parse(
            'https://huggingface.co/Comfy-Org/Hunyuan-Image-2.1/resolve/main/'
            'split_files/diffusion_models/hunyuanimage2.1_fp8_e4m3fn.safetensors?download=true')
        self.assertEqual(parsed.repo_id, 'Comfy-Org/Hunyuan-Image-2.1')
        self.assertEqual(parsed.subfolder, 'split_files/diffusion_models')
        self.assertEqual(parsed.weight_name, 'hunyuanimage2.1_fp8_e4m3fn.safetensors')

    def test_other_huggingface_urls_do_not_parse(self):
        self.assertIsNone(hfhub.HFBlobLink.parse(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/boy.png'))
        self.assertIsNone(hfhub.HFBlobLink.parse(
            'https://huggingface.co/spaces/org/name/resolve/main/file.bin'))

    def test_resolve_downloads_into_the_huggingface_cache(self):
        url = ('https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/'
               'resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors')
        with patch('dgenerate.hfhub.huggingface_hub.hf_hub_download', return_value='cached.safetensors') as hub, \
                patch('dgenerate.hfhub._webcache.create_web_cache_file') as web:
            path = hfhub.webcache_or_hf_blob_download(url, auth_token='token')
        self.assertEqual(path, 'cached.safetensors')
        web.assert_not_called()
        hub.assert_called_once_with(
            repo_id='alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union',
            revision='main',
            subfolder=None,
            filename='Z-Image-Turbo-Fun-Controlnet-Union.safetensors',
            local_files_only=False,
            force_download=False,
            token='token')


if __name__ == '__main__':
    unittest.main()
