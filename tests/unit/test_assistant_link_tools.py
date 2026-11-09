import json
import unittest
from unittest.mock import patch

from dgenerate.assistant import link_tools
from dgenerate.assistant.models import ChatModel


class TestLinkTools(unittest.TestCase):

    def test_civitai_red_uses_that_host(self):
        with patch('dgenerate.assistant.link_tools._fetch_json', return_value={
            'name': 'Style',
            'type': 'LORA',
            'modelVersions': [{
                'id': 9,
                'name': 'krea2',
                'baseModel': 'Krea 2',
                'files': [{
                    'id': 1,
                    'type': 'Model',
                    'downloadUrl': 'https://civitai.com/api/download/models/9',
                }],
            }],
        }) as fetch_json:
            text = link_tools.fetch_civitai_link(
                'https://civitai.red/models/742211/style?modelVersionId=9')
        self.assertEqual(fetch_json.call_args.args[0], 'https://civitai.red/api/v1/models/742211')
        self.assertIn('https://civitai.red/api/download/models/9', text)
        self.assertNotIn('civitai.com', text)

    def test_find_link_urls(self):
        text = (
            'Use https://civitai.com/models/1307708/wan and '
            'https://huggingface.co/black-forest-labs/FLUX.1-dev please')
        urls = link_tools.find_link_urls(text)
        self.assertEqual(len(urls), 2)
        self.assertIn('civitai.com', urls[0])
        red = link_tools.find_link_urls('https://civitai.red/models/1/style')
        self.assertEqual(red, ['https://civitai.red/models/1/style'])

    def test_fetches_send_a_chrome_user_agent(self):
        with patch('requests.get') as get:
            get.return_value.raise_for_status.return_value = None
            get.return_value.json.return_value = {}
            link_tools._fetch_json('https://civitai.com/api/v1/models/1')
        headers = get.call_args.kwargs['headers']
        self.assertIn('Chrome', headers['user-agent'])
        self.assertNotIn('Android', headers['user-agent'])
        self.assertNotIn('Mobile', headers['user-agent'])
        self.assertNotIn('python-requests', headers['user-agent'].lower())
        self.assertIn('application/json', headers['accept'])

    def test_civitai_falls_back_to_the_other_host(self):
        import requests

        forbidden = requests.Response()
        forbidden.status_code = 403
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(url)
            response = requests.Response()
            if 'civitai.com' in url:
                error = requests.exceptions.HTTPError('403', response=forbidden)
                raise error
            response.status_code = 200
            response._content = b'{"id":1,"name":"ok","type":"LORA","modelVersions":[]}'
            response.json = lambda: {
                'id': 1,
                'name': 'ok',
                'type': 'LORA',
                'modelVersions': [{
                    'id': 9,
                    'name': 'krea2',
                    'baseModel': 'Krea 2',
                    'files': [],
                }],
            }
            return response

        with patch('requests.get', side_effect=fake_get):
            text = link_tools.fetch_civitai_link(
                'https://civitai.com/models/1/style?modelVersionId=9')
        self.assertTrue(any('civitai.red' in u for u in calls))
        self.assertIn('--model-type: krea2', text)

    def test_fetch_link_caches_failures(self):
        link_tools._FETCH_CACHE.clear()
        with patch('dgenerate.assistant.link_tools.fetch_civitai_link',
                   return_value='Failed to fetch. Do not call fetch_link on this URL again.') as fetch:
            first = link_tools.fetch_link('https://civitai.com/models/1/x')
            second = link_tools.fetch_link('https://civitai.com/models/1/x')
        self.assertEqual(first, second)
        self.assertEqual(fetch.call_count, 1)
        link_tools._FETCH_CACHE.clear()

    def test_redact_url_strips_token(self):
        url = 'https://civitai.com/api/download/models/1?token=secret'
        self.assertNotIn('secret', link_tools.redact_url(url))
        self.assertIn('models/1', link_tools.redact_url(url))

    def test_guess_model_type_from_base(self):
        self.assertEqual(link_tools._guess_model_type('Flux.1 D', '', ''), 'flux')
        self.assertEqual(link_tools._guess_model_type('Wan Video 2.1', '', ''), 'wan')
        self.assertEqual(link_tools._guess_model_type('SD 3.5 something', '', ''), 'sd3')
        self.assertIsNone(link_tools._guess_model_type('Hunyuan Video', '', ''))
        self.assertEqual(link_tools._guess_model_type('Hunyuan 1', '', ''), 'hydit')

    def test_civitai_base_picks_the_checkpoint(self):
        cases = {
            'Flux.1 Krea': ('flux', 'black-forest-labs/FLUX.1-Krea-dev'),
            'Flux.1 Kontext': ('flux-kontext', 'black-forest-labs/FLUX.1-Kontext-dev'),
            'Flux.2 D': ('flux2', 'black-forest-labs/FLUX.2-dev'),
            'Flux.2 Klein 4B': ('flux2', 'black-forest-labs/FLUX.2-klein-4B'),
            'Flux.2 Klein 4B-base': ('flux2', 'black-forest-labs/FLUX.2-klein-base-4B'),
            'Flux.2 Klein 9B': ('flux2', 'black-forest-labs/FLUX.2-klein-9B'),
            'Flux.2 Klein 9B-base': ('flux2', 'black-forest-labs/FLUX.2-klein-base-9B'),
            'Qwen': ('qwen-image', 'Qwen/Qwen-Image'),
            'Qwen 2.1': ('qwen-image-21', 'Qwen/Qwen-Image-2.1'),
            'ZImageTurbo': ('z-image', 'Tongyi-MAI/Z-Image-Turbo'),
            'ZImageBase': ('z-image', 'Tongyi-MAI/Z-Image'),
            'Krea 2 Turbo': ('krea2', 'krea/Krea-2-Turbo'),
            'LTXV': ('ltx-video', 'Lightricks/LTX-Video'),
            'LTXV2': ('ltx2', 'Lightricks/LTX-2'),
            'LTXV 2.3': ('ltx2', 'Lightricks/LTX-2.3'),
            'LTXV 2.5': ('ltx2', 'Lightricks/LTX-2.5-Diffusers'),
            'Wan Video 1.3B t2v': ('wan', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),
            'Wan Video 14B i2v 720p': ('wan', 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers'),
            'Wan Video 2.2 T2V-A14B': ('wan', 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'),
            'Wan Video 2.2 I2V-A14B': ('wan', 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'),
            'Wan Video 2.2 TI2V-5B': ('wan', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'),
            'Illustrious': ('sdxl', 'stabilityai/stable-diffusion-xl-base-1.0'),
            'NoobAI': ('sdxl', 'stabilityai/stable-diffusion-xl-base-1.0'),
            'Kolors': ('kolors', 'Kwai-Kolors/Kolors-diffusers'),
            'SD 3.5 Medium': ('sd3', 'stabilityai/stable-diffusion-3.5-medium'),
        }
        for base, (model_type, repo) in cases.items():
            self.assertEqual(link_tools._guess_model_type(base, 'v1', ''), model_type, base)
            self.assertEqual(link_tools._checkpoint_repo(base, model_type), repo, base)
        for base in ('Hunyuan Video', 'Wan Video 2.5 T2V', 'Flux.3', 'Anima', 'Chroma'):
            self.assertIsNone(link_tools._guess_model_type(base, 'Wan2.5 high', ''), base)
            self.assertIsNone(link_tools._checkpoint_repo(base, 'wan'), base)
        self.assertEqual(link_tools._guess_model_type('Krea 2', 'krea2', ''), 'krea2')
        self.assertEqual(
            link_tools._guess_model_type(None, 'fal/flux-2-klein-4b-spritesheet-lora', 'mentions flux'),
            'flux2')
        self.assertEqual(
            link_tools._guess_model_type(None, 'black-forest-labs/FLUX.2-klein-4B', ''),
            'flux2')
        self.assertEqual(
            link_tools._guess_model_type(
                None, 'black-forest-labs/FLUX.1-Krea-dev', 'Photographic finetune. This is not Krea 2.'),
            'flux')
        self.assertEqual(
            link_tools._guess_model_type(None, 'krea/Krea-2-Raw', 'mentions flux in passing'),
            'krea2')
        self.assertEqual(
            link_tools._guess_model_type(None, 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers', ''),
            'hydit')
        self.assertEqual(
            link_tools._guess_model_type(
                None, 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers', 'a hunyuan video org'),
            'hyimage-21')
        self.assertEqual(
            link_tools._guess_model_type(
                None, 'hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers', ''),
            'hyimage-21-refiner')
        self.assertEqual(
            link_tools._guess_model_type(
                None, 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v', ''),
            'hyvideo-15')

    def test_hub_names_keep_sibling_types_apart(self):
        from dgenerate.assistant.prompt import _request_kinds
        cases = {
            'Lightricks/LTX-Video': 'ltx-video',
            'DeepBeepMeep/LTX_Video': 'ltx-video',
            'QuantStack/LTXV-13B-0.9.8-dev-GGUF': 'ltx-video',
            'city96/LTX-Video-gguf': 'ltx-video',
            'charlied98/ltxv_2b_lora_soft_draw_v1': 'ltx-video',
            'Lightricks/LTX-2.5': 'ltx2',
            'Comfy-Org/ltx-2': 'ltx2',
            'Kijai/LTXV2_comfy': 'ltx2',
            'Muapi/ltxv-2.3-i2v-cowgirl-ride': 'ltx2',
            'unsloth/LTX-2.3-GGUF': 'ltx2',
            'Qwen/Qwen-Image-2.1': 'qwen-image-21',
            'Qwen/Qwen-Image-2.1-Turbo': 'qwen-image-21',
            'unsloth/Qwen-Image-2.1-GGUF': 'qwen-image-21',
            'qwen_image_2.1_Q4_0.gguf': 'qwen-image-21',
            'Qwen/Qwen-Image-Edit-2511': 'qwen-image-edit',
            'Qwen/Qwen-Image-Edit-2509': 'qwen-image-edit',
            'Qwen_Image_Edit-Q4_K_S.gguf': 'qwen-image-edit',
            'Qwen/Qwen-Image-Layered': 'qwen-image-layered',
            'Qwen/Qwen-Image': 'qwen-image',
            'Qwen_Image-Q4_K_S.gguf': 'qwen-image',
            'Comfy-Org/Flux-2-Dev': 'flux2',
            'Comfy-Org/Flux-2-Klein-4B': 'flux2',
            'black-forest-labs/FLUX.2-klein-base-4B': 'flux2',
            'black-forest-labs/FLUX.2-klein-9b-kv': 'flux2-klein-kv',
            'mflux-community/flux2-klein-9b-kv-mflux-q4': 'flux2-klein-kv',
            'QuantStack/FLUX.2-Klein-9B-KV-GGUF': 'flux2-klein-kv',
            'Wan-AI/Wan2.1-T2V-1.3B-Diffusers': 'wan',
            'Comfy-Org/Wan-2.1': 'wan',
            'city96/Wan2.1-I2V-14B-480P-gguf': 'wan',
            'Wan-AI/Wan2.2-Animate-14B-Diffusers': 'wan-animate',
            'AtelierDarren/Wan2.2_Animate': 'wan-animate',
            'Wan-AI/Wan2.2-Animate-2-14B-Diffusers': 'wan-animate-2',
            'stabilityai/stable-diffusion-xl-base-1.0': 'sdxl',
            'stabilityai/sdxl-turbo': 'sdxl',
            'diffusers/sdxl-instructpix2pix-768': 'sdxl-pix2pix',
            'stabilityai/stable-diffusion-3.5-large': 'sd3',
            'BleachNick/SD3_UltraEdit_freeform': 'sd3-pix2pix',
            'CaptainZZZ/sd3-instructpix2pix': 'sd3-pix2pix',
            'timbrooks/instruct-pix2pix': 'pix2pix',
            'Kwai-Kolors/Kolors-diffusers': 'kolors',
            'Kwai-Kolors/Kolors-Inpainting': 'kolors',
            'Muapi/kolors-style-asian-face-for-flux1-dev': 'flux',
            'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers': 'kandinsky5-image',
            'kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers': 'kandinsky5-image',
            'kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers': 'kandinsky5-video',
            'kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers': 'kandinsky5-video',
            'black-forest-labs/flux-3-image-klein': 'flux3',
            'molbal/Flux-3-Klein-GGUF': 'flux3',
            'kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers': 'kandinsky6-video',
            'kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers': 'kandinsky6-video',
            'Motif-Technologies/Motif-Video-2B': 'motif-video',
            'stabilityai/stable-cascade-prior': 's-cascade',
            'dn6/StableCascade-prior': 's-cascade',
            'DeepFloyd/IF-I-XL-v1.0': 'if',
            'DeepFloyd/IF-II-L-v1.0': 'ifs',
            'stabilityai/stable-diffusion-x4-upscaler': 'upscaler-x4',
            'stabilityai/sd-x2-latent-upscaler': 'upscaler-x2',
            'Tongyi-MAI/Z-Image-Turbo': 'z-image',
            'Tongyi-MAI/Z-Image-Omni-Base': 'z-image-omni',
            'krea/Krea-2-Turbo': 'krea2',
            'Comfy-Org/Krea-2': 'krea2',
            'tencent/HunyuanImage-2.1': 'hyimage-21',
            'Comfy-Org/Hunyuan-Image-2.1': 'hyimage-21',
            'calcuis/hunyuanimage-gguf': 'hyimage-21',
            'Tencent-Hunyuan/HunyuanDiT-Diffusers': 'hydit',
            'comfyanonymous/hunyuan_dit_comfyui': 'hydit',
            'DeepBeepMeep/HunyuanVideo1.5': 'hyvideo-15',
            'Comfy-Org/Hunyuan-Video-1.5': 'hyvideo-15',
        }
        unsupported = (
            'tencent/HunyuanImage-3.0',
            'Comfy-Org/Hunyuan-Video',
            'city96/HunyuanVideo-gguf',
            'DeepBeepMeep/Kandinsky5',
        )
        for name, expected in cases.items():
            self.assertEqual(link_tools._guess_model_type(None, name, ''), expected, name)
        for name in unsupported:
            self.assertIsNone(link_tools._guess_model_type(None, name, ''), name)
        self.assertEqual(_request_kinds('Qwen/Qwen-Image-2.1'), ['sequential'])
        self.assertEqual(_request_kinds('black-forest-labs/FLUX.2-klein-base-4B'), ['klein4-full'])
        self.assertEqual(
            _request_kinds('stabilityai/stable-diffusion-3.5-large'), ['sd3-full'])
        self.assertEqual(_request_kinds('city96/Wan2.1-I2V-14B-480P-gguf'), ['sequential'])
        self.assertEqual(_request_kinds('QuantStack/LTXV-13B-0.9.8-dev-GGUF'), ['ltx-gguf'])
        self.assertEqual(_request_kinds('Lightricks/LTX-2.5'), ['sequential'])
        self.assertNotIn('sequential', _request_kinds('tencent/HunyuanImage-3.0'))
        self.assertEqual(
            _request_kinds('kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers'), ['sequential'])
        self.assertEqual(_request_kinds('black-forest-labs/flux-3-image-klein'), ['sequential'])

    def test_example_configs_match_their_model_type(self):
        import re
        from pathlib import Path
        type_re = re.compile(r'--model-type\s+"?([\w.-]+)')
        root = Path(__file__).resolve().parents[2] / 'examples'
        mismatches = []
        for path in root.rglob('*.dgen'):
            pending = None
            for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
                stripped = raw.strip()
                if not stripped or stripped.startswith(('#', '\\', '{%', '{{', '--', '!', '%')):
                    match = type_re.search(stripped)
                    if match and pending:
                        expected = match.group(1)
                        guessed = link_tools._guess_model_type(None, pending, '')
                        allowed = guessed == expected or (
                            expected == 'ifs-img2img' and guessed == 'ifs')
                        if not allowed and 'civitai.com' not in pending.lower():
                            mismatches.append(f'{expected} guessed {guessed}: {pending}')
                        pending = None
                    continue
                head = stripped.split()[0].strip('"').split(';')[0]
                if '/' in head or head.startswith('http'):
                    pending = head
        self.assertEqual(mismatches, [])

    def test_publish_title_does_not_override_the_selected_base_model(self):
        text = link_tools._format_civitai(
            'https://civitai.com/models/742211/3d-render-style-flux-or-pony?modelVersionId=3392854',
            {
                'name': '3D Render Style (Flux | Pony)',
                'type': 'LORA',
                'modelVersions': [
                    {'id': 3392854, 'name': 'krea2', 'baseModel': 'Krea 2'},
                    {'id': 919014, 'name': 'v1.0 Pony', 'baseModel': 'Pony'},
                    {'id': 830009, 'name': 'v1.0 Flux', 'baseModel': 'Flux.1 D'},
                ],
            },
            {'id': 3392854, 'name': 'krea2', 'baseModel': 'Krea 2', 'trainedWords': ['3d render of']},
            version_was_requested=True,
        )
        self.assertIn('--model-type: krea2', text)
        self.assertIn('krea/Krea-2-Raw', text)
        self.assertIn('baseModel Pony', text)
        self.assertIn('not selected', text)
        self.assertIn('3d render of', text)
        self.assertIn('write these in --prompts', text)
        self.assertIn('token= is not a LoRA argument', text)

    def test_description_mention_does_not_override_the_name(self):
        guessed = link_tools._guess_model_type(
            'Flux.1 D', 'v1', 'people also use this with kontext')
        self.assertEqual(guessed, 'flux')
        named = link_tools._guess_model_type('Flux.1 D', 'Flux Kontext dev', '')
        self.assertEqual(named, 'flux-kontext')

    def test_missing_huggingface_repo_is_not_a_real_model(self):
        import httpx
        from huggingface_hub.errors import RepositoryNotFoundError

        url = 'https://huggingface.co/someone/FLUX.1-dev-Q4-GGUF'
        response = httpx.Response(404, request=httpx.Request('GET', url))

        class Api:
            def __init__(self, headers=None):
                pass

            def model_info(self, repo_id, revision=None):
                raise RepositoryNotFoundError('404 Repository not found', response=response)

        with patch('huggingface_hub.HfApi', Api):
            text = link_tools.fetch_huggingface_link(url)
        self.assertIn('someone/FLUX.1-dev-Q4-GGUF does not exist', text)
        self.assertIn('Do not invent', text)
        self.assertNotIn('gated', text.lower())

    def test_gated_huggingface_refusal_says_the_repo_is_real(self):
        import httpx
        from huggingface_hub.errors import GatedRepoError

        url = 'https://huggingface.co/black-forest-labs/FLUX.1-dev'
        response = httpx.Response(403, request=httpx.Request('GET', url))

        class Api:
            def __init__(self, headers=None):
                pass

            def model_info(self, repo_id, revision=None):
                raise GatedRepoError('403 gated', response=response)

        with patch('huggingface_hub.HfApi', Api):
            text = link_tools.fetch_huggingface_link(url)
        self.assertIn('exists and is gated', text)
        self.assertIn('HF_TOKEN', text)
        self.assertNotIn('does not exist', text)

    def test_huggingface_dataset_is_not_a_model_repo(self):
        text = link_tools.fetch_huggingface_link(
            'https://huggingface.co/datasets/huggingface/documentation-images')
        self.assertIn('not a model repo', text)

    def test_huggingface_url_keeps_the_file_path(self):
        parsed = link_tools._parse_hf_url(
            'https://huggingface.co/example-org/example-loras/raw/main/weights/Sample%20Name/sample_name.md')
        self.assertEqual(parsed['repo_id'], 'example-org/example-loras')
        self.assertEqual(parsed['kind'], 'raw')
        self.assertEqual(parsed['path'], 'weights/Sample Name/sample_name.md')
        root = link_tools._parse_hf_url('https://huggingface.co/models/org/repo')
        self.assertEqual(root['kind'], 'repo')
        self.assertEqual(root['path'], '')

    def test_huggingface_repo_lists_every_weight_and_the_readme(self):
        files = [
            'README.md',
            'weights/Other Name/other_name.safetensors',
            'weights/Sample Name/sample_name.safetensors',
            'weights/Sample Name/sample_name.md',
            'weights/Sample Name/captions/0000.txt',
            'previews/sample.jpg',
        ]
        text = link_tools._format_huggingface(
            'https://huggingface.co/example-org/example-loras',
            'example-org/example-loras',
            kind='repo', path='', pipeline='text-to-image', gated=False,
            tags=['krea2', 'lora'], card_text='', files=files,
            texts={'README.md': 'A collection of LoRAs.'},
            suggested='krea2')
        self.assertIn('weights/Sample Name/sample_name.safetensors', text)
        self.assertIn('weights/Sample Name/sample_name.md', text)
        self.assertNotIn('Notable files:', text)
        self.assertIn('1 .txt', text)
        self.assertNotIn('0000.txt', text)
        self.assertIn('subfolder=Folder/Name With Spaces;weight-name=file.safetensors', text)
        self.assertIn('not a file to copy', text)
        self.assertIn('README.md:', text)
        self.assertIn('--model-type: krea2', text)
        self.assertIn('raw/main/', text)
        self.assertIn('gated: no', text)
        self.assertIn('Do not add an HF_TOKEN block', text)
        gated = link_tools._format_huggingface(
            'https://huggingface.co/black-forest-labs/FLUX.1-dev',
            'black-forest-labs/FLUX.1-dev',
            kind='repo', path='', pipeline='text-to-image', gated=True,
            tags=[], card_text='', files=['README.md'], texts={}, suggested='flux')
        self.assertIn('gated: yes', gated)
        self.assertIn('Write the HF_TOKEN early-exit', gated)

    def test_gguf_folder_lists_the_quant_files_to_copy(self):
        files = [
            'README.md',
            'flux1-dev-Q4_K_S.gguf',
            'flux1-dev-Q5_K_S.gguf',
            'flux1-dev-Q6_K.gguf',
            'flux1-dev-Q8_0.gguf',
        ]
        text = link_tools._format_huggingface(
            'https://huggingface.co/city96/FLUX.1-dev-gguf',
            'city96/FLUX.1-dev-gguf',
            kind='repo', path='', pipeline='', gated=False, tags=[],
            card_text='', files=files, texts={}, suggested=None)
        self.assertIn('Copy one of these quant paths exactly. Do not invent a quant suffix.', text)
        self.assertIn('flux1-dev-Q5_K_S.gguf', text)
        self.assertIn('flux1-dev-Q6_K.gguf', text)
        self.assertNotIn('not a file to copy', text)

    def test_comfy_folder_lists_the_quant_files_to_copy(self):
        files = [
            'README.md',
            'diffusion_models/krea2_raw_bf16.safetensors',
            'diffusion_models/krea2_raw_int8_convrot.safetensors',
            'diffusion_models/krea2_turbo_fp8_scaled.safetensors',
            'diffusion_models/krea2_turbo_fp8mixed.safetensors',
            'diffusion_models/krea2_turbo_nvfp4.safetensors',
            'vae/ae.safetensors',
        ]
        text = link_tools._format_huggingface(
            'https://huggingface.co/Comfy-Org/Krea-2',
            'Comfy-Org/Krea-2',
            kind='repo', path='', pipeline='', gated=False, tags=[],
            card_text='', files=files, texts={}, suggested=None)
        self.assertIn('Copy one of these quant paths exactly. Do not invent a quant suffix.', text)
        self.assertIn('diffusion_models/krea2_turbo_fp8mixed.safetensors', text)
        self.assertIn('diffusion_models/krea2_turbo_nvfp4.safetensors', text)
        self.assertNotIn('not a file to copy', text)
        quant_at = text.index('Quant files')
        weights_at = text.index('Weight files')
        self.assertLess(quant_at, weights_at)
        quant_block = text[quant_at:weights_at]
        self.assertNotIn('vae/ae.safetensors', quant_block)

    def test_offline_fetch_lists_the_hub_cache_and_does_not_call_the_hub(self):
        import os
        import tempfile
        from unittest.mock import patch

        import huggingface_hub

        import dgenerate.hfhub as hfhub

        commit = 'abc123'
        with tempfile.TemporaryDirectory() as cache:
            snap = os.path.join(
                cache, 'models--vantagewithai--LTX-2.5-GGUF', 'snapshots', commit, 'dev')
            os.makedirs(snap)
            refs = os.path.join(cache, 'models--vantagewithai--LTX-2.5-GGUF', 'refs')
            os.makedirs(refs)
            with open(os.path.join(refs, 'main'), 'w', encoding='utf-8') as handle:
                handle.write(commit + '\n')
            for name in (
                    'ltx-2.5-22b-dev-transformer-Q2_K.gguf',
                    'ltx-2.5-22b-dev-transformer-Q4_K_S.gguf'):
                with open(os.path.join(snap, name), 'wb') as handle:
                    handle.write(b'gguf')
            real_cache = huggingface_hub.constants.HF_HUB_CACHE
            huggingface_hub.constants.HF_HUB_CACHE = cache
            link_tools._FETCH_CACHE.clear()
            try:
                with hfhub.offline_mode_context(True), \
                        patch('huggingface_hub.HfApi.model_info', side_effect=AssertionError('network')):
                    text = link_tools.fetch_huggingface_link(
                        'https://huggingface.co/vantagewithai/LTX-2.5-GGUF')
                    missing = link_tools.fetch_huggingface_link(
                        'https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF')
                    civit = link_tools.fetch_link('https://civitai.com/models/1')
            finally:
                huggingface_hub.constants.HF_HUB_CACHE = real_cache
                link_tools._FETCH_CACHE.clear()
        self.assertIn('only the files already in the Hugging Face cache', text)
        self.assertIn('dev/ltx-2.5-22b-dev-transformer-Q2_K.gguf', text)
        self.assertIn('gated: unknown while offline', text)
        self.assertNotIn('gated: no', text)
        self.assertIn('not in the local cache', missing)
        self.assertIn('Do not invent a quant suffix', missing)
        self.assertIn('CivitAI link is not in the local cache', civit)

    def test_huggingface_note_url_returns_that_note_and_a_quoted_lora(self):
        files = [
            'README.md',
            'weights/Sample Name/sample_name.safetensors',
            'weights/Sample Name/sample_name.md',
            'weights/Other Name/other_name.safetensors',
        ]
        note = 'weights/Sample Name/sample_name.md'
        text = link_tools._format_huggingface(
            'https://huggingface.co/example-org/example-loras/raw/main/' + note.replace(' ', '%20'),
            'example-org/example-loras',
            kind='raw', path=note, pipeline='', gated=False, tags=['krea2'],
            card_text='', files=files,
            texts={note: 'Trigger word: SampleName', 'README.md': 'collection'},
            suggested='krea2')
        self.assertIn('Trigger word: SampleName', text)
        self.assertIn(
            '--loras "example-org/example-loras;subfolder=weights/Sample Name;'
            'weight-name=sample_name.safetensors"',
            text)
        self.assertNotIn('other_name', text)

    def test_krea2_tag_is_a_model_type(self):
        self.assertEqual(link_tools._guess_model_type(None, 'example-org/example-lora', 'krea2'), 'krea2')

    def test_fetch_huggingface_reads_the_named_note(self):
        files = [
            'README.md',
            'weights/Sample Name/sample_name.safetensors',
            'weights/Sample Name/sample_name.md',
        ]

        class Info:
            tags = ['krea2']
            pipeline_tag = None
            gated = False
            card_data = None
            siblings = []

        class Api:
            def __init__(self, headers=None):
                pass

            def model_info(self, repo_id, revision=None):
                return Info()

            def list_repo_files(self, repo_id, revision=None):
                return files

        def fake_text(repo_id, filename, revision, limit):
            if filename.endswith('sample_name.md'):
                return 'Trigger word: SampleName'
            return 'readme body'

        url = ('https://huggingface.co/example-org/example-loras/raw/main/'
               'weights/Sample%20Name/sample_name.md')
        with patch('huggingface_hub.HfApi', Api), \
                patch('dgenerate.assistant.link_tools._hf_text_file', side_effect=fake_text):
            text = link_tools.fetch_huggingface_link(url)
        self.assertIn('Trigger word: SampleName', text)
        self.assertIn('weight-name=sample_name.safetensors', text)
        self.assertIn('readme body', text)

    @patch('dgenerate.assistant.link_tools._fetch_json')
    def test_download_url_marks_the_selected_file(self, fetch_json):
        fetch_json.return_value = {
            'modelId': 1,
            'id': 9,
            'name': 'v1',
            'baseModel': 'SD 1.5',
            'files': [
                {'id': 1, 'type': 'Model', 'downloadUrl': 'https://civitai.com/api/download/models/9?fileId=1'},
                {'id': 2, 'type': 'VAE', 'downloadUrl': 'https://civitai.com/api/download/models/9?fileId=2'},
            ],
        }
        text = link_tools.fetch_civitai_link(
            'https://civitai.com/api/download/models/9?fileId=2')
        self.assertIn('VAE [selected by fileId]', text)
        self.assertNotIn('Model [selected by fileId]', text)

    @patch('dgenerate.assistant.link_tools._fetch_json')
    def test_fetch_civitai_model_page(self, fetch_json):
        fetch_json.return_value = {
            'name': 'Test Checkpoint',
            'type': 'Checkpoint',
            'modelVersions': [{
                'id': 99,
                'name': 'v1',
                'baseModel': 'Flux.1 D',
                'description': 'Use 20 steps',
                'trainedWords': [],
                'files': [{'type': 'Model', 'downloadUrl': 'https://civitai.com/api/download/models/99'}],
            }],
        }
        text = link_tools.fetch_civitai_link('https://civitai.com/models/1/test')
        self.assertIn('Suggested dgenerate --model-type: flux', text)
        self.assertIn('Checkpoint', text)

    def test_run_tool_fetch_link(self):
        with patch('dgenerate.assistant.link_tools.fetch_link', return_value='summary'):
            out = link_tools.run_tool('fetch_link', json.dumps({'url': 'https://civitai.com/models/1/x'}))
        self.assertEqual(out, 'summary')

    def test_check_config_tool_reports_a_comment_between_options(self):
        config = (
            '#! /usr/bin/env dgenerate --file\n'
            '#! dgenerate 6.0.0\n'
            '\n'
            'stabilityai/stable-diffusion-xl-base-1.0\n'
            '--model-type sdxl\n'
            '# between options\n'
            '--prompts "a fox"\n')
        text = link_tools.run_tool('check_config', {'config': config})
        self.assertIn('comment', text)
        self.assertIn('check_config again', text)
        names = [item['function']['name'] for item in link_tools.tool_definitions(fetch=False, check=True)]
        self.assertEqual(names, ['check_config', 'submit_config'])


class TestChatModelToolLoop(unittest.TestCase):

    def test_submit_config_is_the_reply(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        calls = {'n': 0}

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                calls['n'] += 1
                return {
                    'choices': [{
                        'message': {
                            'content': 'here is a code block that must be ignored',
                            'tool_calls': [{
                                'id': 'call_1',
                                'function': {
                                    'name': 'submit_config',
                                    'arguments': json.dumps({'config': 'the config'}),
                                },
                            }],
                        },
                    }],
                }

        chat._server = Server()
        with patch('dgenerate.assistant.check.consider_submission', return_value=(True, 'the config\n', 1)):
            reply = chat.complete(
                [{'role': 'user', 'content': 'hi'}],
                tools=[{'type': 'function', 'function': {'name': 'submit_config'}}])
        self.assertEqual(reply, 'the config\n')
        self.assertEqual(calls['n'], 1)

    def test_tool_loop_then_answer(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        calls = {'n': 0}
        log = []

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                calls['n'] += 1
                if calls['n'] == 1:
                    return {
                        'choices': [{
                            'message': {
                                'content': '',
                                'tool_calls': [{
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'fetch_link',
                                        'arguments': json.dumps({
                                            'url': 'https://civitai.com/models/1/x',
                                        }),
                                    },
                                }],
                            },
                        }],
                    }
                return {
                    'choices': [{
                        'message': {
                            'content': '```dgen\nstable-diffusion-v1-5/stable-diffusion-v1-5\n--prompts "cat"\n```',
                        },
                    }],
                }

        chat._server = Server()
        messages = [{'role': 'user', 'content': 'hi'}]
        with patch('dgenerate.assistant.models.run_tool', return_value='Suggested dgenerate --model-type: sd'):
            reply = chat.complete(messages, link_tools=True, on_tool=log.append)
        self.assertIn('stable-diffusion-v1-5', reply)
        self.assertTrue(any('fetch_link' in line for line in log))
        self.assertEqual(calls['n'], 2)
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertIn('tool_calls', messages[1])
        self.assertEqual(messages[2]['role'], 'tool')
        self.assertIn('--model-type: sd', messages[2]['content'])

    def test_logged_tool_url_drops_the_token(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        calls = {'n': 0}
        log = []

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                calls['n'] += 1
                if calls['n'] == 1:
                    return {
                        'choices': [{
                            'message': {
                                'tool_calls': [{
                                    'id': 'call_1',
                                    'function': {
                                        'name': 'fetch_link',
                                        'arguments': json.dumps({
                                            'url': 'https://civitai.com/api/download/models/1?token=secret',
                                        }),
                                    },
                                }],
                            },
                        }],
                    }
                return {'choices': [{'message': {'content': '```dgen\nsd\n```'}}]}

        chat._server = Server()
        with patch('dgenerate.assistant.models.run_tool', return_value='ok'):
            chat.complete([{'role': 'user', 'content': 'hi'}], link_tools=True, on_tool=log.append)
        self.assertFalse(any('secret' in line for line in log))
        self.assertTrue(any('fetch_link' in line for line in log))

    def test_tools_rejection_falls_back_and_is_logged(self):
        import io
        from contextlib import redirect_stderr

        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                if on_chunk is None:
                    return {'error': {'message': 'tools are not supported by this template'}}
                on_chunk({'choices': [{'delta': {'content': '```dgen\nstable-diffusion-v1-5/stable-diffusion-v1-5\n```'}}]})

        chat._server = Server()
        err = io.StringIO()
        with redirect_stderr(err):
            reply = chat.complete([{'role': 'user', 'content': 'hi'}], link_tools=True)
        self.assertIn('stable-diffusion-v1-5', reply)
        self.assertIn('link tools unavailable', err.getvalue())

    def test_context_overflow_keeps_link_tools(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        seen = []

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                seen.append('tools' in body)
                if len(seen) == 1:
                    return {'error': {'message': 'request exceeds the available context size'}}
                return {'choices': [{'message': {'content': '```dgen\nblack-forest-labs/FLUX.1-schnell\n```'}}]}

        chat._server = Server()
        reply = chat.complete([
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'first'},
            {'role': 'user', 'content': 'https://civitai.com/models/1/x'},
        ], link_tools=True)
        self.assertIn('FLUX.1-schnell', reply)
        self.assertEqual(seen, [True, True])

    def test_context_overflow_on_repair_keeps_the_draft(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        bodies = []

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                bodies.append(body)
                if len(bodies) == 1:
                    return {'error': {'message': 'request exceeds the available context size'}}
                return {'choices': [{'message': {'content': '```dgen\nkrea/Krea-2-Raw\n--text-encoders x\n```'}}]}

        chat._server = Server()
        draft = '```dgen\nkrea/Krea-2-Raw\n--text-encers x\n```'
        reply = chat.complete([
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'x' * 40000},
            {'role': 'assistant', 'content': draft},
            {'role': 'user', 'content': 'Fix --text-encers and reply with the complete corrected config.'},
        ], link_tools=True)
        self.assertIn('--text-encoders', reply)
        resent = [m.get('content') for m in bodies[1]['messages']]
        self.assertEqual(resent[0], 'rules')
        self.assertIn('--text-encers', resent[1])
        self.assertIn('Fix --text-encers', resent[2])
        self.assertNotIn('x' * 40000, ''.join(resent))

    def test_context_overflow_repair_with_config_does_not_also_resend_the_draft(self):
        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        bodies = []

        class Server:
            def handle_chat_completions(self, body, on_chunk=None):
                bodies.append(body)
                if len(bodies) == 1:
                    return {'error': {'message': 'request exceeds the available context size'}}
                return {'choices': [{'message': {'content': 'ok'}}]}

        chat._server = Server()
        chat.complete([
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'retrieved examples'},
            {'role': 'assistant', 'content': '```dgen\nkrea/Krea-2-Raw\n```'},
            {'role': 'user', 'content': 'The rejected config:\n\n```dgen\nkrea/Krea-2-Raw\n```\n\nFix it.'},
        ], link_tools=True)
        resent = bodies[1]['messages']
        self.assertEqual(len(resent), 2)
        self.assertIn('The rejected config', resent[1]['content'])


if __name__ == '__main__':
    unittest.main()
