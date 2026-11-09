import unittest

from dgenerate.assistant.check import check_config


def _cfg(*lines: str) -> str:
    return '#! /usr/bin/env dgenerate --file\n#! dgenerate 6.0.0\n\n' + '\n'.join(lines) + '\n'


def _messages(report: dict) -> str:
    return ' '.join(e['message'] for e in report['errors'])


class TestAssistantCheck(unittest.TestCase):

    def test_ltx_video_repo_is_not_ltx2(self):
        report = check_config(_cfg(
            'Lightricks/LTX-Video',
            '--model-type ltx2',
            '--prompts "A horse walks across a field."',
        ))
        self.assertFalse(report['ok'], report)
        text = _messages(report)
        self.assertIn('LTX-Video', text)
        self.assertIn('--model-type ltx-video', text)
        self.assertIn('Lightricks/LTX-2.5-Diffusers', text)

    def test_ltx25_repo_is_not_ltx_video(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx-video',
            '--prompts "A horse walks across a field."',
        ))
        self.assertFalse(report['ok'], report)
        text = _messages(report)
        self.assertIn('--model-type ltx2', text)
        self.assertIn('not ltx-video', text)

    def test_ltx23_repo_is_not_ltx_video(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.3',
            '--model-type ltx-video',
            '--prompts "A horse walks across a field."',
        ))
        self.assertFalse(report['ok'], report)
        text = _messages(report)
        self.assertIn('LTX-2.3', text)
        self.assertIn('not ltx-video', text)
        self.assertIn('Keep that repo', text)

    def test_ltx23_repo_with_ltx2_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.3',
            '--model-type ltx2',
            '--prompts "A horse walks across a field."',
        ))
        self.assertNotIn('not ltx-video', _messages(report))

    def test_ltx25_repo_with_ltx2_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2',
            '--prompts "A horse walks across a field."',
        ))
        self.assertNotIn('is LTX-Video', _messages(report))
        self.assertNotIn('not ltx-video', _messages(report))

    def test_krea2_repo_is_not_flux(self):
        report = check_config(_cfg(
            'krea/Krea-2-Raw',
            '--model-type flux',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type krea2, not flux', text)
        self.assertIn('black-forest-labs/FLUX.1-dev', text)
        self.assertIn('Keep that repo', text)

    def test_flux_krea_dev_is_not_krea2(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-Krea-dev',
            '--model-type krea2',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type flux, not krea2', text)
        self.assertIn('krea/Krea-2-Raw', text)

    def test_krea2_repo_with_krea2_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'krea/Krea-2-Turbo',
            '--model-type krea2',
            '--prompts "a fox"',
        ))
        self.assertNotIn('uses --model-type', _messages(report))

    def test_hunyuan_dit_is_not_video(self):
        report = check_config(_cfg(
            'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
            '--model-type hyvideo-15',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type hydit, not hyvideo-15', text)
        self.assertIn('HunyuanVideo-1.5', text)

    def test_hunyuan_video_is_not_dit(self):
        report = check_config(_cfg(
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v',
            '--model-type hydit',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type hyvideo-15, not hydit', text)
        self.assertIn('HunyuanDiT', text)

    def test_hunyuan_image_is_not_dit(self):
        report = check_config(_cfg(
            'hunyuanvideo-community/HunyuanImage-2.1-Diffusers',
            '--model-type hydit',
            '--prompts "a fox"',
        ))
        self.assertIn(
            'uses --model-type hyimage-21, not hydit', _messages(report))

    def test_hunyuan_image_refiner_keeps_its_type(self):
        report = check_config(_cfg(
            'hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers',
            '--model-type hyimage-21',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type hyimage-21-refiner, not hyimage-21', text)
        self.assertIn('Keep that repo', text)

    def test_wan_animate_is_not_wan(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.2-Animate-14B-Diffusers',
            '--model-type wan',
            '--prompts "a dancer"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type wan-animate, not wan', text)
        self.assertIn('Wan2.1-T2V-1.3B-Diffusers', text)

    def test_wan_animate_2_is_not_wan_animate(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.2-Animate-2-14B-Diffusers',
            '--model-type wan-animate',
            '--prompts "a dancer"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type wan-animate-2, not wan-animate', text)
        self.assertIn('Keep that repo', text)

    def test_wan_t2v_is_not_animate(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            '--model-type wan-animate',
            '--prompts "a fox runs"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type wan, not wan-animate', text)
        self.assertIn('Wan2.2-Animate-14B-Diffusers', text)

    def test_wan_t2v_with_wan_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            '--model-type wan --dtype bfloat16',
            '--model-sequential-offload',
            '--video-lengths 2',
            '--prompts "a fox runs"',
        ))
        self.assertNotIn('uses --model-type', _messages(report))

    def test_qwen_image_edit_is_not_qwen_image(self):
        report = check_config(_cfg(
            'Qwen/Qwen-Image-Edit',
            '--model-type qwen-image',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type qwen-image-edit, not qwen-image', text)
        self.assertIn('Qwen/Qwen-Image', text)
        self.assertIn('Keep that repo', text)

    def test_qwen_image_is_not_edit(self):
        report = check_config(_cfg(
            'Qwen/Qwen-Image',
            '--model-type qwen-image-edit',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type qwen-image, not qwen-image-edit', text)
        self.assertIn('Qwen/Qwen-Image-Edit', text)

    def test_qwen_image_layered_and_21(self):
        layered = check_config(_cfg(
            'Qwen/Qwen-Image-Layered',
            '--model-type qwen-image',
            '--prompts "a fox"',
        ))
        self.assertIn(
            'uses --model-type qwen-image-layered, not qwen-image',
            _messages(layered))
        newer = check_config(_cfg(
            'Qwen/Qwen-Image-2.1',
            '--model-type qwen-image',
            '--prompts "a fox"',
        ))
        self.assertIn(
            'uses --model-type qwen-image-21, not qwen-image',
            _messages(newer))

    def test_flux_fill_is_not_flux_or_kontext(self):
        fill = check_config(_cfg(
            'black-forest-labs/FLUX.1-Fill-dev',
            '--model-type flux',
            '--prompts "a fox"',
        ))
        text = _messages(fill)
        self.assertIn('uses --model-type flux-fill, not flux', text)
        self.assertIn('black-forest-labs/FLUX.1-dev', text)
        swapped = check_config(_cfg(
            'black-forest-labs/FLUX.1-Fill-dev',
            '--model-type flux-kontext',
            '--prompts "a fox"',
        ))
        self.assertIn(
            'uses --model-type flux-fill, not flux-kontext', _messages(swapped))

    def test_flux_kontext_is_not_flux(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-Kontext-dev',
            '--model-type flux',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type flux-kontext, not flux', text)
        self.assertIn('black-forest-labs/FLUX.1-dev', text)

    def test_flux_dev_with_flux_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux --dtype bfloat16',
            '--prompts "a fox"',
        ))
        self.assertNotIn('uses --model-type', _messages(report))

    def test_sdxl_pix2pix_is_not_sdxl(self):
        report = check_config(_cfg(
            'diffusers/sdxl-instructpix2pix-768',
            '--model-type sdxl',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type sdxl-pix2pix, not sdxl', text)
        self.assertIn('stabilityai/stable-diffusion-xl-base-1.0', text)
        self.assertIn('Keep that repo', text)

    def test_instruct_pix2pix_is_not_sdxl(self):
        report = check_config(_cfg(
            'timbrooks/instruct-pix2pix',
            '--model-type sdxl',
            '--prompts "a fox"',
        ))
        text = _messages(report)
        self.assertIn('uses --model-type pix2pix, not sdxl', text)
        self.assertIn('stabilityai/stable-diffusion-xl-base-1.0', text)

    def test_sdxl_base_with_sdxl_is_not_a_family_mismatch(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --prompts "a fox"',
        ))
        self.assertNotIn('uses --model-type', _messages(report))

    def test_inpaint_without_mask_templated_seed(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --output-path base --prompts "mountains"',
            '',
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            '--model-type sdxl',
            '--image-seeds {{ quote(first(last_images)) }}',
            '--prompts "mountains"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('inpainting', _messages(report).lower())

    def test_inpaint_without_mask_plain_seed(self):
        report = check_config(_cfg(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            '--model-type sdxl',
            '--image-seeds path/to/photo.png',
            '--prompts "a beach"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('mask', _messages(report).lower())

    def test_inpaint_with_mask_ok(self):
        report = check_config(_cfg(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--image-seeds path/to/photo.png;path/to/mask.png',
            '--prompts "a beach"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_flux_fill_needs_mask(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-Fill-dev',
            '--model-type flux-fill --dtype bfloat16',
            '--image-seeds path/to/photo.png',
            '--prompts "a dog on a bench"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertTrue(
            any(w in _messages(report).lower() for w in ('mask', 'inpaint')),
            report)

    def test_flux_kontext_needs_seeds(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-Kontext-dev',
            '--model-type flux-kontext --dtype bfloat16',
            '--prompts "replace the sky with sunset"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('image-seeds', _messages(report).lower())

    def test_gguf_as_model_path(self):
        report = check_config(_cfg(
            'path/to/flux1-dev-Q4_K_S.gguf',
            '--model-type flux --dtype bfloat16',
            '--prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('.gguf', _messages(report).lower())

    def test_gguf_as_transformer_ok(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux --dtype bfloat16',
            '--transformer path/to/flux1-dev-Q4_K_S.gguf',
            '--model-cpu-offload',
            '--quantizer bnb;bits=4 --quantizer-map text_encoder',
            '--prompts "a fox"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_image_process_overwrites_last_images(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --output-path base --prompts "mountains"',
            '',
            r'\image_process {{ quote(first(last_images)) }}',
            '--processors outpaint-mask;box=128',
            '--output mask.png -ox',
            '',
            r'\image_process {{ quote(first(last_images)) }}',
            '--processors letterbox;box-size=128;box-is-padding=True',
            '--output letterboxed.png -ox',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('last_images', _messages(report))

    def test_letterbox_named_mask(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --output-path base --prompts "mountains"',
            '',
            r'\set input_image {{ quote(first(last_images)) }}',
            r'\image_process {{ input_image }}',
            '--processors outpaint-mask;box=128',
            '--output mask.png -ox',
            '',
            r'\image_process mask.png',
            '--processors letterbox;box-size=128;box-is-padding=True',
            '--output letterboxed.png -ox',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('mask', _messages(report).lower())

    def test_letterbox_then_patchmatch_ok(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--output-path base --prompts "mountains"',
            '',
            r'\set input_image {{ quote(first(last_images)) }}',
            r'\image_process {{ input_image }}',
            '--processors outpaint-mask;box=128',
            '--output mask.png -ox',
            '',
            r'\image_process {{ input_image }}',
            '--processors letterbox;box-size=128;box-is-padding=True',
            '--output letterboxed.png -ox',
            '',
            r'\image_process letterboxed.png',
            '--processors patchmatch;mask=mask.png;seed=42',
            '--output filled.png -ox',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_saved_image_process_ok(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--output-path base --prompts "mountains"',
            '',
            r'\set input_image {{ quote(first(last_images)) }}',
            r'\set outpaint_box 128',
            '',
            r'\image_process {{ input_image }}',
            '--processors outpaint-mask;box={{ outpaint_box }}',
            '--output mask.png -ox',
            '',
            r'\image_process {{ input_image }}',
            '--processors letterbox;box-size={{ outpaint_box }};box-is-padding=True \\',
            '    patchmatch;mask=mask.png;seed=42',
            '--output filled.png -ox',
            '',
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--image-seeds filled.png;mask.png',
            '--image-seed-strengths 0.85',
            '--prompts "mountains"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_runaway_comments(self):
        comments = '\n'.join(f'# repeated comment {i}' for i in range(25))
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --prompts "mountains"',
            '',
            comments,
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('comments', _messages(report).lower())

    def test_pt_then_image_seed_strengths(self):
        report = check_config(_cfg(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--output-size 512 --image-format pt --prompts "a tiger"',
            '',
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--output-size 1024',
            '--image-seeds {{ quote(last_images) }}',
            '--image-seed-strengths 0.4',
            '--prompts "a tiger"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('latents', _messages(report).lower())

    def test_two_step_refine_latents_ok(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--image-format pt --denoising-end 0.8 --output-path base',
            '--prompts "a castle"',
            '',
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--image-seeds {{ quote(last_images) }}',
            '--denoising-start 0.8 --output-path refined',
            '--prompts "a castle"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_hires_fix_images_ok(self):
        report = check_config(_cfg(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--dtype float16 --output-size 512 --output-path base',
            '--prompts "a tiger"',
            '',
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--dtype float16 --output-size 1024 --output-path hires',
            '--image-seeds {{ quote(last_images) }}',
            '--image-seed-strengths 0.4',
            '--prompts "a tiger"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_upscaler_shrinks(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --output-size 1024 --prompts "a portrait"',
            '',
            'stabilityai/stable-diffusion-x4-upscaler',
            '--model-type upscaler-x4 --variant fp16',
            '--output-size 256',
            '--image-seeds {{ quote(first(last_images)) }}',
            '--prompts "a portrait"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('shrink', _messages(report).lower())

    def test_upscaler_without_seeds(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-x4-upscaler',
            '--model-type upscaler-x4 --variant fp16',
            '--prompts "sharp"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('image-seeds', _messages(report).lower())

    def test_last_animations_on_kontext(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-Kontext-dev',
            '--model-type flux-kontext --dtype bfloat16',
            '--image-seeds {{ quote(first(last_animations)) }}',
            '--prompts "replace the water with sunset"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('last_animations', _messages(report))

    def test_last_animations_after_image_then_ltx(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --prompts "a dog"',
            '',
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2 --dtype bfloat16',
            '--image-seeds {{ quote(first(last_animations)) }}',
            '--video-lengths 2',
            '--prompts "the dog wags its tail"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('last_images', _messages(report))

    def test_image_to_video_last_images_ok(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16 --prompts "a dog"',
            '',
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2 --dtype bfloat16',
            '--model-sequential-offload',
            '--image-seeds {{ quote(first(last_images)) }}',
            '--video-lengths 2',
            '--prompts "the dog wags its tail"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_refiner_png_without_denoising_start(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --prompts "a castle"',
            '',
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            '--model-type sdxl',
            '--image-seeds {{ quote(last_images) }}',
            '--prompts "a castle"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('denoising-start', _messages(report))

    def test_sdxl_refiner_on_base_ok(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0',
            '--prompts "a castle"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_scheduler_help_is_not_generation(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --schedulers help',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('help', _messages(report).lower())

    def test_output_file_reused(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--control-nets diffusers/controlnet-canny-sdxl-1.0',
            '--control-image-processors canny;output-file=canny_result.png',
            '--image-seeds path/to/pose.jpg',
            '--prompts "a fighter"',
            '',
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --variant fp16 --dtype float16',
            '--image-seeds canny_result.png',
            '--prompts "a fighter"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('output-file', _messages(report))

    def test_delayed_expansion_still_rejected(self):
        report = check_config(_cfg(
            r'{% if have_cuda() %}',
            r'\set model stabilityai/stable-diffusion-xl-base-1.0',
            r'{{ model }}',
            '--model-type sdxl --prompts "a fox"',
            r'{% endif %}',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('{{ model }}', _messages(report))

    def test_delayed_escape_inside_non_token_if_ok(self):
        report = check_config(_cfg(
            r'{% if have_cuda() %}',
            r'\set model stabilityai/stable-diffusion-xl-base-1.0',
            r"{{ '{{ model }}' }}",
            '--model-type sdxl --prompts "a fox"',
            r'{% endif %}',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_token_if_must_exit_not_wrap(self):
        report = check_config(_cfg(
            r'\set civit_ai_token %CIVIT_AI_TOKEN%',
            r'{% if civit_ai_token.strip() %}',
            r'\set model https://civitai.com/api/download/models/1?token={{civit_ai_token}}',
            r"{{ '{{ model }}' }}",
            '--model-type sdxl --prompts "a fox"',
            r'{% endif %}',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('exit', _messages(report).lower())

    def test_civitai_early_exit_ok(self):
        report = check_config(_cfg(
            r'\set token %CIVIT_AI_TOKEN%',
            r'{% if not token.strip() %}',
            r'\print Set CIVIT_AI_TOKEN',
            r'\exit',
            r'{% endif %}',
            r'\set model https://civitai.com/api/download/models/1?token={{token}}',
            r'{{ model }}',
            '--model-type sdxl --prompts "a fox"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_three_part_repo(self):
        report = check_config(_cfg(
            'stabilityai/stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('organization/name', _messages(report))

    def test_no_invocation(self):
        report = check_config(_cfg('# just a comment'))
        self.assertFalse(report['ok'], report)
        self.assertIn('no dgenerate invocation', _messages(report))

    def test_print_says_example(self):
        report = check_config(_cfg(
            r'\print Set HF_TOKEN environmental variable or --auth-token to run this example!',
            r'\exit',
            '',
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux --prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('example', _messages(report).lower())

    def test_prompt_weighter_without_syntax(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux',
            '--prompt-weighter sd-embed',
            '--prompts "alien planet surface with strange creatures"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('no weighting', _messages(report))

    def test_sd_embed_plus_plus_is_not_a_weight(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux',
            '--prompt-weighter sd-embed',
            '--prompts "C++ game screenshot"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('no weighting', _messages(report))

    def test_compel_plus_counts(self):
        report = check_config(_cfg(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            '--prompt-weighter compel',
            '--prompts "fox+ in the snow"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_prompt_weighter_with_syntax_ok(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux',
            '--model-sequential-offload',
            '--prompt-weighter sd-embed',
            '--prompts "(alien creatures:1.3), bioluminescent flora"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_flux_negative_prompt(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux',
            '--prompts "alien planet; blurry, low quality"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('negative prompt', _messages(report))

    def test_flux_gguf_wallpaper_weighter_copy(self):
        report = check_config(_cfg(
            r'\set token %HF_TOKEN%',
            '',
            r"{% if not token.strip() and not '--auth-token' in injected_args %}",
            r'    \print Set HF_TOKEN environmental variable or --auth-token to run this example!',
            r'    \exit',
            r'{% endif %}',
            '',
            'black-forest-labs/FLUX.1-dev',
            '--model-type flux',
            '--dtype bfloat16',
            '--transformer https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf',
            '--prompt-weighter sd-embed',
            '--output-size 1920x1080',
            '--prompts "alien planet surface with strange creatures; blurry, low quality"',
        ))
        self.assertFalse(report['ok'], report)
        msg = _messages(report).lower()
        self.assertIn('example', msg)
        self.assertIn('no weighting', msg)
        self.assertIn('negative prompt', msg)

    def test_last_images_in_same_for_loop(self):
        report = check_config(_cfg(
            r'{% for image in last_images %}',
            'stabilityai/stable-diffusion-xl-base-1.0',
            '--model-type sdxl --image-seeds {{ quote(image) }} --prompts "refine"',
            '',
            'Kwai-Kolors/Kolors-diffusers',
            '--model-type kolors --image-seeds {{ quote(last_images) }} --prompts "again"',
            r'{% endfor %}',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('last_images', _messages(report))

    def test_blank_line_names_the_split_invocation(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2',
            '--prompts "a fox"',
            '',
            '--guidance-scales 3',
        ))
        text = _messages(report)
        self.assertNotIn('required: model_path', text)
        self.assertIn('blank line', text)
        self.assertIn('Do not wrap --prompts', text)

    def test_comment_between_options_ends_the_invocation(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.2-klein-base-9B',
            '--model-type flux2',
            '--loras "https://civitai.com/api/download/models/1"',
            '# Flux never takes a negative prompt.',
            '--prompts "a vaporwave horse"',
        ))
        text = _messages(report)
        self.assertIn('comment on line', text)
        self.assertIn('above the model path', text)
        self.assertNotIn('backslash', text.lower())

    def test_wan_animate_needs_pose_or_driving(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.2-Animate-14B-Diffusers',
            '--model-type wan-animate --dtype bfloat16',
            '--image-seeds examples/media/earth.jpg',
            '--prompts "a dancer"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('wan-pose=', _messages(report))

    def test_wan_option_on_ltx(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2 --dtype bfloat16',
            '--wan-boundary-ratios 0.9',
            '--prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('wan', _messages(report).lower())

    def test_wan_text_to_video_ok(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            '--model-type wan --dtype bfloat16',
            '--model-sequential-offload',
            '--video-lengths 2',
            '--prompts "a fox runs"',
        ))
        self.assertTrue(report['ok'], report['errors'])

    def test_klein_4b_on_16gib_needs_a_quantizer(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.2-klein-4B',
            '--model-type flux2 --dtype bfloat16',
            '--inference-steps 4 --guidance-scales 1',
            '--prompts "a red fox"',
        ), memory=(True, 16.0, True, False))
        self.assertFalse(report['ok'], report)
        self.assertIn('--quantizer bnb;bits=4', _messages(report))

    def test_klein_base_4b_on_16gib_needs_a_quantizer(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.2-klein-base-4B',
            '--model-type flux2 --dtype bfloat16',
            '--inference-steps 4 --guidance-scales 1',
            '--prompts "a red fox"',
        ), memory=(True, 16.0, True, False))
        self.assertFalse(report['ok'], report)
        self.assertIn('--quantizer bnb;bits=4', _messages(report))
        self.assertNotIn('sequential-offload', _messages(report))

    def test_lora_trigger_word_is_not_a_uri_argument(self):
        report = check_config(_cfg(
            'krea/Krea-2-Raw',
            '--model-type krea2 --dtype bfloat16',
            '--model-sequential-offload',
            '--loras "https://civitai.com/api/download/models/3262678?fileId=3146115;token=vap0rglitch"',
            '--prompts "vap0rglitch, a fox"',
        ), memory=(True, 24.0, True, False))
        text = _messages(report)
        self.assertIn('Unknown path argument "token"', text)
        self.assertIn('weight-name', text)
        plain = check_config(_cfg(
            'krea/Krea-2-Raw',
            '--model-type krea2 --dtype bfloat16',
            '--model-sequential-offload',
            '--loras "https://civitai.com/api/download/models/3262678?fileId=3146115"',
            '--prompts "vap0rglitch, a fox"',
        ), memory=(True, 24.0, True, False))
        self.assertNotIn('Unknown path argument', _messages(plain))

    def test_klein_4b_quantizer_is_enough(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.2-klein-4B',
            '--model-type flux2 --dtype bfloat16',
            '--quantizer bnb;bits=4',
            '--prompts "a red fox"',
        ), memory=(True, 16.0, True, False))
        self.assertNotIn('memory flags', _messages(report))

    def test_spaced_blob_folder_is_the_whole_file(self):
        import dgenerate.assistant.check as assistant_check

        repo = 'realrebelai/Kandinsky-6.0-Pro-5s_GGUFs'
        transformer = (
            f'https://huggingface.co/{repo}/blob/main/'
            'Diffusion Model/Full/Kandinsky-6.0-Pro-5s-Q4_K_S.gguf')
        encoder = (
            'Qwen2_5_VLForConditionalGeneration;model='
            f'https://huggingface.co/{repo}/blob/main/'
            'Text Encoder/K6-Qwen2.5-VL-7B-w4a8.safetensors')
        found = assistant_check._huggingface_blob_files([
            transformer,
            encoder,
            transformer + '?download=true',
            'model=' + transformer + ';revision=main',
            transformer + ' and then a fox',
        ])
        self.assertEqual(
            [(item[0], item[2]) for item in found],
            [
                (repo, 'Diffusion Model/Full/Kandinsky-6.0-Pro-5s-Q4_K_S.gguf'),
                (repo, 'Text Encoder/K6-Qwen2.5-VL-7B-w4a8.safetensors'),
            ])

        files = [
            'Diffusion Model/Full/Kandinsky-6.0-Pro-5s-Q4_K_S.gguf',
            'Text Encoder/K6-Qwen2.5-VL-7B-w4a8.safetensors',
        ]
        real_list = assistant_check._repo_file_list
        real_cached = assistant_check._hub_cache_has
        real_repo = assistant_check._repo_problem
        assistant_check._repo_file_list = lambda repo_id, cache: cache.setdefault(repo_id, files)
        assistant_check._hub_cache_has = lambda *args: False
        assistant_check._repo_problem = lambda *args, **kwargs: None
        try:
            kept = check_config(_cfg(
                'kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers',
                '--model-type kandinsky6-video',
                '--dtype bfloat16',
                '--model-sequential-offload',
                f'--transformer "{transformer}"',
                f'--text-encoders "{encoder}" +',
                '--inference-steps 50',
                '--guidance-scales 5',
                '--video-lengths 5',
                '--video-fps 24',
                '--animation-format mp4',
                '--prompts "a fox"',
            ), memory=(True, 16.0, True, False))
            missing = check_config(_cfg(
                'kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers',
                '--model-type kandinsky6-video',
                '--dtype bfloat16',
                '--model-sequential-offload',
                '--transformer "' + transformer.replace('Q4_K_S', 'Q8_0') + '"',
                '--prompts "a fox"',
            ), memory=(True, 16.0, True, False))
        finally:
            assistant_check._repo_file_list = real_list
            assistant_check._hub_cache_has = real_cached
            assistant_check._repo_problem = real_repo
        self.assertNotIn('has no file', _messages(kept))
        self.assertIn(
            'has no file "Diffusion Model/Full/Kandinsky-6.0-Pro-5s-Q8_0.gguf"',
            _messages(missing))
        self.assertNotIn('has no file "Diffusion"', _messages(missing))

    def test_cached_hf_blob_is_not_listed_on_the_hub(self):
        import dgenerate.assistant.check as assistant_check

        listed = []
        real_list = assistant_check._repo_file_list
        real_cached = assistant_check._hub_cache_has
        real_repo = assistant_check._repo_problem

        def listing(repo, cache):
            listed.append(repo)
            cache[repo] = ['gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf']
            return cache[repo]

        assistant_check._repo_file_list = listing
        assistant_check._hub_cache_has = lambda repo, revision, path: 'Q4_K_M' in path
        assistant_check._repo_problem = lambda *args, **kwargs: None
        try:
            report = check_config(_cfg(
                'Lightricks/LTX-2.5-Diffusers',
                '--model-type ltx2 --dtype bfloat16',
                '--model-sequential-offload',
                '--text-encoders "Gemma4UnifiedForConditionalGeneration;model=https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF/blob/main/gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf"',
                '--guidance-scales 1',
                '--prompts "a fox walks"',
            ), memory=(True, 24.0, True, False))
        finally:
            assistant_check._repo_file_list = real_list
            assistant_check._hub_cache_has = real_cached
            assistant_check._repo_problem = real_repo
        self.assertNotIn('elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF', listed)
        self.assertNotIn('has no file', _messages(report))

    def test_offline_blob_must_be_in_the_local_cache(self):
        import dgenerate.assistant.check as assistant_check
        import dgenerate.assistant.link_tools as link_tools
        import dgenerate.hfhub as hfhub

        listed = []
        real_list = assistant_check._repo_file_list
        real_cached = assistant_check._hub_cache_has
        real_files = link_tools._cached_repo_files

        assistant_check._repo_file_list = lambda repo, cache: listed.append(repo)
        assistant_check._hub_cache_has = lambda repo, revision, path: False
        link_tools._cached_repo_files = lambda repo, revision: [
            'dev/ltx-2.5-22b-dev-transformer-Q4_K_S.gguf']
        try:
            with hfhub.offline_mode_context(True):
                missing = check_config(_cfg(
                    'Lightricks/LTX-2.5-Diffusers',
                    '--model-type ltx2 --dtype bfloat16',
                    '--model-sequential-offload',
                    '--transformer https://huggingface.co/vantagewithai/LTX-2.5-GGUF/blob/main/dev/ltx-2.5-22b-dev-transformer-Q5_K_S.gguf',
                    '--guidance-scales 3',
                    '--prompts "a fox walks"',
                ), memory=(True, 24.0, True, False))
                kept = check_config(_cfg(
                    'Lightricks/LTX-2.5-Diffusers',
                    '--model-type ltx2 --dtype bfloat16',
                    '--model-sequential-offload',
                    '--transformer https://huggingface.co/vantagewithai/LTX-2.5-GGUF/blob/main/dev/ltx-2.5-22b-dev-transformer-Q4_K_S.gguf',
                    '--guidance-scales 3',
                    '--prompts "a fox walks"',
                ), memory=(True, 24.0, True, False))
        finally:
            assistant_check._repo_file_list = real_list
            assistant_check._hub_cache_has = real_cached
            link_tools._cached_repo_files = real_files
        text = _messages(missing)
        self.assertIn('has no file "dev/ltx-2.5-22b-dev-transformer-Q5_K_S.gguf"', text)
        self.assertIn('dev/ltx-2.5-22b-dev-transformer-Q4_K_S.gguf', text)
        self.assertNotIn('has no file', _messages(kept))
        self.assertEqual(listed, [])

    def test_huggingface_file_cache_checks_the_hub_cache_only(self):
        import os
        import tempfile

        import dgenerate.assistant.check as assistant_check
        import huggingface_hub

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            cached_path = handle.name
        real_hub = huggingface_hub.try_to_load_from_cache

        def hub(repo_id, filename, revision=None, **kwargs):
            if filename.endswith('Q4_K_M.gguf'):
                return cached_path
            return None

        huggingface_hub.try_to_load_from_cache = hub
        try:
            self.assertTrue(assistant_check._hub_cache_has(
                'elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF', 'main',
                'gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf'))
            self.assertFalse(assistant_check._hub_cache_has(
                'elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF', 'main',
                'gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf'))
        finally:
            huggingface_hub.try_to_load_from_cache = real_hub
            os.remove(cached_path)

    def test_ltx25_gemma4_q8_blob_does_not_exist(self):
        report = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2 --dtype bfloat16',
            '--model-sequential-offload',
            '--transformer https://huggingface.co/realrebelai/LTX-2.5_GGUFs/blob/main/LTX-2.5-Distilled-Q8_0.gguf',
            '--text-encoders "Gemma4UnifiedForConditionalGeneration;model=https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF/blob/main/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf"',
            '--guidance-scales 1 --video-fps 24',
            '--prompts "a fox walks"',
        ), memory=(True, 24.0, True, False))
        text = _messages(report)
        self.assertIn('has no file "gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf"', text)
        self.assertIn('gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf', text)
        self.assertIn('gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf', text)
        self.assertNotIn('has no file "LTX-2.5-Distilled-Q8_0.gguf"', text)
        kept = check_config(_cfg(
            'Lightricks/LTX-2.5-Diffusers',
            '--model-type ltx2 --dtype bfloat16',
            '--model-sequential-offload',
            '--text-encoders "Gemma4UnifiedForConditionalGeneration;model=https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF/blob/main/gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf"',
            '--guidance-scales 1',
            '--prompts "a fox walks"',
        ), memory=(True, 24.0, True, False))
        self.assertNotIn('has no file', _messages(kept))

    def test_memory_check_always_runs(self):
        report = check_config(_cfg(
            'black-forest-labs/FLUX.2-klein-4B',
            '--model-type flux2 --prompts "a red fox"',
        ))
        self.assertIn('memory flags', _messages(report))

    def test_z_image_omni_unreleased(self):
        report = check_config(_cfg(
            'Tongyi-MAI/Z-Image-Omni-Base',
            '--model-type z-image-omni --dtype bfloat16',
            '--prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('not released', _messages(report).lower())


if __name__ == '__main__':
    unittest.main()
