import unittest

from dgenerate.assistant.check import check_config


def _cfg(*lines: str) -> str:
    return '#! /usr/bin/env dgenerate --file\n#! dgenerate 6.0.0\n\n' + '\n'.join(lines) + '\n'


def _messages(report: dict) -> str:
    return ' '.join(e['message'] for e in report['errors'])


class TestAssistantCheck(unittest.TestCase):

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
            '--model-type ltx --dtype bfloat16',
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
            '--model-type ltx --dtype bfloat16',
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
            '--model-type ltx',
            '--prompts "a fox"',
            '',
            '--guidance-scales 3',
        ))
        text = _messages(report)
        self.assertNotIn('required: model_path', text)
        self.assertIn('blank line', text)
        self.assertIn('Do not wrap --prompts', text)

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
            '--model-type ltx --dtype bfloat16',
            '--wan-boundary-ratios 0.9',
            '--prompts "a fox"',
        ))
        self.assertFalse(report['ok'], report)
        self.assertIn('wan', _messages(report).lower())

    def test_wan_text_to_video_ok(self):
        report = check_config(_cfg(
            'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            '--model-type wan --dtype bfloat16',
            '--video-lengths 2',
            '--prompts "a fox runs"',
        ))
        self.assertTrue(report['ok'], report['errors'])


if __name__ == '__main__':
    unittest.main()
