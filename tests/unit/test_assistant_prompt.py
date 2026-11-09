import unittest

from dgenerate.assistant.corpus import Chunk
from dgenerate.assistant.models import (
    REPLY_RESERVE, answer_text, context_tokens_needed, prompt_token_estimate, reply_token_limit)
from dgenerate.assistant.prompt import (
    MODEL_TABLE,
    extract_config,
    first_draft_recipes,
    lift_comments_between_options,
    offload_plan,
    prefer_weighting_examples,
    repair_message,
    reply_has_config,
    system_message,
    take_draft,
    user_message,
    wants_prompt_weighting,
)


def _chunk(source: str, text: str = '--prompts "a fox"') -> Chunk:
    return Chunk(id=source, kind='example', title=source, source=source, text=text)


class _FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks


class TestAnswerText(unittest.TestCase):

    def test_config_after_a_closed_thought(self):
        reply = '<think>plan the file</think>\n```dgen\nblack-forest-labs/FLUX.1-dev\n```'
        self.assertIn('FLUX.1-dev', answer_text(reply))
        self.assertNotIn('<think>', answer_text(reply))

    def test_unfinished_thought_is_not_a_config(self):
        self.assertEqual(answer_text('<think>still planning the invocation'), '')

    def test_config_inside_the_thought_is_kept(self):
        reply = '<think>\n```dgen\nstabilityai/stable-diffusion-xl-base-1.0\n```\n</think>'
        self.assertIn('stable-diffusion-xl-base-1.0', answer_text(reply))

    def test_reasoning_channel_holds_the_config(self):
        reasoning = '```dgen\nblack-forest-labs/FLUX.1-schnell\n```'
        self.assertIn('FLUX.1-schnell', answer_text('', reasoning))

    def test_reply_limit_uses_the_whole_window(self):
        self.assertEqual(reply_token_limit(32768, 0), -1)

    def test_reply_limit_respects_a_ceiling(self):
        self.assertEqual(reply_token_limit(32768, 4096), 4096)

    def test_reply_limit_does_not_exceed_the_window(self):
        self.assertEqual(reply_token_limit(8192, 9000), 8192)

    def test_prompt_estimate_rounds_up_at_three_characters(self):
        self.assertEqual(prompt_token_estimate(''), 0)
        self.assertEqual(prompt_token_estimate('ab'), 1)
        self.assertEqual(prompt_token_estimate('abcd'), 2)

    def test_instructions_need_more_than_a_5000_token_window(self):
        from dgenerate.assistant.prompt import system_message, user_message

        needed = context_tokens_needed(
            system_message('6.0.0', check_tool=True),
            user_message('a fox', '', []),
            reserve=REPLY_RESERVE)
        self.assertGreater(needed, 5000)
        self.assertLess(needed, 32768)

    def test_default_context_leaves_room_for_a_reply(self):
        import json

        from dgenerate.assistant.catalog import DEFAULT_CTX
        from dgenerate.assistant.link_tools import tool_definitions
        from dgenerate.assistant.prompt import (
            MAX_EXAMPLE_CHARS, MAX_EXAMPLES, MAX_GUIDE_CHARS, MAX_GUIDES, MODEL_TABLE)

        # The largest first prompt: system text, tool schemas, every example and
        # guide section the retriever can attach, and the model table.
        retrieved = (
            MAX_EXAMPLES * MAX_EXAMPLE_CHARS
            + MAX_GUIDES * MAX_GUIDE_CHARS
            + len(MODEL_TABLE)
            + 8000)
        needed = context_tokens_needed(
            system_message(
                '6.0.0', link_tools=True, machine='### machine\n' + 'x' * 400, check_tool=True),
            'x' * retrieved,
            json.dumps(tool_definitions(fetch=True, check=True)),
            reserve=8192)
        self.assertLess(needed, DEFAULT_CTX)


class TestThoughtDisplay(unittest.TestCase):
    def setUp(self):
        from dgenerate.assistant.models import set_progress_styles
        set_progress_styles('<plain>', '<thought>', '<tool>', '<talk>')

    def tearDown(self):
        from dgenerate.assistant.models import set_progress_styles
        set_progress_styles()

    def test_thought_streams_and_the_config_stays_out(self):
        from io import StringIO

        from dgenerate.assistant.models import _ThoughtDisplay

        out = StringIO()
        display = _ThoughtDisplay(True, out)
        closer = '</' + 'think>'
        display.add('', 'The wallpaper should be wide. ')
        display.add('', closer + '\n```dgen\nblack-forest-labs/FLUX.1-dev\n```')
        text = display.finish()
        shown = out.getvalue()
        self.assertIn('Thinking:\n=========\n', shown)
        self.assertIn('wallpaper', shown)
        self.assertNotIn('FLUX.1-dev', shown)
        self.assertIn('FLUX.1-dev', text)

    def test_talk_before_the_fence_is_shown_and_the_config_is_not(self):
        from io import StringIO

        from dgenerate.assistant.models import _ThoughtDisplay

        out = StringIO()
        display = _ThoughtDisplay(True, out)
        display.add('plan a wide frame', '')
        display.add('', 'Using a wide frame.\n```dgen\nblack-forest-labs/FLUX.1-dev\n```')
        text = display.finish()
        shown = out.getvalue()
        self.assertLess(shown.index('<thought>'), shown.index('plan a wide frame'))
        self.assertLess(shown.index('plan a wide frame'), shown.index('<talk>'))
        self.assertLess(shown.index('<talk>'), shown.index('Using a wide frame.'))
        self.assertIn('Using a wide frame.', shown)
        self.assertNotIn('FLUX.1-dev', shown)
        self.assertNotIn('```', shown)
        self.assertIn('FLUX.1-dev', text)

    def test_a_reply_without_thinking_is_talk_until_the_fence(self):
        from io import StringIO

        from dgenerate.assistant.models import _ThoughtDisplay

        out = StringIO()
        display = _ThoughtDisplay(False, out)
        display.add('', 'Here is the config.\n```dgen\norg/model\n```')
        display.finish()
        shown = out.getvalue()
        self.assertIn('Here is the config.', shown)
        self.assertNotIn('Thinking:', shown)
        self.assertNotIn('org/model', shown)
        self.assertTrue(display.shown)

    def test_style_markers_are_whatever_was_configured(self):
        from io import StringIO

        from dgenerate.assistant.models import _ThoughtDisplay, set_progress_styles

        set_progress_styles()
        plain = StringIO()
        display = _ThoughtDisplay(True, plain)
        display.add('plan a wide frame', '')
        display.finish()
        self.assertNotIn('<thought>', plain.getvalue())
        self.assertIn('Thinking:', plain.getvalue())

        green = '\033[32m'
        set_progress_styles(thought=green)
        colored = StringIO()
        display = _ThoughtDisplay(True, colored)
        display.add('plan a wide frame', '')
        display.finish()
        self.assertIn(green, colored.getvalue())

    def test_a_config_alone_is_not_shown(self):
        from io import StringIO

        from dgenerate.assistant.models import _ThoughtDisplay

        display = _ThoughtDisplay(False, StringIO())
        display.add('', '```dgen\norg/model\n```')
        display.finish()
        self.assertFalse(display.shown)


class TestControlCode(unittest.TestCase):
    def test_backslash_escapes_become_the_named_characters(self):
        from dgenerate.assistant.cli import _control_code

        self.assertEqual(_control_code(r'\033[32m'), '\033[32m')
        self.assertEqual(_control_code(r'\x1b[1t'), '\x1b[1t')
        self.assertEqual(_control_code(r'\e[0m'), '\x1b[0m')
        self.assertEqual(_control_code('\x1b[0t'), '\x1b[0t')


class TestAssistantPrompt(unittest.TestCase):

    def test_wan_gguf_stays_with_its_parent(self):
        prompt = MODEL_TABLE
        i2v = prompt.index('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')
        i2v_gguf = prompt.index('wan2.1-i2v-14b-480p-Q4_K_S.gguf')
        t2v_gguf = prompt.index('wan2.1-t2v-14b-Q4_K_S.gguf')
        self.assertLess(i2v, i2v_gguf)
        self.assertLess(i2v_gguf, t2v_gguf)
        self.assertIn('Wan-AI/Wan2.1-T2V-14B-Diffusers', prompt[t2v_gguf:t2v_gguf + 400])
        self.assertIn('A Wan GGUF or FP8 file loads only on the parent named with it.', prompt)
        self.assertLess(i2v, prompt.index('wan2.1_i2v_480p_14B_fp8_scaled.safetensors'))
        self.assertLess(
            prompt.index('Wan-AI/Wan2.1-T2V-14B-Diffusers'),
            prompt.index('wan2.1_t2v_14B_fp8_scaled.safetensors'))

    def test_quant_file_stays_in_its_bullet(self):
        prompt = MODEL_TABLE
        bullets = prompt.split('\n- ')
        pairs = (
            ('black-forest-labs/FLUX.1-schnell', 'flux1-schnell-Q4_K_S.gguf'),
            ('black-forest-labs/FLUX.1-dev', 'flux1-dev-fp8.safetensors'),
            ('black-forest-labs/FLUX.1-Fill-dev', 'flux1-fill-dev-Q4_K_S.gguf'),
            ('black-forest-labs/FLUX.1-Kontext-dev', 'flux1-kontext-dev-Q4_K_S.gguf'),
            ('krea/Krea-2-Turbo', 'krea2_turbo_nvfp4.safetensors'),
            ('krea/Krea-2-Raw', 'Krea-2-Base-Q4_K_M.gguf'),
            ('FLUX.2-klein-9B', 'flux-2-klein-9b-Q4_K_S.gguf'),
            ('FLUX.2-klein-base-9B', 'flux-2-klein-base-9b-Q4_K_S.gguf'),
            ('Tongyi-MAI/Z-Image-Turbo', 'z_image_turbo_int8_convrot.safetensors'),
            ('Tongyi-MAI/Z-Image', 'z_image_int8_convrot.safetensors'),
            ('Qwen/Qwen-Image-Edit-2509', 'qwen_image_edit_2509_int8_convrot.safetensors'),
            ('Qwen/Qwen-Image-2.1-Turbo', 'qwen_image_2.1_turbo_int8_convrot.safetensors'),
            ('Qwen/Qwen-Image-2.1', 'qwen_image_2.1_Q4_0.gguf'),
            ('stable-diffusion-3-medium-diffusers', 'sd3_medium-Q4_K_S.gguf'),
            ('stable-diffusion-3.5-large', 'sd3.5_large_fp8_scaled.safetensors'),
            ('stable-diffusion-3.5-medium', 'sd3.5_medium-Q4_K_S.gguf'),
            ('HunyuanImage-2.1-Distilled-Diffusers', 'hunyuanimage2.1_distilled_fp8_e4m3fn.safetensors'),
            ('HunyuanImage-2.1-Diffusers', 'hunyuanimage2.1-q4_k_m.gguf'),
            ('480p_t2v_distilled', 'hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors'),
            ('480p_t2v', 'hunyuanvideo1.5_480p_t2v-Q4_K_S.gguf'),
            ('Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers', 'kandinsky5Pro_t2v_sft_5s-Q4_K_S.gguf'),
            ('ltx-2.5-22b-distilled-transformer-nvfp4.safetensors', 'LTX-2.5-Distilled-Q4_K_S.gguf'),
        )
        for parent, filename in pairs:
            hits = [bullet for bullet in bullets if parent in bullet and filename in bullet]
            self.assertTrue(hits, f'{filename} is not with {parent}')
        self.assertLess(
            prompt.index('hunyuanvideo1.5_480p_t2v-Q4_K_S.gguf'),
            prompt.index('HunyuanVideo-1.5-Diffusers-480p_i2v'))
        self.assertLess(
            prompt.index('kandinsky5Pro_t2v_sft_5s-Q4_K_S.gguf'),
            prompt.index('Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers'))
        t2v22 = prompt.index('Wan-AI/Wan2.2-T2V-A14B-Diffusers')
        i2v22 = prompt.index('Wan-AI/Wan2.2-I2V-A14B-Diffusers')
        self.assertLess(t2v22, prompt.index('Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf'))
        self.assertLess(prompt.index('Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf'), i2v22)
        self.assertLess(i2v22, prompt.index('Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf'))
        self.assertLess(
            prompt.index('Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf'),
            prompt.index('Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf'))

    def test_wants_prompt_weighting(self):
        self.assertTrue(wants_prompt_weighting('use prompt weighting'))
        self.assertTrue(wants_prompt_weighting('sd-embed on the creatures'))
        self.assertFalse(wants_prompt_weighting('flux dev gguf wallpaper'))

    def test_prefer_flux_weighting_example(self):
        flux = _chunk('examples/flux/prompt_weighting/flux-sd-embed-config.dgen')
        sd = _chunk('examples/stablediffusion/prompt_weighting/sd-embed-config.dgen')
        gguf = _chunk('examples/flux/gguf/flux-dev-config.dgen')
        chosen = prefer_weighting_examples(
            'flux dev gguf and prompt weighting',
            [gguf],
            _FakeIndex([sd, flux]),
        )
        self.assertEqual(chosen[0].source, flux.source)
        self.assertIn(gguf, chosen)

    def test_prefer_moves_weighting_example_first(self):
        flux = _chunk('examples/flux/prompt_weighting/flux-sd-embed-config.dgen')
        gguf = _chunk('examples/flux/gguf/flux-dev-config.dgen')
        chosen = prefer_weighting_examples(
            'use prompt weighting',
            [gguf, flux],
            _FakeIndex([]),
        )
        self.assertEqual(chosen[0].source, flux.source)

    def test_first_draft_overrides_example_wording(self):
        recipes = first_draft_recipes('flux dev gguf and prompt weighting')
        self.assertIn('Do not copy that sentence', recipes)
        self.assertIn(r'\print Set HF_TOKEN environmental variable or pass --auth-token.', recipes)
        self.assertIn('no negative prompt', recipes)
        self.assertIn('Include one HF_TOKEN early-exit at the top of the config only', recipes)
        self.assertIn('Do not add CIVIT_AI_TOKEN', recipes)
        self.assertIn('(main subject:1.3)', recipes)
        self.assertNotIn('--prompt-weighter compel', recipes)
        self.assertIn('Write --spectrum or --sada alone', recipes)
        self.assertIn('Do not copy --spectrum-warmup-steps', recipes)
        self.assertIn('--sada already selects those', system_message('6.0.0'))

    def test_public_checkpoint_omits_hf_token(self):
        sdxl = first_draft_recipes('sdxl photo of a horse')
        self.assertIn('Do not add HF_TOKEN', sdxl)
        self.assertNotIn('Include one HF_TOKEN early-exit', sdxl)
        schnell = first_draft_recipes('flux schnell wallpaper')
        self.assertIn('Do not add HF_TOKEN', schnell)

    def test_civitai_token_only_for_civitai_links(self):
        civit = first_draft_recipes('download the sdxl checkpoint from civitai')
        self.assertIn('Include one CIVIT_AI_TOKEN early-exit at the top of the config only', civit)
        self.assertIn('Do not put token=', civit)
        self.assertIn('trigger word goes in --prompts', first_draft_recipes(
            'https://civitai.com/models/1/style'))
        red = first_draft_recipes('https://civitai.red/models/1/style')
        self.assertIn('Include one CIVIT_AI_TOKEN early-exit at the top of the config only', red)
        hugging_face = first_draft_recipes('sdxl photo of a horse')
        self.assertIn('Do not add CIVIT_AI_TOKEN', hugging_face)
        self.assertNotIn('Include one CIVIT_AI_TOKEN early-exit', hugging_face)

    def test_first_draft_compel(self):
        recipes = first_draft_recipes('use compel prompt weighting')
        self.assertIn('--prompt-weighter compel', recipes)
        self.assertIn('(main subject)+', recipes)
        self.assertNotIn('no negative prompt', recipes)

    def test_offload_follows_the_family(self):
        from dgenerate.assistant.prompt import memory_problem

        plan = offload_plan(True, 16, True, True)
        self.assertIn('krea2 otherwise, and wan', plan)
        self.assertIn('motif-video', plan)
        self.assertIn('kandinsky5-image', plan)
        self.assertIn('qwen-image-21', plan)
        self.assertIn('ltx-video', plan)
        self.assertIn('32 GiB and above covers 32, 48, and 80 GiB cards.', plan)
        self.assertIn('This computer has 16.0 GiB, the 16 GiB tier.', plan)
        self.assertIn(
            'krea2, quantized transformer and quantized text encoder',
            plan)
        self.assertIn('every tier: --model-cpu-offload. No quantizer.', plan)
        self.assertRegex(plan, r'krea2 otherwise.*\n  every tier: --model-sequential-offload\. No quantizer')
        packed = memory_problem(
            'krea2', 'krea/Krea-2-Raw',
            transformer='Krea-2-Base-Q4_K_M.gguf',
            text_encoders=['Qwen3VL-4B-Instruct-Q4_K_M.gguf'],
            cuda=True, gib=16, bitsandbytes=True, sdnq=False)
        self.assertIn('--model-cpu-offload', packed)
        self.assertIsNone(memory_problem(
            'krea2', 'krea/Krea-2-Raw',
            transformer='Krea-2-Base-Q4_K_M.gguf',
            text_encoders=['Qwen3VL-4B-Instruct-Q4_K_M.gguf'],
            cpu=True,
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIn('--model-sequential-offload', memory_problem(
            'krea2', 'krea/Krea-2-Raw',
            transformer='Krea-2-Base-Q4_K_M.gguf',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIn('flux, flux-fill, flux-kontext, full checkpoint', plan)
        self.assertIn('16 GiB, 24 GiB, 32 GiB and above: --quantizer bnb;bits=4', plan)
        self.assertIn('flux, flux-fill, flux-kontext, control lora', plan)
        self.assertIn('qwen-image, qwen-image-edit', plan)
        self.assertIn('flux2 dev, flux2 klein 9b, flux2-klein-kv, full checkpoint', plan)
        self.assertIn('flux2 klein 4b, full checkpoint', plan)
        self.assertIn('z-image, full checkpoint', plan)
        self.assertIn('sd3, sd3-pix2pix, full checkpoint', plan)
        self.assertIn('--quantizer-map text_encoder', plan)
        self.assertIn('--quantizer-map text_encoder_3', plan)
        self.assertIn('kolors', plan)
        self.assertIn('s-cascade', plan)
        self.assertIn('every tier: --model-cpu-offload and --second-model-cpu-offload. No quantizer.', plan)
        self.assertIn('every tier: no quantizer and no offload.', plan)
        self.assertIn('That is the baseline.', plan)
        self.assertIn('can still exceed 16 GiB at a larger resolution, such as 2048', plan)
        self.assertIn('or at a longer video length', plan)
        self.assertIn('--vae-tiling', plan)
        self.assertIn('calls for a stronger one', plan)

        roomy = offload_plan(True, 32, True, True)
        self.assertIn('This computer has 32.0 GiB, the 32 GiB and above tier.', roomy)
        self.assertIn('32 GiB and above: --quantizer bnb;bits=8', roomy)
        self.assertIn('24 GiB, 32 GiB and above: no offload and no quantizer', roomy)
        self.assertIn('32 GiB and above: --quantizer bnb;bits=4', roomy)
        self.assertIn('qwen-image-21', roomy)
        self.assertIn('--model-sequential-offload. No quantizer, including a GGUF or FP8 file.', roomy)

        tight = offload_plan(True, 10, True, False)
        self.assertIn('This computer has 10.0 GiB, the 12 GiB tier.', tight)
        self.assertIn('8 GiB, 12 GiB: --model-sequential-offload', tight)
        self.assertIn('bits4-quant-type=nf4', tight)

        cpu = offload_plan(False, 64, True, True)
        self.assertNotIn('--quantizer bnb', cpu)
        self.assertIn('sd, pix2pix, sdxl, sdxl-pix2pix', cpu)
        self.assertIn('no quantizer', cpu)

        klein = memory_problem(
            'flux2', 'black-forest-labs/FLUX.2-klein-4B',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False)
        self.assertIn('--quantizer bnb;bits=4', klein)
        self.assertIsNone(memory_problem(
            'flux2', 'black-forest-labs/FLUX.2-klein-4B', quantizer='bnb;bits=4',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIsNone(memory_problem(
            'flux2', 'black-forest-labs/FLUX.2-dev', sequential=True,
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIn('must not be quantized', memory_problem(
            'z-image', 'Tongyi-MAI/Z-Image-Turbo', quantizer='bnb;bits=4', sequential=True,
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        base = memory_problem(
            'flux2', 'black-forest-labs/FLUX.2-klein-base-4B',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False)
        self.assertIn('--quantizer bnb;bits=4', base)
        self.assertNotIn('sequential-offload', base)
        self.assertIsNone(memory_problem(
            'flux2', 'black-forest-labs/FLUX.2-klein-base-4B', quantizer='bnb;bits=4',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIn('must not be quantized', memory_problem(
            'hydit', 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers', quantizer='bnb;bits=4',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIn('must not use offload', memory_problem(
            'hydit', 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers', sequential=True,
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))
        self.assertIsNone(memory_problem(
            'hydit', 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
            cuda=True, gib=16, bitsandbytes=True, sdnq=False))

    def test_ltx25_gemma4_gguf_names_the_real_quants(self):
        self.assertIn('gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf', MODEL_TABLE)
        self.assertIn('gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf', MODEL_TABLE)
        self.assertIn('gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf', MODEL_TABLE)
        self.assertIn('There is no Q8_0', MODEL_TABLE)
        system = system_message('6.0.0', link_tools=True)
        self.assertIn('Do not change its quant suffix', system)
        self.assertIn('fp8_scaled Comfy file is not an fp8_e4m3fn', system)
        self.assertIn('unless the fetch_link tool lists the requested quant', system)
        self.assertIn('Call fetch_link once on that repo', system)
        self.assertIn('except once on the repo of the closest example', system)
        self.assertIn('lists only the files already in the Hugging Face cache', system)
        bare = system_message('6.0.0', link_tools=False)
        self.assertNotIn('call fetch_link', bare)
        self.assertIn('unless the fetch_link tool lists the requested quant', bare)

    def test_resolution_is_in_the_system_prompt(self):
        plan = offload_plan(True, 16, True, False)
        self.assertIn('each tier line is for that model\'s usual size', plan)
        self.assertIn('A larger resolution or a longer video uses more memory', plan)
        self.assertIn('--model-sequential-offload is the strongest', plan)
        self.assertIn('On 24 GiB and above the tier line often still fits', plan)
        self.assertIn('Video models do not take --vae-tiling', plan)
        system = system_message('6.0.0', machine='### machine\n' + plan)
        self.assertIn('example size and video length, which is the baseline', system)
        self.assertIn('A larger resolution or a longer video uses more memory', system)
        for request in (
            'krea 2 at 2048 with the int8 comfy quant',
            'flux dev at 2048x2048',
            'sdxl wallpaper, 4k',
            'wan clip at 1280x720',
            'qwen image 1536x1536',
        ):
            self.assertNotIn('This request is above the example size', first_draft_recipes(request), request)

    def test_first_draft_states_the_memory_flags(self):
        klein = first_draft_recipes(
            'sprite sheet with https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora')
        self.assertNotIn('Memory flags for this request', klein)
        plan = offload_plan(True, 16, True, False)
        self.assertIn('flux2 klein 4b, full checkpoint', plan)
        self.assertIn('16 GiB, 24 GiB, 32 GiB and above: --quantizer bnb;bits=4', plan)
        self.assertIn('flux2 dev, flux2 klein 9b, flux2-klein-kv, full checkpoint', plan)
        self.assertIn('8 GiB, 12 GiB, 16 GiB, 24 GiB: --model-sequential-offload', plan)

    def test_a_comment_between_options_is_moved_above_the_model(self):
        text = extract_config(
            '```dgen\n'
            'Tongyi-MAI/Z-Image-Turbo\n'
            '--model-type z-image\n'
            '# do not leave this between options\n'
            '--prompts "a horse"\n'
            '```',
            '6.0.0')
        lines = text.splitlines()
        self.assertLess(
            lines.index('# do not leave this between options'),
            lines.index('Tongyi-MAI/Z-Image-Turbo'))
        model = lines.index('Tongyi-MAI/Z-Image-Turbo')
        self.assertEqual(lines[model + 1], '--model-type z-image')
        self.assertEqual(lines[model + 2], '--prompts "a horse"')

    def test_comments_stay_with_their_own_invocation(self):
        text = lift_comments_between_options(
            '#! /usr/bin/env dgenerate --file\n'
            '#! dgenerate 6.0.0\n'
            '\n'
            '# generate\n'
            'stabilityai/stable-diffusion-xl-base-1.0\n'
            '--model-type sdxl\n'
            '# between the first options\n'
            '--prompts "a fox"\n'
            '\n'
            '# upscale the result\n'
            '{% for image in last_images %}\n'
            'stabilityai/stable-diffusion-x4-upscaler\n'
            '--model-type upscaler-x4\n'
            '# between the second options\n'
            '--image-seeds {{ quote(image) }}\n'
            '--prompts "a fox"\n'
            '{% endfor %}\n')
        lines = text.splitlines()
        first_model = lines.index('stabilityai/stable-diffusion-xl-base-1.0')
        second_model = lines.index('stabilityai/stable-diffusion-x4-upscaler')
        self.assertLess(lines.index('# between the first options'), first_model)
        self.assertLess(first_model, second_model)
        self.assertLess(lines.index('# upscale the result'), second_model)
        self.assertGreater(lines.index('# upscale the result'), lines.index('--prompts "a fox"'))
        self.assertLess(lines.index('# between the second options'), second_model)
        self.assertGreater(lines.index('# between the second options'), lines.index('{% for image in last_images %}'))
        self.assertEqual(lines[second_model + 1], '--model-type upscaler-x4')
        self.assertEqual(lines[second_model + 2], '--image-seeds {{ quote(image) }}')
        self.assertEqual(lines[second_model + 3], '--prompts "a fox"')
        self.assertEqual(lines[0], '#! /usr/bin/env dgenerate --file')
        self.assertEqual(lines[1], '#! dgenerate 6.0.0')

    def test_repair_keeps_the_draft_when_the_reply_is_prose(self):
        draft = (
            '#! /usr/bin/env dgenerate --file\n'
            '#! dgenerate 6.0.0\n'
            '\n'
            'krea/Krea-2-Raw\n'
            '--model-type krea2\n'
            '--text-encoders "Qwen3VLModel;model=encoder.gguf"\n')
        prose = (
            'I don\'t have the original config text in this conversation. '
            'The error mentions `--text-encoders`.\n')
        self.assertFalse(reply_has_config(prose))
        kept, used = take_draft(draft, prose, '6.0.0')
        self.assertFalse(used)
        self.assertIn('--text-encoders', kept)
        self.assertNotIn("don't have", kept)

    def test_repair_message_includes_the_rejected_config(self):
        class Index:
            def argument(self, option):
                return None

        draft = 'krea/Krea-2-Raw\n--text-encoders "Qwen3VLModel;model=encoder.gguf"\n'
        text = repair_message(
            ['--text-encers is not a dgenerate option, did you mean --text-encoders?'],
            [], Index(), draft)
        self.assertIn('```dgen', text)
        self.assertIn('krea/Krea-2-Raw', text)
        self.assertIn('--text-encoders', text)
        self.assertIn('complete corrected config', text)

    def test_z_image_words_get_the_flag_on_its_own_line(self):
        plan = offload_plan(True, 16, True, False)
        self.assertIn('z-image, full checkpoint', plan)
        self.assertIn('8 GiB, 12 GiB, 16 GiB: --model-sequential-offload. Do not pass --quantizer', plan)
        self.assertIn('24 GiB, 32 GiB and above: no offload and no quantizer', plan)

    def test_distilled_klein_9b_requests_the_token(self):
        text = first_draft_recipes(
            'https://civitai.com/models/816857/neon-retrowave-by-chronoknight-flux2-klein-9b-flux-and-il')
        self.assertIn('black-forest-labs/FLUX.2-klein-9B', text)
        self.assertIn('This request uses a gated checkpoint', text)
        public = first_draft_recipes('black-forest-labs/FLUX.2-klein-4B')
        self.assertNotIn('This request uses a gated checkpoint', public)
        for gated in (
            'Flux.1 D',
            'stabilityai/stable-diffusion-3.5-large',
            'sd 3.5',
            'ltx2',
            'deep floyd',
            'Lightricks/LTX-Video',
        ):
            self.assertIn('gated checkpoint', first_draft_recipes(gated), gated)
        for public_name in (
            'krea/Krea-2-Raw',
            'black-forest-labs/FLUX.2-klein-base-4B',
            'FLUX.1-schnell',
        ):
            self.assertNotIn(
                'This request uses a gated checkpoint', first_draft_recipes(public_name), public_name)

    def test_model_is_told_after_the_examples(self):
        system = system_message('6.0.0')
        self.assertIn('you are writing a config the user will run', system)
        self.assertIn('Machine section', system)
        self.assertIn('Do not write a total_memory check', system)
        fitted = system_message('6.0.0', machine='### machine\nFor a large checkpoint on this computer, write --model-sequential-offload and no other offload flag.')
        self.assertIn('--model-sequential-offload and no other offload flag', fitted)
        msg = user_message('use flux and prompt weighting', 'retrieved examples say this example', [])
        self.assertGreater(msg.index('Do not copy that sentence'), msg.index('this example'))
        self.assertEqual(msg.count('### write the config from these rules'), 1)

    def test_edit_mode_keeps_the_open_config(self):
        editor = '#! /usr/bin/env dgenerate --file\nstable-diffusion-v1-5/stable-diffusion-v1-5\n--prompts "a fox"\n'
        msg = user_message('make the fox red', 'retrieved examples', [], editor=editor)
        self.assertIn('Current config in the editor', msg)
        self.assertIn('Do not wrap a line', msg)
        self.assertIn('--prompts "a fox"', msg)
        self.assertIn('make the fox red', msg)
        self.assertNotIn('### write the config from these rules', msg)


class TestAssistantSampling(unittest.TestCase):
    def test_omitted_flags_follow_the_model_card(self):
        from dgenerate.assistant.cli import create_parser, sampling_from_args

        parser = create_parser('assistant')
        plain = sampling_from_args(parser.parse_args(['a cat']))
        self.assertEqual(plain, {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'presence_penalty': 1.5,
            'repeat_penalty': 1.0,
        })
        thinking = sampling_from_args(parser.parse_args(['--think', 'a cat']))
        self.assertEqual(thinking['temperature'], 1.0)
        self.assertEqual(thinking['top_p'], 0.95)
        self.assertEqual(thinking['top_k'], 20)
        self.assertEqual(thinking['min_p'], 0.0)
        self.assertEqual(thinking['presence_penalty'], 0.0)
        self.assertEqual(thinking['repeat_penalty'], 1.0)

    def test_a_passed_flag_overrides_only_that_value(self):
        from dgenerate.assistant.cli import create_parser, sampling_from_args

        parser = create_parser('assistant')
        sampling = sampling_from_args(parser.parse_args([
            '--think', '--temperature', '0.2', '--top-k', '10', 'a cat']))
        self.assertEqual(sampling['temperature'], 0.2)
        self.assertEqual(sampling['top_k'], 10)
        self.assertEqual(sampling['top_p'], 0.95)
        self.assertEqual(sampling['presence_penalty'], 0.0)

    def test_completion_sends_the_recommended_values(self):
        from dgenerate.assistant.models import ChatModel

        chat = ChatModel.__new__(ChatModel)
        chat.think = False
        chat.effort = 'medium'
        seen = {}

        class Server:
            def handle_chat_completions(self, body, on_chunk):
                seen['body'] = body

        chat._server = Server()
        chat.complete([{'role': 'user', 'content': 'hi'}])
        body = seen['body']
        self.assertEqual(body['temperature'], 0.7)
        self.assertEqual(body['top_p'], 0.8)
        self.assertEqual(body['top_k'], 20)
        self.assertEqual(body['min_p'], 0.0)
        self.assertEqual(body['presence_penalty'], 1.5)
        self.assertEqual(body['repeat_penalty'], 1.0)
        self.assertFalse(body['chat_template_kwargs']['enable_thinking'])

        chat.think = True
        chat.complete([{'role': 'user', 'content': 'hi'}], temperature=0.2)
        body = seen['body']
        self.assertEqual(body['temperature'], 0.2)
        self.assertEqual(body['top_p'], 0.95)
        self.assertEqual(body['presence_penalty'], 0.0)
        self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(body['chat_template_kwargs']['reasoning_effort'], 'medium')


class TestAssistantWithoutXllamacpp(unittest.TestCase):
    def test_missing_xllamacpp_stops_before_loading_a_model(self):
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        import dgenerate.assistant.cli as cli

        err = io.StringIO()
        with patch.object(cli._catalog, 'xllamacpp_installed', return_value=False), redirect_stderr(err):
            code = cli.main(['a fox eating waffles'])
        self.assertEqual(code, 1)
        self.assertIn('xllamacpp is not installed', err.getvalue())
        self.assertNotIn('Loading the embedding model', err.getvalue())

    def test_small_context_stops_before_the_embedding_model(self):
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        import dgenerate.assistant.cli as cli

        err = io.StringIO()
        with patch.object(cli._catalog, 'xllamacpp_installed', return_value=True), redirect_stderr(err):
            code = cli.main(['--ctx', '5000', 'a fox eating waffles'])
        text = err.getvalue()
        self.assertEqual(code, 1)
        self.assertIn('context window is 5000 tokens', text)
        self.assertIn('before any examples are retrieved', text)
        self.assertNotIn('Loading the embedding model', text)
        self.assertNotIn('Loading Qwen', text)


if __name__ == '__main__':
    unittest.main()
