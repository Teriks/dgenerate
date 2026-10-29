import unittest

from dgenerate.assistant.corpus import Chunk
from dgenerate.assistant.models import answer_text, reply_token_limit
from dgenerate.assistant.prompt import (
    extract_config,
    first_draft_recipes,
    lift_comments_between_options,
    offload_plan,
    prefer_weighting_examples,
    system_message,
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


class TestThoughtDisplay(unittest.TestCase):
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
        self.assertIn('wallpaper', shown)
        self.assertNotIn('FLUX.1-dev', shown)
        self.assertIn('FLUX.1-dev', text)


class TestAssistantPrompt(unittest.TestCase):

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
        self.assertIn('krea2, wan', plan)
        self.assertIn('motif-video', plan)
        self.assertIn('kandinsky5-image', plan)
        self.assertIn('qwen-image-21', plan)
        self.assertIn('ltx-video', plan)
        self.assertRegex(plan, r'krea2.*--model-sequential-offload\. No quantizer')
        self.assertIn('flux, flux-fill, flux-kontext, full checkpoint: --quantizer bnb;bits=4.', plan)
        self.assertIn('flux, flux-fill, flux-kontext, control lora: --model-sequential-offload.', plan)
        self.assertIn('qwen-image, qwen-image-edit', plan)
        self.assertIn('flux2 dev, flux2 klein 9b, flux2-klein-kv, full checkpoint: --model-sequential-offload.', plan)
        self.assertIn('flux2 klein 4b, full checkpoint: --quantizer bnb;bits=4.', plan)
        self.assertIn('z-image, full checkpoint: --model-sequential-offload.', plan)
        self.assertIn('sd3, sd3-pix2pix, full checkpoint: --model-sequential-offload.', plan)
        self.assertIn('--quantizer-map text_encoder', plan)
        self.assertIn('--quantizer-map text_encoder_3', plan)
        self.assertIn('kolors: --quantizer bnb;bits=4.', plan)
        self.assertIn('s-cascade: --model-cpu-offload and --second-model-cpu-offload.', plan)
        self.assertIn('hydit: no quantizer and no offload.', plan)

        roomy = offload_plan(True, 32, True, True)
        self.assertIn('sd3, sd3-pix2pix, full checkpoint: --quantizer bnb;bits=8.', roomy)
        self.assertIn('z-image, full checkpoint: no offload and no quantizer.', roomy)
        self.assertIn('qwen-image, qwen-image-edit, qwen-image-layered, full checkpoint: --quantizer bnb;bits=4.', roomy)
        self.assertIn('qwen-image-21', roomy)
        self.assertIn('--model-sequential-offload. No quantizer, including a GGUF or FP8 file.', roomy)

        tight = offload_plan(True, 10, True, False)
        self.assertIn('flux, flux-fill, flux-kontext, full checkpoint: --model-sequential-offload.', tight)
        self.assertIn('flux2 klein 4b, full checkpoint: --model-sequential-offload.', tight)
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

    def test_first_draft_states_the_memory_flags(self):
        machine = (True, 16.0, True, False)
        klein = first_draft_recipes(
            'sprite sheet with https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora',
            machine=machine)
        self.assertIn('Flux.2 Klein 4B full checkpoint (--model-type flux2, not flux): write --quantizer bnb;bits=4.', klein)
        self.assertIn('no {% if have_cuda() %}', klein)
        self.assertIn('no total_memory', klein)
        dev = first_draft_recipes('flux.2 dev photo of a horse', machine=machine)
        self.assertIn('Flux.2 dev, Klein 9B, or Klein KV full checkpoint: write --model-sequential-offload.', dev)
        self.assertNotIn('Klein 4B full checkpoint', dev)

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

    def test_z_image_words_get_the_flag_on_its_own_line(self):
        text = first_draft_recipes(
            'can you depict yourself using z image turbo',
            machine=(True, 16.0, True, False))
        self.assertIn('Z-Image full checkpoint: write --model-sequential-offload.', text)
        self.assertIn('Do not pass --quantizer.', text)
        self.assertIn('no \\setp gpu_memory_gib', text)
        self.assertIn('no {% if have_cuda() %}', text)

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


if __name__ == '__main__':
    unittest.main()
