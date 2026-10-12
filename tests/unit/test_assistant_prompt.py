import unittest

from dgenerate.assistant.corpus import Chunk
from dgenerate.assistant.models import answer_text, reply_token_limit
from dgenerate.assistant.prompt import (
    first_draft_recipes,
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
        self.assertIn('Include the HF_TOKEN early-exit', recipes)
        self.assertIn('Do not add CIVIT_AI_TOKEN', recipes)
        self.assertIn('(main subject:1.3)', recipes)
        self.assertNotIn('--prompt-weighter compel', recipes)

    def test_public_checkpoint_omits_hf_token(self):
        sdxl = first_draft_recipes('sdxl photo of a horse')
        self.assertIn('Do not add HF_TOKEN', sdxl)
        self.assertNotIn('Include the HF_TOKEN early-exit', sdxl)
        schnell = first_draft_recipes('flux schnell wallpaper')
        self.assertIn('Do not add HF_TOKEN', schnell)

    def test_civitai_token_only_for_civitai_links(self):
        civit = first_draft_recipes('download the sdxl checkpoint from civitai')
        self.assertIn('Include the CIVIT_AI_TOKEN early-exit', civit)
        hugging_face = first_draft_recipes('sdxl photo of a horse')
        self.assertIn('Do not add CIVIT_AI_TOKEN', hugging_face)
        self.assertNotIn('Include the CIVIT_AI_TOKEN early-exit', hugging_face)

    def test_first_draft_compel(self):
        recipes = first_draft_recipes('use compel prompt weighting')
        self.assertIn('--prompt-weighter compel', recipes)
        self.assertIn('(main subject)+', recipes)
        self.assertNotIn('no negative prompt', recipes)

    def test_model_is_told_after_the_examples(self):
        system = system_message('6.0.0')
        self.assertIn('you are writing a config the user will run', system)
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
