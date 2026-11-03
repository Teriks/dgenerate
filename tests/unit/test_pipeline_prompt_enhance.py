import re
import types
import unittest
import unittest.mock

import torch

import dgenerate.arguments as _arguments
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.promptenhance as _promptenhance
import dgenerate.prompt as _prompt
import dgenerate.renderloopconfig as _renderloopconfig


def _config(**values):
    config = _renderloopconfig.RenderLoopConfig()
    for name, value in values.items():
        setattr(config, name, value)
    return config


class TestPipelinePromptEnhance(unittest.TestCase):
    def test_flag_is_optional_text(self):
        parser = _arguments._create_parser(add_help=False, prints_usage=False)[0]
        bare = parser.parse_args(['model', '--pipeline-prompt-enhance'])
        self.assertIs(bare.pipeline_prompt_enhance, True)
        custom = parser.parse_args(
            ['model', '--pipeline-prompt-enhance', 'be precise'])
        self.assertEqual(custom.pipeline_prompt_enhance, 'be precise')
        omitted = parser.parse_args(['model'])
        self.assertIsNone(omitted.pipeline_prompt_enhance)

    def test_ltx_prompt_enhancer_is_an_ltx2_uri(self):
        parser = _arguments._create_parser(add_help=False, prints_usage=False)[0]
        parsed = parser.parse_args(
            ['model', '--ltx-prompt-enhancer', 'google/gemma-4-E2B-it'])
        self.assertEqual(parsed.ltx_prompt_enhancer, 'google/gemma-4-E2B-it')
        rejected = _config(
            model_path='org/sd',
            model_type=_enums.ModelType.SD,
            ltx_prompt_enhancer='google/gemma-4-E2B-it')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            rejected.check()
        self.assertIn('ltx2', str(raised.exception))

    def test_enhance_seeds_are_their_own_combination(self):
        config = _config(
            model_path='org/z-image',
            model_type=_enums.ModelType.Z_IMAGE,
            seeds=[1, 2],
            pipeline_prompt_enhance=True,
            pipeline_prompt_enhance_seeds=[9])
        steps = list(config.iterate_diffusion_args())
        self.assertEqual([step.seed for step in steps], [1, 2])
        self.assertEqual([step.pipeline_prompt_enhance_seed for step in steps], [9, 9])
        counted = _config(
            model_path='org/z-image',
            model_type=_enums.ModelType.Z_IMAGE,
            prompts=[_prompt.Prompt()],
            seeds=[1],
            guidance_scales=[0],
            inference_steps=[9],
            pipeline_prompt_enhance_seeds=[4, 5])
        counted.prompts[0].positive = 'a nice horse'
        self.assertEqual(counted.calculate_generation_steps(), 2)
        self.assertEqual(len(list(counted.iterate_diffusion_args())), 2)

    def test_orphan_pipeline_prompt_options_imply_enhance(self):
        for kwargs in (
            {'pipeline_prompt_enhance_seeds': [4]},
            {'pipeline_prompt_enhance_top_p': 0.95},
            {'pipeline_prompt_enhance_temperature': 0.3},
        ):
            config = _config(
                model_path='org/z-image',
                model_type=_enums.ModelType.Z_IMAGE,
                **kwargs)
            config.check()
            self.assertIsNone(config.pipeline_prompt_enhance, kwargs)
            self.assertTrue(config.is_pipeline_prompt_enhance_enabled(), kwargs)
            self.assertEqual(
                config.calculate_generation_steps(),
                len(list(config.iterate_diffusion_args())))

    def test_wrapper_implies_pipeline_prompt_enhance(self):
        from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper

        class Wrapper:
            model_type = _enums.ModelType.Z_IMAGE

        args = _pipeline_args()
        args.pipeline_prompt_enhance = None
        args.pipeline_prompt_enhance_top_p = 0.9
        DiffusionPipelineWrapper._auto_pipeline_prompt_enhance_check(Wrapper(), args)
        self.assertIs(args.pipeline_prompt_enhance, True)
        self.assertEqual(args.pipeline_prompt_enhance_top_p, 0.9)

    def test_orphan_pipeline_prompt_options_still_reject_unsupported_model(self):
        config = _config(
            model_path='org/sd',
            model_type=_enums.ModelType.SD,
            pipeline_prompt_enhance_top_p=0.9)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()
        self.assertIsNone(config.pipeline_prompt_enhance)

    def test_rejected_on_unsupported_model_type(self):
        rejected = _config(
            model_path='org/sd',
            model_type=_enums.ModelType.SD,
            pipeline_prompt_enhance=True)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            rejected.check()
        self.assertNotIn('kolors', str(raised.exception))
        self.assertIn('ltx2', str(raised.exception))
        self.assertTrue(_enums.model_type_supports_pipeline_prompt_enhance('krea2'))
        self.assertTrue(_enums.model_type_supports_pipeline_prompt_enhance('kandinsky5-image'))
        self.assertTrue(_enums.model_type_supports_pipeline_prompt_enhance('kandinsky5-video'))
        self.assertIn('krea2', _enums.PIPELINE_PROMPT_ENHANCE_MODEL_TYPES)
        self.assertFalse(_enums.model_type_supports_pipeline_prompt_enhance(_enums.ModelType.SD))
        self.assertFalse(_enums.model_type_supports_pipeline_prompt_enhance(_enums.ModelType.KOLORS))
        qwen = _promptenhance.sampling_for(_enums.ModelType.Z_IMAGE)
        mistral = _promptenhance.sampling_for(_enums.ModelType.FLUX2, 'Mistral3ForConditionalGeneration')
        klein = _promptenhance.sampling_for(_enums.ModelType.FLUX2, 'Qwen3ForCausalLM')
        self.assertEqual(qwen.presence_penalty, 1.5)
        self.assertEqual(klein.top_k, qwen.top_k)
        self.assertNotEqual(mistral.top_p, qwen.top_p)

    def test_blank_system_prompt_means_auto(self):
        config = _config(
            model_path='org/z-image',
            model_type=_enums.ModelType.Z_IMAGE,
            pipeline_prompt_enhance='   ')
        try:
            config.check()
        except _renderloopconfig.RenderLoopConfigError as error:
            self.assertNotIn('pipeline_prompt_enhance', str(error))
        self.assertIs(config.pipeline_prompt_enhance, True)

    def test_hyimage_refiner_keeps_the_image_prompt(self):
        name = 'Qwen2_5_VLForConditionalGeneration'
        prompt = _promptenhance.auto_system_prompt(
            _enums.ModelType.HYIMAGE_21_REFINER, True, encoder_name=name)
        self.assertIn('image generation model', prompt)
        self.assertIn('in the user\'s language', prompt)
        self.assertNotIn('editing', prompt)
        self.assertTrue(_enums.model_type_supports_pipeline_prompt_enhance('hyimage-21-refiner'))

    def test_kandinsky5_uses_edit_and_shot_prompts(self):
        name = 'Qwen2_5_VLForConditionalGeneration'
        image = _promptenhance.auto_system_prompt(
            _enums.ModelType.KANDINSKY5_IMAGE, False, encoder_name=name)
        edit = _promptenhance.auto_system_prompt(
            _enums.ModelType.KANDINSKY5_IMAGE, True, encoder_name=name)
        video = _promptenhance.auto_system_prompt(
            _enums.ModelType.KANDINSKY5_VIDEO, False, encoder_name=name)
        i2v = _promptenhance.auto_system_prompt(
            _enums.ModelType.KANDINSKY5_VIDEO, True, encoder_name=name)
        self.assertIn('image generation model', image)
        self.assertIn('in the user\'s language', image)
        self.assertIn('editing instruction', edit)
        self.assertNotIn('<image1>', edit)
        self.assertIn('video generation model', video)
        self.assertIn('how the camera moves', video)
        self.assertNotIn('first frame', video)
        self.assertIn('first frame', i2v)
        self.assertIn('description of the shot', i2v)

    def test_condition_images_follow_the_reference_slot(self):
        picture = object()
        args = types.SimpleNamespace(images=None, reference_images=[picture])
        seen = _promptenhance._condition_images(
            args, _enums.ModelType.KANDINSKY5_IMAGE)
        self.assertEqual(seen, [picture])
        refiner = _promptenhance._condition_images(
            types.SimpleNamespace(images=None, reference_images=[picture]),
            _enums.ModelType.HYIMAGE_21_REFINER)
        self.assertEqual(refiner, [])

    def test_auto_prompt_uses_edit_for_qwen_edit(self):
        qwen_edit = _promptenhance.auto_system_prompt(_enums.ModelType.QWEN_IMAGE_EDIT, True)
        self.assertIn('in the user\'s language', qwen_edit)
        self.assertIn('Match the language of the user\'s request', qwen_edit)
        self.assertNotIn('Write in English', qwen_edit)
        self.assertIn('<image1>', qwen_edit)
        self.assertNotIn('150 characters', qwen_edit)
        qwen = _promptenhance.auto_system_prompt(_enums.ModelType.QWEN_IMAGE, False)
        self.assertIn('in the user\'s language', qwen)
        self.assertIn('Match the language of the user\'s request', qwen)
        self.assertNotIn('Write in English', qwen)
        self.assertIn('Never translate sign or label text', qwen)
        self.assertIn('about 80 to 150 words', qwen)
        self.assertIn('about 80 to 150 characters', qwen)
        self.assertIn('250 words', qwen)
        self.assertIn('250 characters', qwen)
        self.assertIn('Write in English', _promptenhance.T2I_SYSTEM_PROMPT)
        self.assertIn('about 80 to 150 words', _promptenhance.T2I_SYSTEM_PROMPT)
        self.assertNotIn('characters', _promptenhance.T2I_SYSTEM_PROMPT)
        self.assertIn('about 48 words', _promptenhance.EDIT_SYSTEM_PROMPT)

    def test_qwen25_vl_prompt_enhance_defaults(self):
        name = 'Qwen2_5_VLForConditionalGeneration'
        self.assertTrue(_promptenhance._brief_encoder(_enums.ModelType.QWEN_IMAGE, name))
        qwen25 = _promptenhance.auto_system_prompt(
            _enums.ModelType.QWEN_IMAGE, False, encoder_name=name)
        qwen3 = _promptenhance.auto_system_prompt(
            _enums.ModelType.QWEN_IMAGE_21, False, encoder_name='Qwen3VLForConditionalGeneration')
        self.assertNotIn('Always describe the light', qwen25)
        self.assertIn('Always describe the light', qwen3)
        sampling = _promptenhance.sampling_for(_enums.ModelType.QWEN_IMAGE, name)
        z_image = _promptenhance.sampling_for(_enums.ModelType.Z_IMAGE)
        klein = _promptenhance.sampling_for(
            _enums.ModelType.FLUX2, 'Qwen3ForCausalLM')
        self.assertEqual(sampling.temperature, 0.35)
        self.assertEqual(z_image.temperature, 0.35)
        self.assertGreater(klein.temperature, sampling.temperature)
        self.assertEqual(
            _promptenhance.generation_limits(_enums.ModelType.QWEN_IMAGE, name)[1], 32)
        self.assertIn('Match the language of the user\'s request', qwen25)
        english_prompt = (
            'A white cat asleep on a blue chair beside a window, warm evening light. '
            'A paper sign on the wall reads 「OPEN」.')
        self.assertEqual(_promptenhance._detect_prompt_language(english_prompt), 'en')
        prefixed = _promptenhance._rewrite_user_prompt(
            english_prompt, model_type=_enums.ModelType.QWEN_IMAGE, encoder_name=name)
        self.assertIn('english only', prefixed.lower())
        self.assertTrue(prefixed.endswith(english_prompt))
        chinese = '三只猫和一只狗坐在木桌旁，桌上的纸牌写着「营业中」，傍晚的窗光'
        spanish = (
            'un gato rojo dormido en una silla azul junto a una ventana. en la pared, '
            'un letrero que dice «ABIERTO»')
        self.assertEqual(_promptenhance._detect_prompt_language(spanish), 'es')

        def _fake_prefix(lang, local_files_only):
            if lang == 'en':
                return _promptenhance._QWEN25_ENGLISH_USER_TURN_PREFIX
            return f'[{lang}-prefix]\n\n'

        with unittest.mock.patch.object(
                _promptenhance, '_qwen25_translated_user_prefix', side_effect=_fake_prefix):
            chinese_prefixed = _promptenhance._rewrite_user_prompt(
                chinese, model_type=_enums.ModelType.QWEN_IMAGE, encoder_name=name)
            self.assertTrue(chinese_prefixed.startswith('[zh-prefix]'))
            spanish_prefixed = _promptenhance._rewrite_user_prompt(
                spanish, model_type=_enums.ModelType.QWEN_IMAGE, encoder_name=name)
            self.assertTrue(spanish_prefixed.startswith('[es-prefix]'))
        self.assertNotIn('english only', chinese_prefixed.lower())
        self.assertEqual(
            _promptenhance._rewrite_user_prompt(
                'A white cat', model_type=_enums.ModelType.QWEN_IMAGE_21,
                encoder_name='Qwen3VLForConditionalGeneration'),
            'A white cat')
        self.assertIn(
            'english only',
            _promptenhance._rewrite_user_prompt(
                english_prompt, model_type=_enums.ModelType.Z_IMAGE,
                encoder_name='Qwen3Model').lower())

    def test_auto_prompt_krea_and_z_image(self):
        krea = _promptenhance.auto_system_prompt(_enums.ModelType.KREA2, True)
        self.assertIn('Write in English', krea)
        self.assertNotIn('You see the input image', krea)
        self.assertNotIn('ginger', krea)
        omni = _promptenhance.auto_system_prompt(_enums.ModelType.Z_IMAGE_OMNI, True)
        self.assertIn('Write in English', omni)
        self.assertNotIn('in the user\'s language', omni)
        self.assertNotIn('You see the input image', omni)
        flux = _promptenhance.auto_system_prompt(
            _enums.ModelType.FLUX2, True, encoder_name='Mistral3ForConditionalGeneration')
        self.assertIn('You see the input image', flux)
        self.assertNotIn('<image1>', flux)
        self.assertIn('first image', flux)
        self.assertIn('Write in the language the user writes in', flux)
        self.assertIn('Match the language of the user\'s request', flux)
        self.assertIn('characters', flux)
        klein = _promptenhance.auto_system_prompt(
            _enums.ModelType.FLUX2_KLEIN_KV, False, encoder_name='Qwen3ForCausalLM')
        self.assertNotIn('You see the input image', klein)
        self.assertIn('Write in the language the user writes in', klein)
        self.assertIn('Match the language of the user\'s request', klein)
        self.assertIn('characters', klein)
        short = _promptenhance.auto_system_prompt(
            _enums.ModelType.QWEN_IMAGE, False,
            overrides=_promptenhance.PromptEnhanceOverrides(max_new_tokens=40, min_new_tokens=0))
        self.assertIn('about 20 words', short)
        self.assertIn('about 40 characters', short)
        self.assertNotIn('80 to 150', short)

    def test_tied_embedding_generation_stops_at_eos(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(6, 4)
                self.config = type('Config', (), {'eos_token_id': 2})()

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
                hidden[:, -1, :] = self.embed_tokens.weight[2]
                return type('Output', (), {'last_hidden_state': hidden})()

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False):
                if tokenize:
                    return {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.tensor([[1]])}
                return 'chat'

            def decode(self, tokens, skip_special_tokens=True):
                return 'done'

        model = Model()
        text = _promptenhance._generate_tied(
            model, Tokenizer(), model.embed_tokens, 'fox', 'system', 1,
            _promptenhance._TokenCounter('Qwen3Model', _promptenhance.MAX_NEW_TOKENS),
            _promptenhance.sampling_for(_enums.ModelType.Z_IMAGE))
        self.assertEqual(text, 'done')

    def test_offloaded_embedding_stays_off_meta(self):
        class Hook:
            def __init__(self, weight):
                self.weights_map = {'weight': weight}

        embed = torch.nn.Embedding(4, 3)
        real = embed.weight.detach().clone()
        embed._hf_hook = Hook(real)
        embed.weight = torch.nn.Parameter(torch.empty(4, 3, device='meta'), requires_grad=False)
        weight = _promptenhance._resident_weight(embed)
        self.assertEqual(weight.device.type, 'cpu')
        placed = _promptenhance._on_real_device(torch.tensor([[1]]), weight)
        self.assertEqual(placed.device.type, 'cpu')
        self.assertEqual(placed.tolist(), [[1]])

    def test_token_counter_updates_one_line(self):
        import io
        import dgenerate.messages as _messages
        buf = io.StringIO()
        previous = _messages.get_message_file()
        _messages.set_message_file(buf)
        try:
            counter = _promptenhance._TokenCounter('Qwen3VLModel', 512)
            counter.add()
            counter.add()
            counter.finish()
            counter.finish()
        finally:
            _messages.set_message_file(previous)
        self.assertEqual(
            buf.getvalue(),
            '\rPipeline prompt enhance: Qwen3VLModel, 1 out of max 512 tokens'
            '\rPipeline prompt enhance: Qwen3VLModel, 2 out of max 512 tokens\n')
        logits = torch.tensor([float('nan'), float('inf'), 1.0, -3.0])
        token = _promptenhance._sample(
            logits, torch.Generator().manual_seed(0),
            _promptenhance.sampling_for(_enums.ModelType.Z_IMAGE))
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, logits.numel())
        self.assertEqual(_promptenhance.clean_response('Prompt: A fox.'), 'A fox.')
        chatglm = 'ยังไงก็ตาม \n image prompt: A nice horse eating grass.'
        self.assertEqual(
            _promptenhance.clean_response(chatglm), 'A nice horse eating grass.')
        echo = (
            "The sky is dark, the moon is bright. It's raining. "
            "Expand the user's request into one paragraph in the user's language "
            'that describes the finished image.')
        self.assertEqual(
            _promptenhance.clean_response(echo),
            "The sky is dark, the moon is bright. It's raining.")
        prefix_echo = '将您的整个回答写入下面的用户请求中。 不将你的回答翻译成任何其他语言。'
        self.assertIsNone(_promptenhance.clean_response(prefix_echo))
        spam = (
            'A photograph of a horse.\n'
            '- The horse is grazing on the grass.\n'
            '- The horse is grazing on the grass.\n'
            '- The horse is grazing on the grass.')
        self.assertEqual(_promptenhance.clean_response(spam), 'A photograph of a horse.')
        self.assertEqual(_promptenhance.clean_response('"A fox."'), 'A fox.')
        self.assertEqual(
            _promptenhance.clean_response('- A red cat on a blue chair.'),
            'A red cat on a blue chair.')
        sign = 'A sign reading "OPEN" above the door.'
        self.assertEqual(_promptenhance.clean_response(sign), sign)
        self.assertIsNone(_promptenhance.clean_response('  '))
        self.assertEqual(
            _promptenhance.clean_response('<think>planning</think>A chestnut horse.'),
            'A chestnut horse.')
        self.assertIsNone(_promptenhance.clean_response('<think>never finished'))

    def test_finalize_multilingual_rewrite(self):
        english = (
            'A white cat asleep on a blue chair beside a window, warm evening light. '
            'A paper sign on the wall reads 「OPEN」.')
        chinese = '三只猫和一只狗坐在木桌旁，桌上的纸牌写着「营业中」，傍晚的窗光'
        spanish = (
            'un gato rojo dormido en una silla azul junto a una ventana. en la pared, '
            'un letrero que dice «ABIERTO»')
        z = _enums.ModelType.Z_IMAGE
        english_rewrite = (
            'A white cat is asleep on a blue chair beside a window, warm evening light. '
            'A paper sign on the wall reads 「OPEN」.')
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                english, english_rewrite, z),
            english_rewrite)
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                english, '一只白猫在窗边睡觉。', z),
            english)
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                english,
                english + '生产总值',
                z),
            english)
        spanish_rewrite = 'A red cat is sleeping on a blue chair next to a window.'
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                spanish, spanish_rewrite, z),
            spanish_rewrite)
        with unittest.mock.patch.object(
                _promptenhance, '_translate_prompt_to_english',
                return_value='Three cats and one dog sit at a wooden table.'):
            kept = _promptenhance.finalize_multilingual_rewrite(
                chinese, '三只猫和一只狗坐在木桌旁。', z)
        self.assertIn('Three cats', kept)
        korean = (
            '시장 골목에서 할머니 한 명, 학생 두 명, 아이 한 명이 떡볶이를 나눠 '
            '먹고 있다. 저녁, 형광등')
        krea = _enums.ModelType.KREA2
        with unittest.mock.patch.object(
                _promptenhance, '_translate_prompt_to_english',
                return_value='An alley market scene with tteokbokki at dusk.'):
            kept = _promptenhance.finalize_multilingual_rewrite(korean, korean, krea)
        self.assertIn('tteokbokki', kept)
        flux = _enums.ModelType.FLUX2
        chinese_rewrite = '三只猫和一只狗坐在木桌旁，傍晚的窗光。'
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                chinese, chinese_rewrite, flux),
            chinese_rewrite)
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                chinese, 'Three cats at a wooden table.', flux),
            chinese)
        klein = _enums.ModelType.FLUX2_KLEIN_KV
        self.assertEqual(
            _promptenhance.finalize_multilingual_rewrite(
                chinese, chinese_rewrite, klein),
            chinese_rewrite)

    def test_translate_prompt_to_english_preserves_sign_literals(self):
        chinese = '木桌上的纸牌写着「营业中」'
        with unittest.mock.patch('dgenerate.translators.ArgosTranslator') as mock_cls:
            mock_cls.return_value.translate.side_effect = (
                lambda chunk: ['The cards on the table say '])
            out = _promptenhance._translate_prompt_to_english(chinese, local_files_only=True)
        self.assertIn('「营业中」', out)
        self.assertNotIn('SIGN', out)

    def test_clean_response_drops_glued_copyright_tail(self):
        source = (
            'A white cat asleep on a blue chair beside a window. A paper sign reads 「OPEN」.opyright')
        self.assertEqual(
            _promptenhance.clean_response(source),
            'A white cat asleep on a blue chair beside a window. A paper sign reads 「OPEN」.')

    def test_ltx_enhance_user_content(self):
        from dgenerate.pipelinewrapper import promptenhance as _promptenhance
        self.assertEqual(
            _promptenhance._ltx_enhance_user_content('fox', False),
            'user prompt: fox')
        self.assertEqual(
            _promptenhance._ltx_enhance_user_content('fox', True)[1]['text'],
            'User Raw Input Prompt: fox.')

    def test_system_prompt_file(self):
        import os
        import tempfile
        from dgenerate.console.formentries.stringfileentry import format_file_prompt_reference
        handle = tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', suffix='.txt', delete=False)
        handle.write('  Be precise.\n')
        handle.close()
        try:
            self.assertEqual(
                _promptenhance.resolve_system_prompt(f'file:{handle.name}'),
                'Be precise.')
            self.assertEqual(
                _promptenhance.resolve_system_prompt(
                    format_file_prompt_reference(handle.name)),
                'Be precise.')
            self.assertEqual(
                format_file_prompt_reference('C:/dir/New Text Document.txt'),
                'file:C:/dir/New Text Document.txt')
            self.assertEqual(
                _promptenhance.resolve_system_prompt(f'file: {handle.name}'),
                'Be precise.')
        finally:
            os.remove(handle.name)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _promptenhance.resolve_system_prompt('file: missing-system-prompt.txt')

    def test_check_keeps_a_system_prompt_file_reference(self):
        import os
        import tempfile
        handle = tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', suffix='.txt', delete=False)
        handle.write('Be precise.\n')
        handle.close()
        reference = f'file:{handle.name}'
        try:
            config = _config(
                model_path='org/z-image',
                model_type=_enums.ModelType.Z_IMAGE,
                pipeline_prompt_enhance=reference)
            config.check()
            self.assertEqual(config.pipeline_prompt_enhance, reference)
        finally:
            os.remove(handle.name)
        missing = _config(
            model_path='org/z-image',
            model_type=_enums.ModelType.Z_IMAGE,
            pipeline_prompt_enhance='file:missing-system-prompt.txt')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            missing.check()
        self.assertIn('does not exist', str(raised.exception))
        self.assertEqual(
            missing.pipeline_prompt_enhance, 'file:missing-system-prompt.txt')

    def test_generate_path_rewrites_prompt(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                ids = kwargs['input_ids']
                return torch.cat([ids, torch.tensor([[4, 5]])], dim=1)

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False):
                if tokenize:
                    return {'input_ids': torch.tensor([[1, 2, 3]])}
                return 'chat'

            def __call__(self, text, return_tensors='pt'):
                return {'input_ids': torch.tensor([[1, 2, 3]])}

            def decode(self, tokens, skip_special_tokens=True):
                return 'Prompt: a red fox in snow'

        pipe = unittest.mock.Mock(
            text_encoder=Encoder(), tokenizer=Tokenizer(),
            _dgenerate_enhance_processor=None)
        rewritten = _promptenhance.rewrite_prompt(
            pipe, 'fox', 'system', [], 'cpu', 1)
        self.assertEqual(rewritten, 'a red fox in snow')

    def test_inference_tensors_can_leave_an_offload_hook(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        with torch.inference_mode():
            parked = layer.weight.detach().clone()
        self.assertTrue(parked.is_inference())
        class Prefixed:
            def __init__(self, dataset, prefix):
                self.dataset = dataset
                self.prefix = prefix

            def __getitem__(self, key):
                return self.dataset[f'{self.prefix}{key}']

            def __iter__(self):
                return iter(key for key in self.dataset if key.startswith(self.prefix))

        layer._hf_hook = type('Hook', (), {
            'weights_map': Prefixed({'visual.proj.weight': parked}, 'visual.proj.'),
        })()
        layer.weight = torch.nn.Parameter(
            torch.empty(2, 2, device='meta'), requires_grad=True)
        with torch.inference_mode():
            _promptenhance._release_inference_tensors(layer)
        restored = layer._hf_hook.weights_map['weight']
        self.assertFalse(restored.is_inference())
        self.assertFalse(layer.weight.requires_grad)
        replacement = torch.nn.Parameter(restored, requires_grad=layer.weight.requires_grad)
        self.assertEqual(tuple(replacement.shape), (2, 2))

    def test_layer_visit_inside_inference_mode_stores_real_weights(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        hook = _promptenhance._ExecuteOnDeviceHook(torch.device('cpu'), torch.device('cpu'))
        with torch.inference_mode():
            layer.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
            self.assertTrue(layer.weight.is_inference())
            hook.pre_forward(layer, torch.zeros(1, 2))
            hook.post_forward(layer, None)
        self.assertFalse(layer.weight.is_inference())

    def test_processor_tokenizer_is_used_when_the_pipeline_has_none(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                ids = kwargs['input_ids']
                return torch.cat([ids, torch.tensor([[4]])], dim=1)

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False, enable_thinking=False):
                if tokenize:
                    return {'input_ids': torch.tensor([[1, 2]])}
                return 'chat'

            def decode(self, tokens, skip_special_tokens=True):
                return (
                    'A red fox with a bushy tail rests in soft snow under pale winter light.')

        pipe = type('Pipe', (), {
            'text_encoder': Encoder(),
            'processor': type('Processor', (), {'tokenizer': Tokenizer()})(),
        })()
        rewritten = _promptenhance.rewrite_prompt(
            pipe, 'fox', 'system', [], 'cpu', 1,
            model_type=_enums.ModelType.QWEN_IMAGE_21)
        self.assertEqual(
            rewritten,
            'A red fox with a bushy tail rests in soft snow under pale winter light.')

    def test_empty_generation_falls_back_to_the_source_prompt(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                return kwargs['input_ids']

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False):
                if tokenize:
                    return {'input_ids': torch.tensor([[1]])}
                return 'chat'

            def decode(self, tokens, skip_special_tokens=True):
                return '   '

        pipe = unittest.mock.Mock(text_encoder=Encoder(), tokenizer=Tokenizer())
        self.assertEqual(
            _promptenhance.rewrite_prompt(
                pipe, 'fox', 'system', [], 'cpu', 1,
                model_type=_enums.ModelType.Z_IMAGE),
            'fox')

    def test_diffusion_arguments_skip_ltx(self):
        class Wrapper:
            model_type = _enums.ModelType.LTX2
            device = 'cpu'
            _pipeline = None
            _local_files_only = False

        args = _pipeline_args()
        _promptenhance.enhance_diffusion_arguments(Wrapper(), args)
        self.assertEqual(args.prompt.positive, 'fox')

    def test_qwen25_token_floor_is_not_rejected_before_the_encoder_loads(self):
        def names(attr):
            return attr

        allowed = _promptenhance.PromptEnhanceOverrides(max_new_tokens=40)
        _promptenhance.check_generation_overrides(
            allowed, names, _enums.ModelType.QWEN_IMAGE)
        _promptenhance.check_generation_overrides(
            allowed, names, _enums.ModelType.QWEN_IMAGE,
            encoder_name='Qwen2_5_VLForConditionalGeneration')

        config = _config(
            model_path='org/qwen-image',
            model_type=_enums.ModelType.QWEN_IMAGE,
            pipeline_prompt_enhance_max_new_tokens=40)
        config.check()

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _promptenhance.check_generation_overrides(
                allowed, names, _enums.ModelType.QWEN_IMAGE,
                encoder_name='Qwen3Model')

        too_small = _promptenhance.PromptEnhanceOverrides(max_new_tokens=20)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _promptenhance.check_generation_overrides(
                too_small, names, _enums.ModelType.QWEN_IMAGE)

        crossed = _promptenhance.PromptEnhanceOverrides(
            max_new_tokens=40, min_new_tokens=50)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _promptenhance.check_generation_overrides(
                crossed, names, _enums.ModelType.QWEN_IMAGE,
                encoder_name='Qwen2_5_VLForConditionalGeneration')

        class Qwen3Model:
            pass

        class Pipe:
            text_encoder = Qwen3Model()
            tokenizer = object()

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as raised:
            _promptenhance.rewrite_prompt(
                Pipe(), 'a fox', None, None, 'cpu', 1,
                model_type=_enums.ModelType.QWEN_IMAGE,
                overrides=allowed)
        self.assertIn('pipeline-prompt-enhance-min-new-tokens', str(raised.exception))

    def test_generation_limits_imply_enhancer_flag(self):
        config = _config(
            model_path='org/z-image',
            model_type=_enums.ModelType.Z_IMAGE,
            pipeline_prompt_enhance_max_new_tokens=128)
        config.check()
        self.assertIsNone(config.pipeline_prompt_enhance)

    def test_ltx_token_counter_keeps_gemma_defaults(self):
        class Enhancer:
            pass

        class Pipe:
            prompt_enhancer = Enhancer()

        kwargs = {}
        counter = _promptenhance.attach_ltx_token_counter(Pipe(), kwargs)
        self.assertEqual(counter.maximum, 600)
        self.assertEqual(counter.encoder_name, 'Enhancer')
        generation = kwargs['generation_kwargs']
        self.assertFalse(generation['do_sample'])
        self.assertEqual(generation['no_repeat_ngram_size'], 5)
        generation['streamer'].put(object())
        token = type('Tok', (), {'ndim': 1, 'shape': (1,)})()
        generation['streamer'].put(token)
        self.assertEqual(counter.count, 1)

        limited = {'max_new_tokens': 100, 'generation_kwargs': {'do_sample': True, 'temperature': 0.5}}
        limited_counter = _promptenhance.attach_ltx_token_counter(Pipe(), limited)
        self.assertEqual(limited_counter.maximum, 100)
        self.assertEqual(limited['generation_kwargs']['temperature'], 0.5)
        self.assertIn('streamer', limited['generation_kwargs'])

    def test_ltx_enhance_prompt_kwargs_merge_gemma_defaults(self):
        class Pipe:
            prompt_enhancer = object()

        overrides = _promptenhance.PromptEnhanceOverrides(
            temperature=0.5, max_new_tokens=100, min_new_tokens=8)
        kwargs = _promptenhance.ltx_enhance_prompt_kwargs(Pipe(), overrides)
        self.assertEqual(kwargs['max_new_tokens'], 100)
        self.assertEqual(kwargs['generation_kwargs']['temperature'], 0.5)
        self.assertTrue(kwargs['generation_kwargs']['do_sample'])
        self.assertEqual(kwargs['generation_kwargs']['min_new_tokens'], 8)

    def test_merged_sampling_overrides_temperature_only(self):
        base = _promptenhance.sampling_for(_enums.ModelType.Z_IMAGE)
        merged = _promptenhance.merged_sampling(
            _enums.ModelType.Z_IMAGE, 'Qwen3Model',
            _promptenhance.PromptEnhanceOverrides(temperature=0.3))
        self.assertEqual(merged.temperature, 0.3)
        self.assertEqual(merged.top_p, base.top_p)

    def test_gpu_headroom_defaults_to_two_gib(self):
        self.assertEqual(
            _promptenhance._headroom_bytes(None),
            2 * 1024 ** 3)
        self.assertEqual(
            _promptenhance._headroom_bytes(
                _promptenhance.PromptEnhanceOverrides(gpu_headroom=1.5)),
            int(1.5 * 1024 ** 3))

    def test_gpu_budget_keeps_a_layer_prefix(self):
        class ToyDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(6, 4)
                self.visual = torch.nn.Embedding(32, 4)
                self.layers = torch.nn.ModuleList(ToyDecoderLayer() for _ in range(3))
                self.norm = torch.nn.LayerNorm(4)

        model = Toy()
        embed = _promptenhance._module_nbytes(model.embed_tokens)
        layer = _promptenhance._module_nbytes(model.layers[0])
        names = _promptenhance._modules_for_gpu(model, embed + 2 * layer)
        self.assertEqual(names, ['embed_tokens', 'layers.0', 'layers.1'])

    def test_layer_hook_leaves_the_kv_cache(self):
        hook = _promptenhance._LayerDeviceHook(torch.device('cpu'))
        hidden = torch.zeros(1, 2)
        cache = tuple((torch.zeros(1), torch.zeros(1)) for _ in range(6))
        pair = (torch.ones(2), torch.ones(2))
        args, kwargs = hook.pre_forward(
            None, hidden, cache, past_key_values=cache, kv_cache=pair)
        self.assertIs(args[1], cache)
        self.assertIs(kwargs['past_key_values'], cache)
        self.assertEqual(tuple(item.shape for item in kwargs['kv_cache']), ((2,), (2,)))

    @unittest.skipUnless(torch.cuda.is_available(), 'language RoPE device alignment needs CUDA')
    def test_language_rotary_accepts_cpu_position_ids(self):
        class TinyRotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('inv_freq', torch.ones(4))

            def forward(self, x, position_ids):
                inv = self.inv_freq[None, None, :, None].float().expand(
                    3, position_ids.shape[1], -1, 1)
                pos = position_ids[:, :, None, :].float()
                return (inv @ pos).transpose(2, 3).to(dtype=x.dtype)

        class ToyDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(4, 4, bias=False)
                self.embed_tokens = torch.nn.Embedding(8, 4)
                self.layers = torch.nn.ModuleList(ToyDecoderLayer() for _ in range(2))
                self.rotary_emb = TinyRotaryEmbedding()
                self.visual.rotary_emb = TinyRotaryEmbedding()

        model = Toy()
        model.embed_tokens.to('cuda')
        embeddings, _, _, _ = _promptenhance._placement_units(model)
        _promptenhance._move_language_rotary(model, embeddings)
        _promptenhance._hook_rotary_inputs(model)
        try:
            self.assertEqual(model.rotary_emb.inv_freq.device.type, 'cuda')
            self.assertEqual(model.visual.rotary_emb.inv_freq.device.type, 'cpu')
            hidden = torch.zeros(1, 2, 4, device='cuda')
            position_ids = torch.ones(3, 1, 2)
            freqs = model.rotary_emb(hidden, position_ids)
            self.assertEqual(freqs.device.type, 'cuda')
            vision = model.visual.rotary_emb(torch.zeros(1, 2, 4), torch.ones(3, 1, 2))
            self.assertEqual(vision.device.type, 'cpu')
        finally:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(model, recurse=True)

    def test_supported_model_types_have_pipeline_prompt_enhance_examples(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        required: dict[str, tuple[str, ...]] = {
            'flux2': (
                'examples/flux2/pipeline_prompt_enhance/text-to-image-config.dgen',
                'examples/flux2/pipeline_prompt_enhance/image-conditioned-config.dgen',
            ),
            'flux2-klein-kv': (
                'examples/flux2/pipeline_prompt_enhance/klein-kv-config.dgen',
            ),
            'z-image': ('examples/z-image/pipeline_prompt_enhance/config.dgen',),
            'qwen-image': ('examples/qwen-image/pipeline_prompt_enhance/text-to-image-config.dgen',),
            'qwen-image-edit': (
                'examples/qwen-image/edit/pipeline_prompt_enhance/config.dgen',
            ),
            'qwen-image-layered': (
                'examples/qwen-image/pipeline_prompt_enhance/layered-config.dgen',
            ),
            'qwen-image-21': (
                'examples/qwen-image-21/pipeline_prompt_enhance/text-to-image-config.dgen',
                'examples/qwen-image-21/pipeline_prompt_enhance/image-conditioned-config.dgen',
            ),
            'hyimage-21': ('examples/hyimage-21/pipeline_prompt_enhance/base-config.dgen',),
            'hyimage-21-refiner': (
                'examples/hyimage-21/pipeline_prompt_enhance/refiner-config.dgen',
            ),
            'kandinsky5-image': (
                'examples/kandinsky5/pipeline_prompt_enhance/text-to-image-config.dgen',
                'examples/kandinsky5/pipeline_prompt_enhance/image-edit-config.dgen',
            ),
            'kandinsky5-video': (
                'examples/kandinsky5/pipeline_prompt_enhance/text-to-video-config.dgen',
                'examples/kandinsky5/pipeline_prompt_enhance/image-to-video-config.dgen',
            ),
            'krea2': ('examples/krea2/pipeline_prompt_enhance/config.dgen',),
            'ltx2': (
                'examples/ltx2/basic/prompt-enhancer-config.dgen',
                'examples/ltx2/pipeline_prompt_enhance/text-to-video-config.dgen',
                'examples/ltx2/pipeline_prompt_enhance/i2v-config.dgen',
            ),
        }
        # No shipped checkpoint yet; enhance still works when a local pipeline is loaded.
        without_public_example = frozenset({'z-image-omni'})
        supported = set(_enums.PIPELINE_PROMPT_ENHANCE_MODEL_TYPES)
        self.assertEqual(set(required) | without_public_example, supported)
        for model_type, paths in required.items():
            for rel in paths:
                path = root / rel
                self.assertTrue(path.is_file(), rel)
                body = path.read_text(encoding='utf-8')
                self.assertRegex(
                    body,
                    rf'--model-type\s+{re.escape(model_type)}\b',
                    msg=rel)

    def test_console_recipes_expose_pipeline_prompt_enhance_controls(self):
        from pathlib import Path

        recipes = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes'
        model_type_re = re.compile(r'^--model-type\s+(\S+)', re.M)
        switch = '"id":"pipeline-prompt-enhance"'
        required_args = (
            '--pipeline-prompt-enhance-seeds',
            '--pipeline-prompt-enhance-temperature',
            '--pipeline-prompt-enhance-top-p',
            '--pipeline-prompt-enhance-top-k',
            '--pipeline-prompt-enhance-presence-penalty',
            '--pipeline-prompt-enhance-max-new-tokens',
            '--pipeline-prompt-enhance-min-new-tokens',
            '--pipeline-prompt-enhance-gpu-headroom',
        )
        supported = set(_enums.PIPELINE_PROMPT_ENHANCE_MODEL_TYPES)

        for path in sorted(recipes.glob('*.recipe')):
            text = path.read_text(encoding='utf-8')
            match = model_type_re.search(text)
            if not match:
                continue
            model_type = match.group(1)
            has_switch = switch in text
            if model_type in supported:
                self.assertTrue(has_switch, path.name)
                self.assertIn('@stringfile[{', text, path.name)
                for arg in required_args:
                    self.assertIn(arg, text, f'{path.name} missing {arg}')
            else:
                self.assertFalse(has_switch, path.name)


class TestHeadGeneration(unittest.TestCase):
    def test_processor_without_chat_template_uses_its_tokenizer(self):
        class Inner:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False, enable_thinking=False):
                if tokenize:
                    return {'input_ids': torch.tensor([[1, 2]])}
                return 'formatted'

            def __call__(self, text, return_tensors='pt'):
                return {'input_ids': torch.tensor([[1]])}

        class Processor:
            def __init__(self):
                self.tokenizer = Inner()
                self.calls = []

            def apply_chat_template(self, *args, **kwargs):
                raise ValueError(
                    'Cannot use apply_chat_template because this processor does not have a chat template.')

            def __call__(self, text=None, return_tensors='pt', images=None):
                self.calls.append((text, images))
                return {'input_ids': torch.tensor([[1, 2, 3]]), 'pixel_values': torch.zeros(1)}

        processor = Processor()
        text_only = _promptenhance._chat_inputs(processor, 'lantern', 'system', None)
        self.assertEqual(text_only['input_ids'].tolist(), [[1, 2]])
        with_image = _promptenhance._chat_inputs(processor, 'edit', 'system', ['frame'])
        self.assertEqual(processor.calls, [('formatted', ['frame'])])
        self.assertIn('pixel_values', with_image)

    def test_presence_penalty_skips_the_prompt(self):
        processor = _promptenhance._PresencePenaltyLogitsProcessor(1.5, 2)
        scores = torch.zeros(1, 6)
        # Prompt tokens are 4 and 4. Produced tokens are 3, 3, and 1.
        changed = processor(torch.tensor([[4, 4, 3, 3, 1]]), scores)
        self.assertEqual(changed[0].tolist(), [0, -1.5, 0, -1.5, 0, 0])

    def test_head_sampling_and_seed_stay_local(self):
        class Tokenizer:
            def __init__(self):
                self.systems = []

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    return_tensors=None, return_dict=False, enable_thinking=False):
                self.systems.append(messages[0]['content'])
                if tokenize:
                    return {'input_ids': torch.tensor([[1, 2, 3]])}
                return 'chat'

            def __call__(self, text, return_tensors='pt', images=None):
                return {'input_ids': torch.tensor([[1, 2]])}

            def decode(self, tokens, skip_special_tokens=True):
                return 'a red fox'

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                torch.manual_seed(99)
                return torch.cat([kwargs['input_ids'], torch.tensor([[4]])], dim=1)

        encoder = Encoder()
        tokenizer = Tokenizer()
        pipe = type('Pipe', (), {})()
        sampling = _promptenhance.Sampling(0.7, 0.8, 0, 1.5)
        counter = _promptenhance._TokenCounter('Qwen', 32)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        text = _promptenhance._generate_with_head(
            pipe, encoder, tokenizer, 'fox', _promptenhance.EDIT_SYSTEM_PROMPT,
            [object()], 7, False, counter, sampling, max_new_tokens=16, min_new_tokens=0)
        after = torch.random.get_rng_state()
        self.assertEqual(text, 'a red fox')
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(tokenizer.systems[-1], _promptenhance.T2I_SYSTEM_PROMPT)
        kwargs = encoder.calls[0]
        self.assertTrue(kwargs['do_sample'])
        self.assertEqual(kwargs['temperature'], 0.7)
        self.assertEqual(kwargs['top_p'], 0.8)
        self.assertEqual(kwargs['top_k'], 0)
        self.assertEqual(len(kwargs['logits_processor']), 1)

        greedy = _promptenhance.Sampling(0, 0.8, 20, 0)
        _promptenhance._generate_with_head(
            pipe, encoder, tokenizer, 'fox', 'custom', [], 3, False, counter, greedy,
            max_new_tokens=8, min_new_tokens=0)
        quiet = encoder.calls[1]
        self.assertFalse(quiet['do_sample'])
        self.assertNotIn('temperature', quiet)
        self.assertNotIn('top_k', quiet)
        self.assertNotIn('logits_processor', quiet)

    def test_disk_offload_weights_are_loaded_for_the_rewrite(self):
        import tempfile

        from dgenerate.pipelinewrapper.videopipelines import diskoffload

        layer = torch.nn.Sequential(torch.nn.Linear(4, 4))
        weight = layer[0].weight.detach().clone()
        folder = tempfile.mkdtemp()
        index = diskoffload._write_weights(layer.state_dict(), folder, desc='test weights')
        diskoffload._install_disk_hooks(layer, index, torch.device('cpu'))
        self.assertEqual(layer[0].weight.device.type, 'meta')
        self.assertTrue(layer._dgenerate_weights_on_disk)
        try:
            with _promptenhance._fast_encoder(layer, 'cpu', offload=True):
                self.assertEqual(layer[0].weight.device.type, 'cpu')
                self.assertTrue(torch.equal(layer[0].weight, weight))
                self.assertFalse(layer._dgenerate_weights_on_disk)
            self.assertTrue(layer._dgenerate_weights_on_disk)
            self.assertEqual(layer[0].weight.device.type, 'meta')
            output = layer(torch.ones(1, 4))
            self.assertEqual(tuple(output.shape), (1, 4))
        finally:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)

    def test_loaded_qwen_processor_is_used_for_the_image(self):
        processor = type('QwenProcessor', (), {
            'image_processor': object(),
            'tokenizer': object(),
        })()
        pipe = type('Pipe', (), {'processor': None, 'tokenizer': processor})()
        encoder = type('Encoder', (), {'name_or_path': 'repo/text_encoder'})()
        found = _promptenhance._vision_processor(pipe, encoder, processor, True)
        self.assertIs(found, processor)


def _pipeline_args():
    import dgenerate.pipelinewrapper as _pipelinewrapper
    args = _pipelinewrapper.DiffusionArguments()
    args.prompt = _prompt.Prompt()
    args.prompt.positive = 'fox'
    args.pipeline_prompt_enhance = True
    args.seed = 1
    return args


if __name__ == '__main__':
    unittest.main()
