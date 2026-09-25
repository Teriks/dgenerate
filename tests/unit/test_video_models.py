import inspect
import math
import os
import tempfile
import unittest
import unittest.mock

import av
import numpy
import PIL.Image

import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.videopipelines as _videopipelines
import dgenerate.prompt as _prompt
import dgenerate.renderloopconfig as _renderloopconfig


def _config(**values):
    config = _renderloopconfig.RenderLoopConfig()
    for name, value in values.items():
        setattr(config, name, value)
    return config


class TestVideoModels(unittest.TestCase):
    def test_end_image_parse(self):
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;end=examples/media/beach.jpg')
        self.assertEqual(parsed.end_image, 'examples/media/beach.jpg')
        self.assertFalse(parsed.is_single_spec)

        with self.assertRaises(_mediainput.ImageSeedFileNotFoundError):
            _mediainput.parse_image_seed_uri(
                'examples/media/earth.jpg;end=examples/media/missing-end.jpg')

    def test_ltx_check_defaults(self):
        config = _config(model_path='org/ltx', model_type=_pipelinewrapper.ModelType.LTX)
        config.check()
        self.assertEqual(config.video_fps, [24.0])
        self.assertEqual(
            config.inference_steps,
            [_pipelinewrapper.constants.DEFAULT_INFERENCE_STEPS])

        listed = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            sigmas=[[1.0, 0.5, 0.25]])
        listed.check()
        self.assertEqual(listed.inference_steps, [3])

    def test_distilled_steps_warning_shows_passed_value(self):
        default = _pipelinewrapper.constants.DEFAULT_INFERENCE_STEPS
        self.assertIsNone(_videopipelines._ltx_ignored_steps_warning(8, 8))
        self.assertEqual(
            _videopipelines._ltx_ignored_steps_warning(None, 8),
            'The LTX distilled checkpoint uses an 8-value sigma schedule '
            f'and ignores --inference-steps {default} (the default).')
        self.assertEqual(
            _videopipelines._ltx_ignored_steps_warning(default, 8),
            'The LTX distilled checkpoint uses an 8-value sigma schedule '
            f'and ignores --inference-steps {default} (the default).')
        self.assertEqual(
            _videopipelines._ltx_ignored_steps_warning(20, 8),
            'The LTX distilled checkpoint uses an 8-value sigma schedule '
            'and ignores --inference-steps 20.')

    def test_unknown_video_type_is_rejected(self):
        with self.assertRaises(ValueError):
            _pipelinewrapper.get_model_type_enum('minimax-h3')
        with self.assertRaises(ValueError):
            _pipelinewrapper.get_model_type_enum('wan-animate')
        self.assertFalse(_pipelinewrapper.model_type_is_video(_pipelinewrapper.ModelType.FLUX))

    def test_ltx_scheduler_uri(self):
        rejected = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            scheduler_uri='EulerDiscreteScheduler')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            rejected.check()
        self.assertIn('FlowMatchEulerDiscreteScheduler', str(raised.exception))

        for uri in (
            'FlowMatchEulerDiscreteScheduler',
            'FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=false;shift=1.0',
            'help',
            'helpargs',
        ):
            allowed = _config(
                model_path='org/ltx',
                model_type=_pipelinewrapper.ModelType.LTX,
                scheduler_uri=uri)
            allowed.check()

    def test_scheduler_uri_applied_before_distilled_check(self):
        class Scheduler:
            def __init__(self):
                self.config = {'use_dynamic_shifting': False}

        class Pipe:
            def __init__(self):
                self.scheduler = Scheduler()

        pipe = Pipe()
        captured = {}

        def apply_scheduler(pipeline, scheduler_uri):
            captured['uri'] = scheduler_uri
            pipeline.scheduler.config['use_dynamic_shifting'] = True

        def invoke(wrapper, pipeline, kwargs):
            captured['kwargs'] = kwargs

            class Output:
                frames = [PIL.Image.new('RGB', (4, 4))]
                audio = None

            return Output()

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('fox')
        args.inference_steps = 50
        args.guidance_scale = 6
        args.audio_guidance_scale = 7
        args.video_fps = 24
        args.video_length = 2
        args.width = 640
        args.height = 384
        args.scheduler_uri = (
            'FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true')

        class Held:
            pipeline = pipe
            family = 'ltx2'

        class Wrapper:
            device = 'cpu'
            model_type = _pipelinewrapper.ModelType.LTX
            model_cpu_offload = False
            model_sequential_offload = False
            model_path = 'org/ltx'
            _revision = None
            _variant = None
            _subfolder = None
            _dtype = None
            _local_files_only = False
            _auth_token = None
            quantizer_uri = None
            quantizer_map = None
            transformer_uri = None
            lora_uris = None
            lora_fuse_scale = None

        with unittest.mock.patch.object(
                _videopipelines, '_create_cached_video_pipeline',
                return_value=Held()), \
                unittest.mock.patch.object(
                    _videopipelines, 'pipeline_for_mode',
                    side_effect=lambda pipeline, mode, family: pipeline), \
                unittest.mock.patch.object(
                    _videopipelines._schedulers, 'load_scheduler',
                    side_effect=apply_scheduler), \
                unittest.mock.patch.object(
                    _videopipelines, '_invoke', side_effect=invoke):
            _videopipelines._call_ltx(Wrapper(), args)

        self.assertEqual(
            captured['uri'],
            'FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true')
        self.assertEqual(captured['kwargs']['num_inference_steps'], 50)
        self.assertNotIn('sigmas', captured['kwargs'])
        self.assertFalse(_videopipelines._ltx_is_distilled(pipe))

    def test_ltx_rejects_control(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg;control=examples/media/earth.jpg'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_end_is_video_only(self):
        config = _config(
            model_path='org/sd',
            image_seeds=['examples/media/earth.jpg;end=examples/media/beach.jpg'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_video_rejects_batch_size_and_seed_strength(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            batch_size=2)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg'],
            image_seed_strengths=[0.4])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_model_path_is_required(self):
        config = _config(model_type=_pipelinewrapper.ModelType.LTX)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_length_product(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            prompts=[_prompt.Prompt(), _prompt.Prompt()],
            video_lengths=[5.0, 2.0])
        self.assertEqual(config.calculate_generation_steps(), 4)

        audio = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            prompts=[_prompt.Prompt(), _prompt.Prompt()],
            audio_guidance_scales=[1.0, 7.0],
            audio_guidance_rescales=[0.5, 0.7])
        self.assertEqual(audio.calculate_generation_steps(), 8)

    def test_ltx_fps_default(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            video_fps=[12.0])
        config.check()
        self.assertEqual(config.video_fps, [12.0])

    def test_canvas_alignment(self):
        aligned = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            output_size=(512, 512))
        aligned.check()

        misaligned = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            output_size=(520, 512))
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            misaligned.check()

    def test_ltx_sigma_expression(self):
        scaled = _videopipelines._eval_sigma_expression(
            'sigmas * 0.95',
            [1.0, 0.5, 0.25])
        self.assertEqual(len(scaled), 3)
        self.assertAlmostEqual(scaled[0], 0.95)
        self.assertAlmostEqual(scaled[2], 0.2375)

        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._eval_sigma_expression('1 + 1', [1.0, 0.5])

    def test_legacy_ltx_schedule_overflow(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_image_seq_len=1024,
            max_image_seq_len=4096,
            base_shift=0.95,
            max_shift=2.05,
            shift_terminal=0.1,
            shift=1.0)

        self.assertTrue(_videopipelines._legacy_ltx_schedule_finite(
            scheduler, 768, 512, 121, 50))
        self.assertFalse(_videopipelines._legacy_ltx_schedule_finite(
            scheduler, 1024, 1024, 753, 50))

    def test_frame_snap(self):
        self.assertEqual(_videopipelines.ltx_num_frames(5, 24), 121)
        self.assertEqual(_videopipelines.ltx_num_frames(2, 16), 33)

    def test_output_normalization(self):
        image = PIL.Image.new('RGB', (4, 4), (1, 2, 3))
        nested = _videopipelines.frames_from_output([[image]])
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0].getpixel((0, 0)), (1, 2, 3))

        channels_last = numpy.zeros((2, 8, 8, 3), dtype=numpy.uint8)
        channels_last[0, :, :, 0] = 10
        last_frames = _videopipelines.frames_from_output(channels_last)
        self.assertEqual(len(last_frames), 2)
        self.assertEqual(last_frames[0].getpixel((0, 0)), (10, 0, 0))

        channels_first = numpy.zeros((2, 3, 8, 8), dtype=numpy.uint8)
        channels_first[1, 1, :, :] = 20
        first_frames = _videopipelines.frames_from_output(channels_first)
        self.assertEqual(first_frames[1].getpixel((0, 0)), (0, 20, 0))

        audio = numpy.zeros((1, 2, 16), dtype=numpy.float32)
        audio[0, 1, 3] = 0.5
        planar = _videopipelines.audio_to_numpy(audio)
        self.assertEqual(planar.shape, (2, 16))
        self.assertAlmostEqual(float(planar[1, 3]), 0.5)

    def test_submodel_arguments(self):
        allowed = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            transformer_uri='Lightricks/LTX-2.5-Diffusers;subfolder=transformer_full',
            lora_uris=['org/lora;scale=0.8'])
        allowed.check()

        quantized = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            quantizer_uri='bnb;bits=4')
        quantized.check()

        rejected = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            vae_uri='org/vae')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            rejected.check()

        mapped = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            quantizer_uri='bnb;bits=4',
            quantizer_map=['text_encoder_2'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            mapped.check()

        encoder_map = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            quantizer_uri='bnb;bits=4',
            quantizer_map=['transformer', 'text_encoder', 'connectors'])
        encoder_map.check()

        rescale = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            guidance_rescales=[0.7])
        rescale.check()

        audio = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            audio_guidance_scales=[7.0],
            audio_guidance_rescales=[0.5])
        audio.check()
        steps = list(audio.iterate_diffusion_args())
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].audio_guidance_scale, 7.0)
        self.assertEqual(steps[0].audio_guidance_rescale, 0.5)

        still = _config(
            model_path='org/sd',
            audio_guidance_scales=[7.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still.check()
        self.assertIn('ltx', str(raised.exception).lower())

        still_rescale = _config(
            model_path='org/sd',
            audio_guidance_rescales=[0.5])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_rescale.check()
        self.assertIn('ltx', str(raised.exception).lower())

        still_length = _config(
            model_path='org/sd',
            video_lengths=[2.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_length.check()
        self.assertIn('ltx', str(raised.exception).lower())

        still_fps = _config(
            model_path='org/sd',
            video_fps=[24.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_fps.check()
        self.assertIn('ltx', str(raised.exception).lower())

        sequence = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            max_sequence_length=1024,
            vae_slicing=True)
        sequence.check()

        too_long = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            max_sequence_length=1025)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            too_long.check()

        tiled = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            vae_tiling=True)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            tiled.check()
        self.assertIn('always tiles', str(raised.exception))

    def test_ltx_rejects_inapplicable_arguments(self):
        cases = {
            'second_prompts': [_prompt.Prompt('style')],
            'third_prompts': [_prompt.Prompt('style')],
            'second_prompt_upscaler_uri': 'dynamicprompts',
            'third_prompt_upscaler_uri': 'dynamicprompts',
            'clip_skips': [1],
            'image_encoder_uri': 'org/encoder',
            'pag': True,
            'pag_scales': [2.0],
            'prompt_weighter_uri': 'compel',
            'safety_checker': True,
            'batch_grid_size': (2, 2),
            'denoising_start': 0.2,
            'denoising_end': 0.8,
            'latents_processors': ['normalize'],
            'original_config': 'model.yaml',
            'frame_start': 3,
            'frame_end': 10,
            'control_image_processors': ['canny'],
            'inpaint_crop': True,
            'vae_tiling': True,
        }
        for name, value in cases.items():
            with self.subTest(name):
                kwargs = {name: value}
                if name in {
                    'latents_processors',
                    'control_image_processors',
                    'inpaint_crop',
                }:
                    kwargs['image_seeds'] = ['examples/media/earth.jpg']
                config = _config(
                    model_path='org/ltx',
                    model_type=_pipelinewrapper.ModelType.LTX,
                    **kwargs)
                with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
                    config.check()

    def test_video_writer_muxes_audio(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'clip.mp4')
        sample_rate = 16000
        samples = numpy.arange(sample_rate // 2, dtype=numpy.float32) / sample_rate
        audio = numpy.sin(2 * math.pi * 440 * samples).reshape(1, -1)
        frames = [PIL.Image.new('RGB', (64, 64), (index * 40, 0, 0)) for index in range(4)]

        with _mediaoutput.VideoWriter(path, 8, audio=audio, audio_sample_rate=sample_rate) as writer:
            for frame in frames:
                writer.write(frame)

        container = av.open(path)
        try:
            self.assertEqual(len(container.streams.video), 1)
            self.assertEqual(len(container.streams.audio), 1)
        finally:
            container.close()

    def test_extra_weight_directories(self):
        self.assertEqual(
            _videopipelines.extra_weight_directories_from_index(None),
            ['audio_vae', 'vocoder'])
        index = {
            '_class_name': 'LTX2Pipeline',
            'transformer': ['diffusers', 'LTX2VideoTransformer3DModel'],
            'vae': ['diffusers', 'AutoencoderKLLTX2Video'],
            'audio_vae': ['diffusers', 'AutoencoderKLLTX2Audio'],
            'vocoder': ['transformers', 'SomeVocoder'],
            'text_encoder': ['transformers', 'Gemma3ForConditionalGeneration'],
            'connector': ['diffusers', 'LTX2TextConnector'],
            'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
        }
        self.assertEqual(
            _videopipelines.extra_weight_directories_from_index(index),
            ['audio_vae', 'vocoder', 'connector'])
        classic = {
            '_class_name': 'LTXPipeline',
            'transformer': ['diffusers', 'LTXVideoTransformer3DModel'],
            'vae': ['diffusers', 'AutoencoderKLLTXVideo'],
            'text_encoder': ['transformers', 'T5EncoderModel'],
            'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
        }
        self.assertEqual(
            _videopipelines.extra_weight_directories_from_index(classic),
            [])

    def test_ltx_family_from_index(self):
        self.assertEqual(
            _videopipelines.ltx_family_from_index({'_class_name': 'LTX2Pipeline'}),
            'ltx2')
        self.assertEqual(
            _videopipelines.ltx_family_from_index({'_class_name': 'LTXPipeline'}),
            'ltx')
        self.assertEqual(
            _videopipelines.ltx_family_from_index({'_class_name': 'LTXImageToVideoPipeline'}),
            'ltx')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.ltx_family_from_index({'_class_name': 'FluxPipeline'})

    def test_default_quant_names_include_connectors(self):
        self.assertTrue(
            _videopipelines._quantize_component('connectors', 'sdnq', None))
        self.assertTrue(
            _videopipelines._quantize_component('transformer', 'sdnq', None))
        self.assertTrue(
            _videopipelines._quantize_component('text_encoder', 'sdnq', None))
        self.assertFalse(
            _videopipelines._quantize_component('vae', 'sdnq', None))
        self.assertFalse(
            _videopipelines._quantize_component('prompt_enhancer', 'sdnq', None))
        self.assertFalse(
            _videopipelines._quantize_component('connectors', None, None))
        self.assertTrue(
            _videopipelines._quantize_component(
                'connectors', 'sdnq', ['connectors']))
        self.assertFalse(
            _videopipelines._quantize_component(
                'connectors', 'sdnq', ['transformer']))

    def test_resolve_ltx_index_class(self):
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from diffusers import LTX2VideoTransformer3DModel
        self.assertIs(
            _videopipelines._resolve_index_class('ltx2', 'LTX2TextConnectors'),
            LTX2TextConnectors)
        self.assertIs(
            _videopipelines._resolve_index_class(
                'diffusers', 'LTX2VideoTransformer3DModel'),
            LTX2VideoTransformer3DModel)

    def test_pretrained_kwargs_uses_torch_dtype(self):
        import torch
        kwargs = _videopipelines._pretrained_kwargs(
            None, None, None, True, None, _pipelinewrapper.DataType.BFLOAT16, True)
        self.assertEqual(kwargs.get('torch_dtype'), torch.bfloat16)
        self.assertNotIn('dtype', kwargs)

    def test_skip_prompt_enhancer(self):
        self.assertIsNone(
            _videopipelines._LTX_SKIP_OPTIONAL_MODULES['prompt_enhancer'])

    def test_quantized_classic_modules_resolves_connectors(self):
        loaded = {}

        def fake_load(component_class, model_path, subfolder, *args, **kwargs):
            loaded[subfolder] = component_class
            return object()

        index = {
            'transformer': ['diffusers', 'LTX2VideoTransformer3DModel'],
            'connectors': ['ltx2', 'LTX2TextConnectors'],
            'vae': ['diffusers', 'AutoencoderKLLTX2Video'],
            'prompt_enhancer': ['transformers', 'Gemma4ForConditionalGeneration'],
        }
        with unittest.mock.patch.object(
                _videopipelines._util, 'fetch_model_index_dict', return_value=index), \
                unittest.mock.patch.object(
                    _videopipelines, '_load_quantized_module', side_effect=fake_load):
            modules = _videopipelines._quantized_classic_modules(
                'org/ltx', None, None, None, _pipelinewrapper.DataType.BFLOAT16,
                'sdnq', None, None, True, 'cpu', False)
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        self.assertIn('connectors', modules)
        self.assertIn('transformer', modules)
        self.assertNotIn('vae', modules)
        self.assertNotIn('prompt_enhancer', modules)
        self.assertIs(loaded['connectors'], LTX2TextConnectors)

    def test_cache_kwargs_omits_mode(self):
        class Wrapper:
            model_path = 'org/ltx'
            model_type = _pipelinewrapper.ModelType.LTX
            _revision = None
            _variant = None
            _subfolder = None
            _dtype = None
            device = 'cpu'
            model_cpu_offload = False
            model_sequential_offload = False
            _local_files_only = False
            _auth_token = None
            quantizer_uri = None
            quantizer_map = None

        keys = _videopipelines._cache_kwargs(Wrapper())
        self.assertNotIn('mode', keys)
        self.assertNotIn('scheduler_uri', keys)
        self.assertNotIn(
            'scheduler_uri',
            inspect.signature(_videopipelines._create_cached_video_pipeline).parameters)
        self.assertNotIn(
            'scheduler_uri',
            inspect.signature(_videopipelines._pipelines._create_diffusion_pipeline).parameters)
        video_exceptions = inspect.getclosurevars(
            _videopipelines._create_cached_video_pipeline).nonlocals['exceptions']
        still_exceptions = inspect.getclosurevars(
            _videopipelines._pipelines._create_diffusion_pipeline).nonlocals['exceptions']
        self.assertIn('local_files_only', video_exceptions)
        self.assertIn('local_files_only', still_exceptions)

    def test_audio_guidance_writeback(self):
        source = _pipelinewrapper.DiffusionArguments()
        source.guidance_scale = 3.0
        source.inference_steps = 30
        source.audio_guidance_scale = 7.0
        source.audio_guidance_rescale = 0.5
        dest = _pipelinewrapper.DiffusionArguments()
        _videopipelines.apply_video_arg_rewrites(source, dest)
        self.assertEqual(dest.guidance_scale, 3.0)
        self.assertEqual(dest.inference_steps, 30)
        self.assertEqual(dest.audio_guidance_scale, 7.0)
        self.assertEqual(dest.audio_guidance_rescale, 0.5)

    def test_audio_sample_rate_warns_without_vocoder(self):
        audio = numpy.zeros((1, 8), dtype=numpy.float32)
        self.assertIsNone(
            _videopipelines.audio_sample_rate_from_pipeline(object(), audio))

        class Vocoder:
            class config:
                output_sampling_rate = 16000

        class Pipe:
            vocoder = Vocoder()

        self.assertEqual(
            _videopipelines.audio_sample_rate_from_pipeline(Pipe(), audio),
            16000)

    def test_pipeline_for_mode_shares_components(self):
        vae = object()

        class Base:
            def __init__(self):
                self.vae = vae
                self.components = {'vae': vae, 'unused': object()}

        class Other:
            def __init__(self, vae=None):
                self.vae = vae
                self.components = {'vae': vae}

            @classmethod
            def from_pipe(cls, pipe):
                raise AssertionError('from_pipe recasts dtype and breaks quantized modules')

        with unittest.mock.patch.object(
                _videopipelines, '_ltx_pipeline_class', return_value=Other):
            converted = _videopipelines.pipeline_for_mode(Base(), 'ltx-image')
        self.assertIs(converted.vae, vae)
        self.assertIs(converted.components['vae'], vae)

    def test_sigma_expression_restores_scheduler(self):
        class Scheduler:
            def __init__(self):
                self.sigmas = [1.0, 0.0]
                self.timesteps = [10]
                self.num_inference_steps = 1

            def set_timesteps(self, steps):
                self.sigmas = [float(steps), 0.0]
                self.timesteps = list(range(steps))
                self.num_inference_steps = steps

        class Pipe:
            scheduler = Scheduler()

        pipe = Pipe()
        values = _videopipelines._ltx_base_sigmas(pipe, False, 4)
        self.assertEqual(values, [4.0, 0.0])
        self.assertEqual(pipe.scheduler.sigmas, [1.0, 0.0])
        self.assertEqual(pipe.scheduler.timesteps, [10])
        self.assertEqual(pipe.scheduler.num_inference_steps, 1)


if __name__ == '__main__':
    unittest.main()
