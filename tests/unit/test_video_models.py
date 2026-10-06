import inspect
import math
import os
import tempfile
import unittest
import unittest.mock

import av
import numpy
import PIL.Image
import torch

import dgenerate.arguments as _arguments
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
            'examples/media/earth.jpg;last-frame=examples/media/beach.jpg')
        self.assertEqual(parsed.end_image, 'examples/media/beach.jpg')
        self.assertFalse(parsed.is_single_spec)

        with self.assertRaises(_mediainput.ImageSeedFileNotFoundError):
            _mediainput.parse_image_seed_uri(
                'examples/media/earth.jpg;last-frame=examples/media/missing-end.jpg')

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
            _pipelinewrapper.get_model_type_enum('hunyuan-video')
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
        args.ltx_audio_guidance_scale = 7
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

    def test_ltx_control_seed(self):
        gif = 'examples/media/rickroll-roll.gif'
        ltx = _pipelinewrapper.ModelType.LTX
        with_first = _mediainput.parse_image_seed_uri(f'examples/media/earth.jpg;control={gif}')
        plain = _mediainput.parse_image_seed_uri(gif)

        self.assertEqual(_videopipelines.classify_video_seed(ltx, with_first, True), 'ltx-control')
        self.assertEqual(_videopipelines.classify_video_seed(ltx, plain, True), 'ltx-control')
        self.assertEqual(_videopipelines.classify_video_seed(ltx, plain), 'ltx-image')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(ltx, with_first)

        self.assertEqual(
            _videopipelines.video_seed_slots(plain, True), (None, None, gif))
        self.assertEqual(
            _videopipelines.video_seed_slots(plain), (gif, None, None))
        self.assertEqual(
            _videopipelines.video_seed_slots(with_first, True),
            ('examples/media/earth.jpg', None, gif))

        ic = 'Lightricks/ic-lora;weight-name=ic.safetensors'
        for seed in (gif, f'examples/media/earth.jpg;control={gif}'):
            config = _config(
                model_path='org/ltx', model_type=ltx, ltx_ic_lora_uri=ic,
                image_seeds=[seed], control_image_processors=['canny'])
            config.check()

        for values in (
                {'image_seeds': [f'examples/media/earth.jpg;control={gif}']},
                {'ltx_ic_lora_uri': ic},
                {'ltx_ic_lora_uri': ic, 'image_seeds': [';last-frame=examples/media/earth.jpg']},
                {'ltx_ic_lora_uri': f'{ic};attention=2', 'image_seeds': [gif]},
                {'image_seeds': ['examples/media/earth.jpg'], 'control_image_processors': ['canny']},
                {'ltx_ic_lora_uri': ic,
                 'image_seeds': [f'examples/media/earth.jpg;control={gif}, examples/media/beach.jpg']}):
            config = _config(model_path='org/ltx', model_type=ltx, **values)
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError, msg=str(values)):
                config.check()

        image_model = _config(model_path='org/sd', ltx_ic_lora_uri=ic)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            image_model.check()

        for values in (
                {'ltx_stg_scales': [1.0]},
                {'ltx_latent_upscale': True},
                {'ltx_use_cross_timestep': False},
                {'video_fps': [24.0]},
        ):
            blocked = _config(
                model_path='org/sd',
                model_type=_pipelinewrapper.ModelType.SD,
                **values)
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError, msg=str(values)):
                blocked.check()

        unused = _config(
            model_path='org/sd',
            model_type=_pipelinewrapper.ModelType.SD,
            ltx_latent_upscale=False)
        unused.check()

    def test_ic_lora_uri(self):
        uri = _pipelinewrapper.uris.ICLoRAUri.parse(
            'org/ic;scale=0.5;attention=0.25;downscale=2;weight-name=w.safetensors;revision=dev')
        self.assertEqual((uri.scale, uri.attention, uri.downscale), (0.5, 0.25, 2))
        lora = _pipelinewrapper.uris.LoRAUri.parse(uri.lora_uri())
        self.assertEqual(
            (lora.model, lora.scale, lora.weight_name, lora.revision),
            ('org/ic', 0.5, 'w.safetensors', 'dev'))
        self.assertIsNone(_pipelinewrapper.uris.ICLoRAUri.parse('org/ic').downscale)
        for bad in ('org/ic;downscale=0', 'org/ic;attention=-1', 'org/ic;scale=x', 'org/ic;mode=1'):
            with self.assertRaises(_pipelinewrapper.uris.InvalidLoRAUriError, msg=bad):
                _pipelinewrapper.uris.ICLoRAUri.parse(bad)

    def test_ltx_seed_processor_chains(self):
        for processors in (['flip', '+', 'mirror'], ['+', 'mirror'], ['flip', '+']):
            config = _config(
                model_path='org/ltx',
                model_type=_pipelinewrapper.ModelType.LTX,
                image_seeds=['examples/media/earth.jpg;last-frame=examples/media/beach.jpg'],
                seed_image_processors=processors)
            config.check()

        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg;last-frame=examples/media/beach.jpg'],
            seed_image_processors=['flip', '+', 'mirror', '+', 'grayscale'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg;last-frame=examples/media/beach.jpg'],
            last_frame_image_processors=['flip'])
        config.check()

        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg;last-frame=examples/media/beach.jpg'],
            last_frame_image_processors=['flip', '+', 'mirror'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            image_seeds=['examples/media/earth.jpg'],
            last_frame_image_processors=['flip'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('last-frame=', str(raised.exception))

    def test_render_loop_video_processor_slots(self):
        import dgenerate.renderloop as _renderloop

        loop = _renderloop.RenderLoop.__new__(_renderloop.RenderLoop)
        loop._c_config = unittest.mock.Mock(
            last_frame_image_processors=None,
            reference_image_processors=None,
            wan_pose_image_processors=None,
            wan_face_image_processors=None,
            wan_driving_image_processors=None,
            wan_background_image_processors=None)
        start, end, control, last_frame, pose = object(), object(), object(), object(), object()

        def slots(seed, control_chain=None, mask_chain=None, extra=None):
            extra = extra or {}
            with unittest.mock.patch.object(
                    _renderloop.RenderLoop, '_load_seed_image_processors', return_value=seed), \
                    unittest.mock.patch.object(
                        _renderloop.RenderLoop, '_load_control_image_processors',
                        return_value=control_chain), \
                    unittest.mock.patch.object(
                        _renderloop.RenderLoop, '_load_mask_image_processors',
                        return_value=mask_chain), \
                    unittest.mock.patch.object(
                        _renderloop.RenderLoop, '_load_config_processors',
                        side_effect=lambda name: extra.get(name)):
                return loop._video_image_processors()

        empty = {
            'control': None, 'mask': None, 'reference': None,
            'wan_pose': None, 'wan_face': None,
            'wan_background': None, 'wan_driving': None}
        self.assertEqual(slots(start), {
            'start': start, 'end': start, **empty})
        self.assertEqual(slots([start, end], control), {
            'start': start, 'end': end, **empty, 'control': control})
        self.assertEqual(slots([None, end]), {
            'start': None, 'end': end, **empty})
        self.assertEqual(slots(start, extra={'last_frame_image_processors': last_frame}), {
            'start': start, 'end': last_frame, **empty})
        self.assertEqual(slots(start, extra={'wan_pose_image_processors': pose}), {
            'start': start, 'end': start, **empty, 'wan_pose': pose})
        with self.assertRaises(_renderloop.RenderLoopConfigError):
            slots([start, end, start])
        with self.assertRaises(_renderloop.RenderLoopConfigError):
            slots(None, [control, control])
        with self.assertRaises(_renderloop.RenderLoopConfigError):
            slots(start, extra={'wan_pose_image_processors': [pose, pose]})

    def test_legacy_ltx_rejects_control_mode(self):
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._ltx_pipeline_class('ltx-control', 'ltx')
        from diffusers import LTX2InContextPipeline
        self.assertIs(
            _videopipelines._ltx_pipeline_class('ltx-control', 'ltx2'), LTX2InContextPipeline)

    def test_lora_reference_downscale_factor(self):
        import safetensors.torch

        directory = tempfile.mkdtemp()
        ic_lora = os.path.join(directory, 'ic.safetensors')
        safetensors.torch.save_file(
            {'w': torch.zeros(1)}, ic_lora, metadata={'reference_downscale_factor': '2'})
        plain = os.path.join(directory, 'plain.safetensors')
        safetensors.torch.save_file({'w': torch.zeros(1)}, plain)

        factor = _videopipelines._ic_lora_downscale_factor
        self.assertEqual(factor(f'{ic_lora};scale=0.8', None, None, True), 2)
        self.assertEqual(factor(ic_lora, 3, None, True), 3)
        self.assertEqual(factor(plain, None, None, True), 1)
        self.assertEqual(factor(f'{directory};weight-name=ic.safetensors', None, None, True), 2)

    def test_ltx_reference_kwargs(self):
        class Pipe:
            vae_temporal_compression_ratio = 8
            vae_spatial_compression_ratio = 32

        class Held:
            reference_downscale_factor = 2

        class Wrapper:
            ltx_ic_lora_uri = 'ic.safetensors;attention=0.5'

        args = _pipelinewrapper.DiffusionArguments()
        args.reference_video_frames = [PIL.Image.new('RGB', (8, 8)) for _ in range(30)]
        kwargs = _videopipelines._ltx_reference_kwargs(Wrapper(), Pipe(), Held(), args, 49, 512, 512)
        self.assertEqual(kwargs['reference_downscale_factor'], 2)
        self.assertEqual(kwargs['conditioning_attention_strength'], 0.5)
        self.assertEqual(len(kwargs['reference_conditions'][0].frames), 25)
        self.assertEqual(kwargs['reference_conditions'][0].frames[0].size, (512, 512))

        wide = _pipelinewrapper.DiffusionArguments()
        wide.aspect_correct = False
        wide.reference_video_frames = [PIL.Image.new('RGB', (720, 480)) for _ in range(17)]
        wide_kwargs = _videopipelines._ltx_reference_kwargs(
            Wrapper(), Pipe(), Held(), wide, 17, 512, 512)
        self.assertEqual(wide_kwargs['reference_conditions'][0].frames[0].size, (512, 512))

        fitted = _pipelinewrapper.DiffusionArguments()
        fitted.aspect_correct = True
        fitted.reference_video_frames = [PIL.Image.new('RGB', (720, 480)) for _ in range(17)]
        fitted_kwargs = _videopipelines._ltx_reference_kwargs(
            Wrapper(), Pipe(), Held(), fitted, 17, 512, 512)
        self.assertEqual(fitted_kwargs['reference_conditions'][0].frames[0].size, (512, 320))

        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._ltx_reference_kwargs(Wrapper(), Pipe(), Held(), args, 49, 544, 512)

    def test_call_ltx_control_mode(self):
        class Scheduler:
            config = {'use_dynamic_shifting': False}

        class Pipe:
            scheduler = Scheduler()
            vae_temporal_compression_ratio = 8
            vae_spatial_compression_ratio = 32
            duration_head = None

        class Held:
            pipeline = Pipe()
            family = 'ltx2'
            reference_downscale_factor = 2

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
            _local_files_only = True
            _auth_token = None
            quantizer_uri = None
            quantizer_map = None
            transformer_uri = None
            lora_uris = ['org/style']
            lora_fuse_scale = None
            ltx_ic_lora_uri = 'org/ic;weight-name=ic.safetensors;attention=0.75;downscale=2'

        captured = {}

        def create(**kwargs):
            captured['cache'] = kwargs
            return Held()

        def invoke(wrapper, pipeline, kwargs):
            captured['kwargs'] = kwargs

            class Output:
                frames = [PIL.Image.new('RGB', (4, 4))]
                audio = None

            return Output()

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('fox')
        args.video_fps = 24
        args.width = 512
        args.height = 512
        args.reference_video_frames = [PIL.Image.new('RGB', (8, 8)) for _ in range(20)]
        args.images = [PIL.Image.new('RGB', (8, 8))]

        with unittest.mock.patch.object(
                _videopipelines, '_create_cached_video_pipeline',
                side_effect=create), \
                unittest.mock.patch.object(
                    _videopipelines, 'pipeline_for_mode',
                    side_effect=lambda pipeline, mode, family: captured.setdefault('mode', mode) and pipeline), \
                unittest.mock.patch.object(
                    _videopipelines._schedulers, 'load_scheduler'), \
                unittest.mock.patch.object(
                    _videopipelines, '_invoke', side_effect=invoke):
            _videopipelines._call_ltx(Wrapper(), args)

            without = Wrapper()
            without.ltx_ic_lora_uri = None
            with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
                _videopipelines._call_ltx(without, args)

        cache = captured['cache']
        self.assertEqual(cache['lora_uris'], ('org/style',))
        self.assertEqual(cache['ltx_ic_lora_uri'], 'org/ic;scale=1.0;weight-name=ic.safetensors')
        self.assertEqual(cache['ic_lora_downscale'], 2)

        kwargs = captured['kwargs']
        self.assertEqual(captured['mode'], 'ltx-control')
        self.assertEqual(kwargs['num_frames'], 17)
        self.assertEqual(len(kwargs['reference_conditions'][0].frames), 17)
        self.assertEqual(kwargs['reference_downscale_factor'], 2)
        self.assertEqual(kwargs['conditioning_attention_strength'], 0.75)
        self.assertEqual([c.index for c in kwargs['conditions']], [0])
        self.assertNotIn('image', kwargs)

    def test_end_is_video_only(self):
        config = _config(
            model_path='org/sd',
            image_seeds=['examples/media/earth.jpg;last-frame=examples/media/beach.jpg'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_video_rejects_batch_size_and_accepts_seed_strength(self):
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
        config.check()
        self.assertEqual(config.image_seed_strengths, [0.4])

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
            ltx_audio_guidance_scales=[1.0, 7.0],
            ltx_audio_guidance_rescales=[0.5, 0.7])
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

        with_vae = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            vae_uri='AutoencoderKLLTX2Video;model=org/vae')
        with_vae.check()

        with_text_encoders = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            text_encoder_uris=[
                'Gemma4UnifiedForConditionalGeneration;model=org/te'])
        with_text_encoders.check()

        with_text_encoder_default = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            text_encoder_uris=['+'])
        with_text_encoder_default.check()

        rejected_second_text_encoders = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            second_model_text_encoder_uris=[
                'Gemma4UnifiedForConditionalGeneration;model=org/te'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            rejected_second_text_encoders.check()

        wan_vae_quant_map = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            quantizer_uri='bnb;bits=4',
            quantizer_map=['vae'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as wan_vae_raised:
            wan_vae_quant_map.check()
        self.assertIn('vae', str(wan_vae_raised.exception))

        wan_encoder_map = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            quantizer_uri='bnb;bits=4',
            quantizer_map=['transformer', 'transformer_2', 'image_encoder'])
        wan_encoder_map.check()

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
            ltx_audio_guidance_scales=[7.0],
            ltx_audio_guidance_rescales=[0.5])
        audio.check()
        steps = list(audio.iterate_diffusion_args())
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].ltx_audio_guidance_scale, 7.0)
        self.assertEqual(steps[0].ltx_audio_guidance_rescale, 0.5)

        still = _config(
            model_path='org/sd',
            ltx_audio_guidance_scales=[7.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still.check()
        self.assertIn('ltx', str(raised.exception).lower())

        still_rescale = _config(
            model_path='org/sd',
            ltx_audio_guidance_rescales=[0.5])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_rescale.check()
        self.assertIn('ltx', str(raised.exception).lower())

        still_length = _config(
            model_path='org/sd',
            video_lengths=[2.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_length.check()
        self.assertIn('video', str(raised.exception).lower())

        still_fps = _config(
            model_path='org/sd',
            video_fps=[24.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            still_fps.check()
        self.assertIn('video', str(raised.exception).lower())

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
        self.assertIn('always tile', str(raised.exception))

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

    def test_video_text_encoder_slots(self):
        index = {
            'text_encoder_2': ['transformers', 'CLIPTextModel'],
            'text_encoder': ['transformers', 'UMT5EncoderModel'],
            'vae': ['diffusers', 'AutoencoderKLWan'],
            '_class_name': 'WanPipeline',
        }
        self.assertEqual(
            _videopipelines._text_encoder_slots(index),
            ['text_encoder', 'text_encoder_2'])
        self.assertEqual(
            _videopipelines._inject_video_text_encoders(
                index, ['+'], 'float16', None, True, 'cpu', False),
            {})
        self.assertEqual(
            _videopipelines._inject_video_text_encoders(
                index, ['null'], 'float16', None, True, 'cpu', False),
            {'text_encoder': None})
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._inject_video_text_encoders(
                index, ['a', 'b', 'c'], 'float16', None, True, 'cpu', False)
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._inject_video_text_encoders(
                {'_class_name': 'X'}, ['+'], 'float16', None, True, 'cpu', False)

    def test_wan_vae_dtype_skips_quantized(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _QuantVae:
            is_quantized = True

            def to(self, *args, **kwargs):
                raise AssertionError('quantized VAE must not be cast')

        pipe = unittest.mock.Mock()
        pipe.vae = _QuantVae()
        _wan._set_wan_vae_dtype(pipe)
        self.assertTrue(_wan._vae_is_quantized(pipe.vae))

    def test_cli_quantizer_map_accepts_wan_component_names(self):
        import argparse

        from dgenerate.arguments import _type_quantizer_map

        self.assertEqual(_type_quantizer_map('image_encoder'), 'image_encoder')
        self.assertEqual(_type_quantizer_map('transformer_2'), 'transformer_2')
        self.assertEqual(_type_quantizer_map('connectors'), 'connectors')
        with self.assertRaises(argparse.ArgumentTypeError):
            _type_quantizer_map('vae')

    def test_wan_animate_2_skips_group_offload_when_quantized(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _QuantTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.quantization_config = {'quant_method': 'sdnq'}

        class _Plain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        pipe = unittest.mock.Mock()
        pipe.transformer = _QuantTransformer()
        pipe.text_encoder = _Plain()
        pipe.image_encoder = None
        pipe.vae = _Plain()
        pipe.video_processor = None
        pipe.image_processor = None
        pipe.guider = None

        with unittest.mock.patch(
                'diffusers.hooks.apply_group_offloading') as apply_offload:
            _wan.place_wan_animate_2(pipe, 'cpu', True, False, False)
        apply_offload.assert_not_called()

    def test_wan_animate_2_group_offloads_full_precision_transformer(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _Plain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        pipe = unittest.mock.Mock()
        pipe.transformer = _Plain()
        pipe.text_encoder = _Plain()
        pipe.image_encoder = None
        pipe.vae = _Plain()
        pipe.video_processor = None
        pipe.image_processor = None
        pipe.guider = None

        with unittest.mock.patch(
                'diffusers.hooks.apply_group_offloading') as apply_offload:
            _wan.place_wan_animate_2(pipe, 'cpu', True, False, False)
        apply_offload.assert_called_once()
        self.assertTrue(hasattr(pipe.text_encoder, '_hf_hook'))
        self.assertFalse(hasattr(pipe.vae, '_hf_hook'))

    def test_wan_animate_2_warns_that_offload_flags_share_placement(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _Plain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        def pipe():
            value = unittest.mock.Mock()
            value.transformer = _Plain()
            value.text_encoder = _Plain()
            value.image_encoder = None
            value.vae = _Plain()
            value.video_processor = None
            value.image_processor = None
            value.guider = None
            return value

        warnings = []
        with unittest.mock.patch(
                'diffusers.hooks.apply_group_offloading'), \
                unittest.mock.patch.object(
                    _wan._messages, 'warning', side_effect=warnings.append):
            _wan.place_wan_animate_2(pipe(), 'cpu', False, False, False)
            self.assertEqual(warnings, [])
            _wan.place_wan_animate_2(pipe(), 'cpu', False, False, True)
        self.assertEqual(len(warnings), 1)
        self.assertIn('--model-group-offload', warnings[0])
        self.assertIn('one placement', warnings[0])

    def test_wan_animate_2_compiles_transformer_blocks(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _Block(torch.nn.Module):
            def forward(self, x):
                return x

        class _Model(torch.nn.Module):
            _repeated_blocks = ['_Block']

            def __init__(self):
                super().__init__()
                self.block = _Block()

        class _Plain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        def pipe(transformer):
            value = unittest.mock.Mock()
            value.transformer = transformer
            value.text_encoder = None
            value.image_encoder = None
            value.vae = None
            value.video_processor = None
            value.image_processor = None
            value.guider = None
            return value

        model = _Model()
        plain = _Plain()
        original_forward = model.block.forward.__func__
        with unittest.mock.patch(
                'importlib.util.find_spec', return_value=None):
            self.assertTrue(_wan._flex_attention_can_compile('mps'))
            self.assertFalse(_wan._flex_attention_can_compile('cuda'))
        self.assertFalse(_wan._flex_attention_can_compile('cpu'))
        warnings = []
        with unittest.mock.patch.object(
                _wan._messages, 'warning', side_effect=warnings.append):
            _wan.place_wan_animate_2(pipe(plain), 'cpu', False, False, False)
            _wan.place_wan_animate_2(pipe(model), 'cpu', False, False, False)
        self.assertIs(model.block.forward.__func__, original_forward)
        self.assertTrue(any('full score matrix' in message for message in warnings))

        with unittest.mock.patch(
                'importlib.util.find_spec', return_value=object()), \
                unittest.mock.patch(
                    'diffusers.hooks.apply_group_offloading') as apply_offload:
            _wan.place_wan_animate_2(pipe(model), 'cuda', True, False, False)
            compiled_forward = model.block.forward
            self.assertTrue(model.block._dgenerate_animate2_forward_compiled)
            _wan.place_wan_animate_2(pipe(model), 'cuda', True, False, False)
        self.assertIs(model.block.forward, compiled_forward)
        apply_offload.assert_called()
        self.assertEqual(apply_offload.call_args.kwargs['num_blocks_per_group'], 1)
        self.assertTrue(apply_offload.call_args.kwargs['use_stream'])

    def test_load_quantized_module_applies_architecture_skips(self):
        import diffusers

        captured = {}

        class FakeTransformer:
            __name__ = 'WanTransformer3DModel'
            __module__ = 'diffusers.models.transformers.transformer_wan'

            @classmethod
            def from_pretrained(cls, model_path, **kwargs):
                captured['quantization_config'] = kwargs.get('quantization_config')
                module = unittest.mock.Mock()
                module.is_loaded_in_8bit = False
                return module

        FakeTransformer.__name__ = 'WanTransformer3DModel'

        with unittest.mock.patch.object(
                _videopipelines, '_quantizer_config',
                return_value=diffusers.BitsAndBytesConfig(load_in_4bit=True)):
            _videopipelines._load_quantized_module(
                FakeTransformer, 'org/wan', 'transformer', None, None,
                _pipelinewrapper.DataType.BFLOAT16, 'bnb;bits=4', None, True,
                None, offload=False)

        skipped = captured['quantization_config'].llm_int8_skip_modules
        self.assertIn('scale_shift_table', skipped)
        self.assertIn('patch_embedding', skipped)
        self.assertIn('condition_embedder', skipped)

    def test_wan_animate_2_null_text_encoder_not_reloaded(self):
        from dgenerate.pipelinewrapper.videopipelines import wan as _wan

        class _Spec:
            def __init__(self):
                self.default_creation_method = 'from_pretrained'
                self.pretrained_model_name_or_path = 'org/wan'

        class _FakePipe:
            def __init__(self, blocks=None, pretrained_model_name_or_path=None):
                self._component_specs = {
                    'text_encoder': _Spec(),
                    'vae': _Spec(),
                    'transformer': _Spec(),
                }
                self.text_encoder = None
                self.vae = None
                self.transformer = None
                self._updated = []
                self._loaded_names = None

            def update_components(self, **kwargs):
                self._updated.append(dict(kwargs))
                for name, value in kwargs.items():
                    setattr(self, name, value)

            def load_components(self, names=None, **kwargs):
                self._loaded_names = list(names) if names is not None else None
                for name in names or ():
                    setattr(self, name, f'loaded-{name}')

        class _FakePipelineClass:
            __name__ = 'WanAnimate2ModularPipeline'

            def __new__(cls, blocks=None, pretrained_model_name_or_path=None):
                return _FakePipe(blocks, pretrained_model_name_or_path)

        class _FakeBlocks:
            pass

        transformer = object()
        with unittest.mock.patch(
                'diffusers.WanAnimate2Blocks', _FakeBlocks), \
                unittest.mock.patch(
                    'diffusers.WanAnimate2DistilledBlocks', _FakeBlocks):
            pipe = _wan.load_wan_animate_2_pipeline(
                _FakePipelineClass,
                'org/wan',
                {'local_files_only': True},
                {'transformer': transformer, 'text_encoder': None},
                _pipelinewrapper.DataType.BFLOAT16)

        self.assertIs(pipe.transformer, transformer)
        self.assertIsNone(pipe.text_encoder)
        self.assertEqual(pipe._loaded_names, ['vae'])
        self.assertIn({'text_encoder': None}, pipe._updated)

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
            _videopipelines._quantize_component('ltx_prompt_enhancer', 'sdnq', None))
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
            'ltx_prompt_enhancer': ['transformers', 'Gemma4ForConditionalGeneration'],
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
        self.assertNotIn('ltx_prompt_enhancer', modules)
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
        source.ltx_audio_guidance_scale = 7.0
        source.ltx_audio_guidance_rescale = 0.5
        dest = _pipelinewrapper.DiffusionArguments()
        _videopipelines.apply_video_arg_rewrites(source, dest)
        self.assertEqual(dest.guidance_scale, 3.0)
        self.assertEqual(dest.inference_steps, 30)
        self.assertEqual(dest.ltx_audio_guidance_scale, 7.0)
        self.assertEqual(dest.ltx_audio_guidance_rescale, 0.5)

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

    def test_legacy_ltx_condition_schedule_without_mu(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        # scheduler_config.json from Lightricks/LTX-Video 0.9.0
        config = {
            'base_image_seq_len': 1024, 'base_shift': 0.95, 'invert_sigmas': False,
            'max_image_seq_len': 4096, 'max_shift': 2.05, 'num_train_timesteps': 1000,
            'shift': 1.0, 'shift_terminal': 0.1, 'use_beta_sigmas': False,
            'use_dynamic_shifting': True, 'use_exponential_sigmas': False,
            'use_karras_sigmas': False,
        }

        class Pipe:
            vae_spatial_compression_ratio = 32
            vae_temporal_compression_ratio = 8

        pipe = Pipe()
        original = FlowMatchEulerDiscreteScheduler.from_config(config)
        pipe.scheduler = original
        expected = _videopipelines._legacy_ltx_timesteps(original, 512, 512, 121, 50)

        timesteps = _videopipelines._fix_legacy_ltx_condition_schedule(pipe, 512, 512, 121, 50)
        self.assertIsNot(pipe.scheduler, original)
        self.assertTrue(original.config.use_dynamic_shifting)
        self.assertFalse(pipe.scheduler.config.use_dynamic_shifting)

        # the condition pipeline calls set_timesteps(timesteps=...) with no mu
        pipe.scheduler.set_timesteps(timesteps=timesteps, device='cpu')
        self.assertTrue(torch.allclose(pipe.scheduler.timesteps, expected, atol=1e-3))

        static = FlowMatchEulerDiscreteScheduler.from_config({**config, 'use_dynamic_shifting': False})
        pipe.scheduler = static
        self.assertIsNone(_videopipelines._fix_legacy_ltx_condition_schedule(pipe, 512, 512, 121, 50))
        self.assertIs(pipe.scheduler, static)

    def test_ltx_allows_frame_slice(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            frame_start=3,
            frame_end=10)
        config.check()

    def test_load_rgb_frames_slices_video(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'clip.mp4')
        with _mediaoutput.VideoWriter(path, 12) as writer:
            for index in range(12):
                writer.write(PIL.Image.new('RGB', (64, 64), (index * 20, 0, 0)))

        frames, fps = _videopipelines.load_rgb_frames(path, frame_start=2, frame_end=9)
        try:
            self.assertEqual(len(frames), 8)
            self.assertAlmostEqual(fps, 12.0)
            self.assertTrue(all(frame.mode == 'RGB' for frame in frames))
        finally:
            for frame in frames:
                frame.close()

        frames, _ = _videopipelines.load_rgb_frames(path, max_frames=5)
        self.assertEqual(len(frames), 5)
        for frame in frames:
            frame.close()

        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.load_rgb_frames(path, frame_start=50)

        frames, fps = _videopipelines.load_rgb_frames('examples/media/earth.jpg')
        self.assertEqual(len(frames), 1)
        self.assertIsNone(fps)
        frames[0].close()

    def test_generic_download_mimetype_uses_extension(self):
        url = ('https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/'
               'examples/wan_animate/animate/video.mp4')
        mime = _mediainput.normalize_downloaded_mimetype(
            url, 'application/octet-stream')
        self.assertEqual(mime, 'video/mp4')
        self.assertTrue(_mediainput.mimetype_is_video(mime))
        self.assertFalse(_mediainput.mimetype_is_static_image(mime))

        mime = _mediainput.normalize_downloaded_mimetype(
            'https://example.test/download?id=1',
            'Application/Octet-Stream; charset=binary',
            cached_path=os.path.join('cache', 'clip.mp4'))
        self.assertTrue(_mediainput.mimetype_is_video(mime))

        mime = _mediainput.normalize_downloaded_mimetype(
            'https://example.test/photo.jpg', 'application/octet-stream')
        self.assertEqual(mime, 'image/jpeg')
        self.assertTrue(_mediainput.mimetype_is_static_image(mime))

        mime = _mediainput.normalize_downloaded_mimetype(
            'https://example.test/data.pfm', 'application/octet-stream')
        self.assertEqual(mime, 'application/octet-stream')
        self.assertTrue(_mediainput.mimetype_is_static_image(mime))

        mime = _mediainput.normalize_downloaded_mimetype(
            'https://example.test/weights.safetensors', 'application/octet-stream')
        self.assertEqual(mime, 'application/octet-stream')

        mime = _mediainput.normalize_downloaded_mimetype(
            'https://example.test/a.png', 'image/png; charset=binary')
        self.assertEqual(mime, 'image/png')

        with unittest.mock.patch(
                'dgenerate.mediainput._webcache.request_mimetype',
                return_value='application/octet-stream') as request:
            self.assertEqual(_mediainput.request_mimetype(url), 'video/mp4')
            request.assert_called_once_with(url, local_files_only=False)

    def test_load_rgb_frames_reads_octet_stream_video_url(self):
        url = ('https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/'
               'examples/wan_animate/animate/video.mp4')
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'clip.mp4')
            with _mediaoutput.VideoWriter(path, 12) as writer:
                for index in range(4):
                    writer.write(PIL.Image.new('RGB', (64, 64), (index * 40, 0, 0)))

            def fake_cache(uri, local_files_only=False, **kwargs):
                self.assertEqual(uri, url)
                return 'application/octet-stream', path

            frames = []
            try:
                with unittest.mock.patch(
                        'dgenerate.mediainput.create_web_cache_file', fake_cache):
                    frames, fps = _videopipelines.load_rgb_frames(url, max_frames=3)
                self.assertEqual(len(frames), 3)
                self.assertAlmostEqual(fps, 12.0)
                self.assertTrue(all(frame.mode == 'RGB' for frame in frames))
            finally:
                for frame in frames:
                    frame.close()

    def test_ltx_conditions_place_clips(self):
        class Pipe:
            vae_temporal_compression_ratio = 8
            duration_head = None

        start = [PIL.Image.new('RGB', (8, 8)) for _ in range(30)]
        end = [PIL.Image.new('RGB', (8, 8)) for _ in range(20)]
        args = _pipelinewrapper.DiffusionArguments()
        args.video_frames = start
        args.end_video_frames = end

        legacy = _videopipelines._ltx_conditions(Pipe(), 'ltx', args, 49)
        self.assertEqual([len(c.video) for c in legacy], [25, 17])
        self.assertEqual([c.frame_index for c in legacy], [0, 32])

        ltx2 = _videopipelines._ltx_conditions(Pipe(), 'ltx2', args, 49)
        self.assertEqual([len(c.frames) for c in ltx2], [25, 17])
        self.assertEqual([c.index for c in ltx2], [0, 4])

        args.end_video_frames = None
        args.end_images = [PIL.Image.new('RGB', (8, 8))]
        mixed = _videopipelines._ltx_conditions(Pipe(), 'ltx2', args, 49)
        self.assertEqual(len(mixed[0].frames), 25)
        self.assertEqual(mixed[1].index, -1)

    def test_ltx_end_clip_needs_length_with_duration_head(self):
        class Pipe:
            vae_temporal_compression_ratio = 8
            duration_head = object()

        args = _pipelinewrapper.DiffusionArguments()
        args.end_video_frames = [PIL.Image.new('RGB', (8, 8)) for _ in range(9)]
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines._ltx_conditions(Pipe(), 'ltx2', args, None)

        args.end_video_frames = None
        args.video_frames = [PIL.Image.new('RGB', (8, 8)) for _ in range(12)]
        conditions = _videopipelines._ltx_conditions(Pipe(), 'ltx2', args, None)
        self.assertEqual(len(conditions[0].frames), 9)

    def test_ltx_condition_index_and_extra_group(self):
        earth = 'examples/media/earth.jpg'
        beach = 'examples/media/beach.jpg'
        parsed = _mediainput.parse_image_seed_uri(
            f'{earth};ltx-index=0;strength=0.5 ++ {beach};ltx-index=8;strength=1')
        self.assertEqual(parsed.ltx_condition_index, 0)
        self.assertEqual(parsed.ltx_condition_strength, 0.5)
        self.assertEqual(parsed.ltx_extra_conditions[0].images, [beach])
        self.assertEqual(parsed.ltx_extra_conditions[0].ltx_condition_index, 8)
        self.assertFalse(parsed.is_single_spec)
        ltx = _pipelinewrapper.ModelType.LTX
        self.assertEqual(
            _videopipelines.classify_video_seed(ltx, parsed), 'ltx-condition')

        placed = _mediainput.parse_image_seed_uri(f'{earth};ltx-index=8')
        self.assertEqual(
            _videopipelines.classify_video_seed(ltx, placed), 'ltx-condition')

        softened = _pipelinewrapper.DiffusionArguments()
        softened.image_seed_strength = 0.7
        softened.images = [PIL.Image.new('RGB', (8, 8))]
        self.assertEqual(_videopipelines._ltx_resolved_strength(None, softened), 0.7)
        self.assertEqual(_videopipelines._ltx_resolved_strength(0.4, softened), 0.4)
        self.assertTrue(_videopipelines._ltx_image_needs_conditions(softened))

    def test_latent_upscale_size_check(self):
        missing = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            ltx_latent_upscale=True)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            missing.check()

        odd = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            ltx_latent_upscale=True,
            output_size=(640, 352))
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            odd.check()

        aligned = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            ltx_latent_upscale=True,
            output_size=(768, 512))
        aligned.check()

        bounds = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            prompts=[_prompt.Prompt('a')],
            ltx_video_min_seconds=[2.0, 4.0],
            ltx_video_max_seconds=[6.0, 8.0])
        bounds.check()
        self.assertEqual(bounds.calculate_generation_steps(), 2)
        pairs = [
            (arg.ltx_video_min_seconds, arg.ltx_video_max_seconds)
            for arg in bounds.iterate_diffusion_args()]
        self.assertEqual(pairs, [(2.0, 6.0), (4.0, 8.0)])

    def test_ltx_two_stage_is_one_generation(self):
        class Scheduler:
            def __init__(self):
                self.config = {'use_dynamic_shifting': True, 'shift_terminal': 0.1}

            @classmethod
            def from_config(cls, config, **overrides):
                scheduler = cls()
                scheduler.config = dict(config)
                scheduler.config.update(overrides)
                return scheduler

        class Pipe:
            def __init__(self):
                self.scheduler = Scheduler()
                self.vae = object()

            def __call__(self, stg_scale=None, audio_stg_scale=None, modality_scale=None,
                         audio_modality_scale=None, spatio_temporal_guidance_blocks=None,
                         use_cross_timestep=True, image_crf=None, min_seconds=1.0,
                         max_seconds=20.0, enable_prompt_enhancement=False, system_prompt=None,
                         decode_timestep=0.0, decode_noise_scale=None, **kwargs):
                raise AssertionError('call_pipeline should be used')

        calls = []

        def invoke(wrapper, pipeline, kwargs):
            calls.append(dict(kwargs))

            class Output:
                pass

            output = Output()
            if kwargs.get('output_type') == 'latent' and 'latents' not in kwargs:
                output.frames = torch.zeros(1, 4, 3, 2, 2)
                output.audio = torch.zeros(1, 2, 4)
            else:
                output.frames = [PIL.Image.new('RGB', (4, 4))]
                output.audio = None
            return output

        class UpPipe:
            def __init__(self, vae, latent_upsampler):
                self.vae = vae

            def enable_sequential_cpu_offload(self, device=None):
                return None

            def enable_model_cpu_offload(self, device=None):
                return None

            def __call__(self, **kwargs):
                calls.append({'upsample': tuple(kwargs['latents'].shape)})
                return (torch.zeros(1, 4, 3, 4, 4),)

        class Held:
            pipeline = Pipe()
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
            _local_files_only = True
            _auth_token = None
            quantizer_uri = None
            quantizer_map = None
            transformer_uri = None
            lora_uris = None
            lora_fuse_scale = None

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('fox')
        args.inference_steps = 8
        args.guidance_scale = 1
        args.ltx_audio_guidance_scale = 1
        args.video_fps = 24
        args.video_length = 5
        args.width = 768
        args.height = 512
        args.ltx_latent_upscale = True
        args.ltx_stg_scale = 1
        args.ltx_modality_scale = 3

        with unittest.mock.patch.object(
                _videopipelines, '_create_cached_video_pipeline', return_value=Held()), \
                unittest.mock.patch.object(
                    _videopipelines, 'pipeline_for_mode',
                    side_effect=lambda pipeline, mode, family: pipeline), \
                unittest.mock.patch.object(
                    _videopipelines._schedulers, 'load_scheduler'), \
                unittest.mock.patch.object(
                    _videopipelines, '_invoke', side_effect=invoke), \
                unittest.mock.patch(
                    'diffusers.pipelines.ltx2.latent_upsampler.LTX2LatentUpsamplerModel.from_pretrained',
                    return_value=type('Upsampler', (), {'to': lambda self, *args, **kwargs: self})()), \
                unittest.mock.patch(
                    'diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample.LTX2LatentUpsamplePipeline',
                    UpPipe):
            frames, audio, sample_rate, fps = _videopipelines._call_ltx(Wrapper(), args)

        self.assertEqual(len(frames), 1)
        self.assertEqual(fps, 24)
        self.assertIsNone(audio)
        self.assertIsNone(sample_rate)
        self.assertEqual(calls[0]['width'], 384)
        self.assertEqual(calls[0]['height'], 256)
        self.assertEqual(calls[0]['output_type'], 'latent')
        self.assertEqual(calls[0]['stg_scale'], 1)
        self.assertEqual(calls[0]['spatio_temporal_guidance_blocks'], [28])
        self.assertEqual(calls[0]['modality_scale'], 3)
        self.assertEqual(calls[1]['upsample'], (1, 4, 3, 2, 2))
        self.assertEqual(calls[2]['width'], 768)
        self.assertEqual(calls[2]['height'], 512)
        self.assertEqual(calls[2]['guidance_scale'], 1)
        self.assertEqual(calls[2]['noise_scale'], calls[2]['sigmas'][0])
        self.assertEqual(len(calls[2]['sigmas']), 3)
        self.assertNotIn('ltx_stg_scale', calls[2])
        self.assertIn('latents', calls[2])

    def test_two_stage_enhances_once_and_hides_enhancer_from_stage_lora(self):
        class Scheduler:
            def __init__(self):
                self.config = {'use_dynamic_shifting': True, 'shift_terminal': 0.1}

            @classmethod
            def from_config(cls, config, **overrides):
                scheduler = cls()
                scheduler.config = dict(config)
                scheduler.config.update(overrides)
                return scheduler

        class Enhancer:
            def to(self, *args, **kwargs):
                return self

        enhancer = Enhancer()
        calls = []

        class Pipe:
            def __init__(self):
                self.scheduler = Scheduler()
                self.vae = object()
                self.prompt_enhancer = enhancer

            def enhance_prompt(self, prompt, system_prompt=None, **kwargs):
                calls.append({'enhance': prompt, 'system': system_prompt})
                return [prompt + ' enhanced']

        def invoke(wrapper, pipeline, kwargs):
            calls.append(dict(kwargs))

            class Output:
                pass

            output = Output()
            if kwargs.get('output_type') == 'latent' and 'latents' not in kwargs:
                output.frames = torch.zeros(1, 4, 3, 2, 2)
                output.audio = None
            else:
                output.frames = [PIL.Image.new('RGB', (4, 4))]
                output.audio = None
            return output

        class UpPipe:
            def __init__(self, vae, latent_upsampler):
                pass

            def __call__(self, **kwargs):
                return (torch.zeros(1, 4, 3, 4, 4),)

        seen = {}

        def load_on_pipeline(pipeline, **kwargs):
            seen['during_lora'] = pipeline.prompt_enhancer

        class Wrapper:
            device = 'cpu'
            model_path = 'org/ltx'
            model_cpu_offload = False
            model_sequential_offload = True
            lora_fuse_scale = None
            _dtype = None
            _auth_token = None
            _local_files_only = True

        pipe = Pipe()
        wrapper = Wrapper()

        args = _pipelinewrapper.DiffusionArguments()
        args.ltx_stage_lora_uris = ['org/ltx;weight-name=stage.safetensors']
        kwargs = {
            'prompt': 'fox',
            'enable_prompt_enhancement': True,
            'width': 768,
            'height': 512,
            'guidance_scale': 3,
        }

        with unittest.mock.patch.object(_videopipelines, '_invoke', side_effect=invoke), \
                unittest.mock.patch.object(
                    _videopipelines._uris.LoRAUri, 'load_on_pipeline', side_effect=load_on_pipeline), \
                unittest.mock.patch(
                    'diffusers.pipelines.ltx2.latent_upsampler.LTX2LatentUpsamplerModel.from_pretrained',
                    return_value=type('Upsampler', (), {'to': lambda self, *args, **kwargs: self})()), \
                unittest.mock.patch(
                    'diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample.LTX2LatentUpsamplePipeline',
                    UpPipe):
            _videopipelines._ltx_two_stage(wrapper, pipe, None, args, kwargs, False)

        self.assertEqual(calls[0]['enhance'], 'fox')
        self.assertTrue(calls[0]['system'])
        self.assertEqual(calls[1]['prompt'], 'fox enhanced')
        self.assertFalse(calls[1]['enable_prompt_enhancement'])
        self.assertEqual(calls[2]['prompt'], 'fox enhanced')
        self.assertFalse(calls[2]['enable_prompt_enhancement'])
        self.assertIsNone(seen['during_lora'])
        self.assertIs(pipe.prompt_enhancer, enhancer)

    def test_diffusion_decoder_does_not_fall_back_to_flex(self):
        import diffusers.models.autoencoders.ltx2_diffusion_decoder as decoder_module

        class MissingKernels:
            def __init__(self):
                raise ImportError(
                    'Install it with `pip install kernels`, or use the default '
                    '`LTX2VideoVaeNeighborhoodAttnProcessor` (FlexAttention) instead.')

        decoder = unittest.mock.Mock()
        wrapper = unittest.mock.Mock(
            model_path='org/ltx',
            _dtype=_pipelinewrapper.DataType.BFLOAT16,
            _auth_token=None,
            _local_files_only=True,
            device='cpu',
            model_cpu_offload=False,
            model_sequential_offload=False,
            model_group_offload=False)
        held = unittest.mock.Mock(spec=[])

        with unittest.mock.patch(
                'diffusers.LTX2VideoDiffusionDecoderModel.from_pretrained',
                return_value=decoder), \
                unittest.mock.patch.object(
                    decoder_module, 'LTX2VideoVaeNeighborhoodNattenProcessor', MissingKernels):
            with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError) as raised:
                _videopipelines._decode_ltx_diffusion(wrapper, None, held, None, None)

        self.assertIn('kernels', str(raised.exception))
        self.assertIn('does not fit in GPU memory', str(raised.exception))
        decoder.set_attn_processor.assert_not_called()


class TestWanModels(unittest.TestCase):
    def test_wan_check_defaults(self):
        config = _config(model_path='org/wan', model_type=_pipelinewrapper.ModelType.WAN)
        config.check()
        self.assertEqual(config.video_fps, [16.0])

        animate = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg;wan-pose=examples/media/rickroll-roll.gif;wan-face=examples/media/rickroll-roll.gif'])
        animate.check()
        self.assertEqual(animate.video_fps, [30.0])

    def test_wan_rejects_video_lengths_for_animate(self):
        config = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg'],
            video_lengths=[2.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('video_lengths', str(raised.exception))

    def test_wan_scheduler_uri(self):
        rejected = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            scheduler_uri='EulerDiscreteScheduler')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            rejected.check()
        self.assertIn('FlowMatchEulerDiscreteScheduler', str(raised.exception))

        allowed = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            scheduler_uri='FlowMatchEulerDiscreteScheduler;shift=5.0')
        allowed.check()

        animate_rejected = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg;wan-pose=examples/media/rickroll-roll.gif;wan-face=examples/media/rickroll-roll.gif'],
            scheduler_uri='FlowMatchEulerDiscreteScheduler')
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            animate_rejected.check()
        self.assertIn('UniPCMultistepScheduler', str(raised.exception))

    def test_wan_animate_requires_pose_or_driving(self):
        missing = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            missing.check()
        self.assertIn('wan-pose=', str(raised.exception))

        driving = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            driving.check()
        self.assertIn('wan-animate-preprocess', str(raised.exception))

        preprocess = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif'],
            wan_animate_preprocess=True)
        preprocess.check()

        custom = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            image_seeds=['examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif'],
            wan_pose_image_processors=['openpose'],
            wan_face_image_processors=['yolo;model=Bingsu/adetailer;weight-name=face_yolov8n.pt'])
        custom.check()

    def test_image_processor_cli_names(self):
        parsed = _arguments.parse_args([
            'org/wan-animate',
            '--model-type', 'wan-animate',
            '--dtype', 'bfloat16',
            '--image-seeds',
            'examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif',
            '--wan-pose-image-processors', 'openpose',
            '--wan-face-image-processors', 'openpose',
            '--wan-driving-image-processors', 'flip',
            '--prompts', 'a dancer',
        ], throw=True, log_error=False)
        self.assertEqual(parsed.wan_pose_image_processors, ['openpose'])
        self.assertEqual(parsed.wan_face_image_processors, ['openpose'])
        self.assertEqual(parsed.wan_driving_image_processors, ['flip'])

        parsed = _arguments.parse_args([
            'org/ltx',
            '--model-type', 'ltx',
            '--dtype', 'bfloat16',
            '--image-seeds',
            'examples/media/earth.jpg;last-frame=examples/media/beach.jpg',
            '--last-frame-image-processors', 'flip',
            '--prompts', 'a fox',
        ], throw=True, log_error=False)
        self.assertEqual(parsed.last_frame_image_processors, ['flip'])

        parsed = _arguments.parse_args([
            'org/wan',
            '--model-type', 'wan',
            '--dtype', 'bfloat16',
            '--image-seeds',
            'examples/media/earth.jpg;control=examples/media/rickroll-roll.gif;'
            'reference=examples/media/mountain.png',
            '--reference-image-processors', 'grayscale',
            '--prompts', 'a dancer',
        ], throw=True, log_error=False)
        self.assertEqual(parsed.reference_image_processors, ['grayscale'])

    def test_wan_rejects_ltx_options(self):
        config = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            ltx_audio_guidance_scales=[7.0])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('ltx', str(raised.exception).lower())

    def test_ltx_rejects_wan_options(self):
        config = _config(
            model_path='org/ltx',
            model_type=_pipelinewrapper.ModelType.LTX,
            wan_boundary_ratios=[0.9])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('wan', str(raised.exception).lower())

    def test_wan_family_from_index(self):
        self.assertEqual(
            _videopipelines.wan_family_from_index({'_class_name': 'WanPipeline'}),
            'wan-t2v')
        self.assertEqual(
            _videopipelines.wan_family_from_index({'_class_name': 'WanImageToVideoPipeline'}),
            'wan-i2v')
        self.assertEqual(
            _videopipelines.wan_family_from_index({'_class_name': 'WanVACEPipeline'}),
            'wan-vace')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.wan_family_from_index({'_class_name': 'WanAnimatePipeline'})
        self.assertEqual(
            _videopipelines.wan_animate_family_from_index(
                {'_class_name': 'WanAnimatePipeline'}),
            'wan-animate')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.wan_animate_family_from_index({'_class_name': 'WanPipeline'})
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError) as raised:
            _videopipelines.wan_family_from_index({'_class_name': 'WanAnimate2ModularPipeline'})
        self.assertIn('wan-animate-2', str(raised.exception))
        self.assertEqual(
            _videopipelines.wan_animate_2_family_from_index({
                '_class_name': 'WanAnimate2Pipeline',
                'scheduler': ['diffusers', 'DPMSolverMultistepScheduler'],
            }),
            'wan-animate-2')
        self.assertEqual(
            _videopipelines.wan_animate_2_family_from_index({
                '_class_name': 'WanAnimate2Pipeline',
                'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
            }),
            'wan-animate-2-distilled')

    def test_wan_num_frames(self):
        self.assertEqual(_videopipelines.wan_num_frames(5, 16, 4), 81)
        self.assertEqual(_videopipelines.wan_num_frames(1, 16, 4), 17)
        self.assertEqual((_videopipelines.wan_num_frames(2, 16, 4) - 1) % 4, 0)

    def test_wan_flf_clip_kind(self):
        first = unittest.mock.Mock()
        first.config.image_dim = 1280
        first.config.pos_embed_seq_len = None
        first.condition_embedder = None
        self.assertEqual(_videopipelines._wan_flf_clip_kind(first), 'first')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError) as raised:
            _videopipelines._require_wan_flf_clip(unittest.mock.Mock(transformer=first))
        self.assertIn('FLF2V', str(raised.exception))

        flf = unittest.mock.Mock()
        flf.config.image_dim = 1280
        flf.config.pos_embed_seq_len = 514
        self.assertEqual(_videopipelines._wan_flf_clip_kind(flf), 'flf')
        _videopipelines._require_wan_flf_clip(unittest.mock.Mock(transformer=flf))

        later = unittest.mock.Mock()
        later.config.image_dim = None
        self.assertEqual(_videopipelines._wan_flf_clip_kind(later), 'none')
        _videopipelines._require_wan_flf_clip(unittest.mock.Mock(transformer=later))
        _videopipelines._require_wan_flf_clip(unittest.mock.Mock(transformer=None))

    def test_control_mask_without_image_is_wan_vace_only(self):
        seed = (
            'control=examples/media/rickroll-roll.gif;'
            'mask=examples/media/dog-on-bench-mask.png')
        wan = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            prompts=['a hiker'],
            image_seeds=[seed])
        wan.check()

        other = _config(
            model_path='org/sd',
            model_type=_pipelinewrapper.ModelType.SD,
            prompts=['a hiker'],
            image_seeds=[seed])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            other.check()
        self.assertIn('Wan VACE', str(raised.exception))

        animate = _config(
            model_path='org/wan-animate',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE,
            prompts=['a hiker'],
            wan_animate_preprocess=True,
            image_seeds=[
                'control=examples/media/rickroll-roll.gif;'
                'mask=examples/media/dog-on-bench-mask.png;'
                'wan-driving=examples/media/rickroll-roll.gif'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            animate.check()
        self.assertIn('character image', str(raised.exception))

    def test_classify_wan_seed(self):
        self.assertEqual(
            _videopipelines.classify_video_seed(_pipelinewrapper.ModelType.WAN, None),
            'wan-txt')
        parsed = _mediainput.parse_image_seed_uri('examples/media/earth.jpg')
        self.assertEqual(
            _videopipelines.classify_video_seed(_pipelinewrapper.ModelType.WAN, parsed),
            'wan-image')
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;last-frame=examples/media/beach.jpg')
        self.assertEqual(
            _videopipelines.classify_video_seed(_pipelinewrapper.ModelType.WAN, parsed),
            'wan-flf')
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;control=examples/media/rickroll-roll.gif')
        self.assertEqual(
            _videopipelines.classify_video_seed(_pipelinewrapper.ModelType.WAN, parsed),
            'wan-vace')
        parsed = _mediainput.parse_image_seed_uri('examples/media/rickroll-roll.gif')
        self.assertEqual(
            _videopipelines.classify_video_seed(_pipelinewrapper.ModelType.WAN, parsed),
            'wan-image')

    def test_classify_wan_animate_seed(self):
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;wan-pose=examples/media/rickroll-roll.gif;'
            'wan-face=examples/media/rickroll-roll.gif')
        self.assertEqual(
            _videopipelines.classify_video_seed(
                _pipelinewrapper.ModelType.WAN_ANIMATE, parsed),
            'wan-animate')
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif')
        self.assertEqual(
            _videopipelines.classify_video_seed(
                _pipelinewrapper.ModelType.WAN_ANIMATE, parsed),
            'wan-animate')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(
                _pipelinewrapper.ModelType.WAN_ANIMATE,
                _mediainput.parse_image_seed_uri('examples/media/earth.jpg'))
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(
                _pipelinewrapper.ModelType.WAN,
                _mediainput.parse_image_seed_uri(
                    'examples/media/earth.jpg;wan-pose=examples/media/rickroll-roll.gif'))

    def test_wan_animate_2_requires_driving(self):
        missing = _config(
            model_path='org/wan-animate-2',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE_2,
            image_seeds=['examples/media/earth.jpg'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            missing.check()
        self.assertIn('wan-driving=', str(raised.exception))

        driving = _config(
            model_path='org/wan-animate-2',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE_2,
            image_seeds=['examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif'])
        driving.check()
        self.assertEqual(driving.video_fps, [24.0])

        pose = _config(
            model_path='org/wan-animate-2',
            model_type=_pipelinewrapper.ModelType.WAN_ANIMATE_2,
            image_seeds=[
                'examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif;'
                'wan-pose=examples/media/rickroll-roll.gif'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            pose.check()
        self.assertIn('wan-pose=', str(raised.exception))

    def test_classify_wan_animate_2_seed(self):
        parsed = _mediainput.parse_image_seed_uri(
            'examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif')
        self.assertEqual(
            _videopipelines.classify_video_seed(
                _pipelinewrapper.ModelType.WAN_ANIMATE_2, parsed),
            'wan-animate-2')

    def test_wan_animate_2_args_reconstruct(self):
        import shlex

        import dgenerate.arguments as _arguments
        import dgenerate.pipelinewrapper.argreconstruct as _argreconstruct

        seed = (
            'examples/media/earth.jpg;wan-driving=examples/media/rickroll-roll.gif;'
            'frame-start=0;frame-end=2')
        argv = [
            'Wan-AI/Wan2.2-Animate-2-14B-Diffusers',
            '--model-type', 'wan-animate-2',
            '--dtype', 'bfloat16',
            '--model-sequential-offload',
            '--inference-steps', '40',
            '--video-fps', '24',
            '--vae',
            'AutoencoderKLWan;model=Wan-AI/Wan2.2-Animate-2-14B-Diffusers;subfolder=vae;dtype=float32',
            '--wan-segment-frame-lengths', '81',
            '--wan-prev-segment-frames', '1',
            '--max-sequence-length', '512',
            '--wan-driving-image-processors', 'flip',
            '--output-size', '640x800',
            '--animation-format', 'mp4',
            '--seeds', '1234',
            '--image-seeds', seed,
            '--prompts', 'A person dances in place.',
        ]
        config = _arguments.parse_args(argv, throw=True, log_error=False)
        generated = next(config.iterate_diffusion_args())
        wrapper = _pipelinewrapper.DiffusionPipelineWrapper(
            model_path=config.model_path,
            model_type=config.model_type,
            dtype=config.dtype,
            device='cpu',
            model_sequential_offload=config.model_sequential_offload,
            vae_uri=config.vae_uri)
        command = _argreconstruct.gen_dgenerate_command(
            wrapper,
            generated,
            extra_opts=[
                ('--animation-format', config.animation_format),
                ('--image-seeds', config.image_seeds[0]),
                ('--wan-driving-image-processors', config.wan_driving_image_processors),
            ],
            omit_device=True)
        again = _arguments.parse_args(
            shlex.split(command)[1:], throw=True, log_error=False)
        again_args = next(again.iterate_diffusion_args())
        self.assertEqual(again.model_path, config.model_path)
        self.assertEqual(again.model_type, config.model_type)
        self.assertEqual(again.dtype, config.dtype)
        self.assertTrue(again.model_sequential_offload)
        self.assertEqual(again.video_fps, config.video_fps)
        self.assertEqual(again.inference_steps, config.inference_steps)
        self.assertEqual(again.vae_uri, config.vae_uri)
        self.assertEqual(again.wan_segment_frame_lengths, [81])
        self.assertEqual(again.wan_prev_segment_frames, [1])
        self.assertEqual(again.max_sequence_length, 512)
        self.assertEqual(again.image_seeds, config.image_seeds)
        self.assertEqual(again.wan_driving_image_processors, ['flip'])
        self.assertEqual(again.animation_format, 'mp4')
        self.assertEqual(again.seeds, [1234])
        self.assertEqual(again_args.width, 640)
        self.assertEqual(again_args.height, 800)
        self.assertEqual(again_args.wan_segment_frame_length, 81)
        self.assertEqual(again_args.video_fps, 24.0)
        self.assertIn('--model-type wan-animate-2', command)
        self.assertIn('wan-driving=', command)

    def test_call_wan_animate_2_kwargs(self):
        captured = {}

        class Pipe:
            image_encoder = object()

            def __call__(self, **kwargs):
                captured.update(kwargs)
                return [PIL.Image.new('RGB', (4, 4))]

        class Held:
            pipeline = Pipe()
            family = 'wan-animate-2-distilled'

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('a dancer')
        args.inference_steps = 30
        args.video_fps = 24
        args.width = 640
        args.height = 800
        args.images = [PIL.Image.new('RGB', (8, 8))]
        args.wan_driving_video_frames = [PIL.Image.new('RGB', (8, 8)) for _ in range(81)]
        args.wan_driving_video_fps = 16

        class Wrapper:
            device = 'cpu'
            model_type = _pipelinewrapper.ModelType.WAN_ANIMATE_2
            model_cpu_offload = False
            model_sequential_offload = False
            model_group_offload = False

        pipe = Pipe()
        with unittest.mock.patch.object(
                _videopipelines, '_video_pipeline',
                return_value=(pipe, Held())):
            frames, audio, rate, fps = _videopipelines.wan._call_wan_animate_2(
                Wrapper(), args)

        self.assertEqual(captured['num_inference_steps'], 10)
        self.assertEqual(captured['output'], 'videos')
        self.assertEqual(captured['output_type'], 'pil')
        self.assertEqual(captured['driving_video_fps'], 16)
        self.assertEqual(captured['segment_frame_length'], 81)
        self.assertEqual(captured['width'], 640)
        self.assertEqual(fps, 24)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(captured['driving_video']), 81)
        self.assertIsNone(audio)
        self.assertIsNone(rate)

    def test_wan_animate_2_segment_fits_resampled_clip(self):
        fit = _videopipelines.wan._fit_wan_animate_2_segment
        # 49 frames at 30 fps resampled to 8 fps is 13 frames. An 81-frame
        # segment cannot be padded from that; 25 can.
        seen = _videopipelines.wan._animate2_resampled_count(49, 30, 8)
        self.assertEqual(seen, 13)
        self.assertEqual(fit(seen, 81, 1), 25)
        self.assertEqual(fit(49, 81, 1), 81)
        self.assertEqual(fit(13, 25, 1), 25)

    def test_call_wan_animate_2_shortens_segment(self):
        captured = {}
        warnings = []

        class Pipe:
            image_encoder = object()

            def __call__(self, **kwargs):
                captured.update(kwargs)
                return [PIL.Image.new('RGB', (4, 4))]

        class Held:
            pipeline = Pipe()
            family = 'wan-animate-2'

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('a kitten')
        args.inference_steps = 10
        args.video_fps = 8
        args.width = 640
        args.height = 800
        args.images = [PIL.Image.new('RGB', (8, 8))]
        args.wan_driving_video_frames = [
            PIL.Image.new('RGB', (8, 8)) for _ in range(49)]
        args.wan_driving_video_fps = 30
        args.wan_segment_frame_length = 81
        args.wan_prev_segment_frames = 1

        class Wrapper:
            device = 'cpu'
            model_type = _pipelinewrapper.ModelType.WAN_ANIMATE_2
            model_cpu_offload = False
            model_sequential_offload = False
            model_group_offload = False

        pipe = Pipe()
        with unittest.mock.patch.object(
                _videopipelines, '_video_pipeline',
                return_value=(pipe, Held())), \
                unittest.mock.patch.object(
                    _videopipelines.wan._messages, 'warning',
                    side_effect=warnings.append):
            _videopipelines.wan._call_wan_animate_2(Wrapper(), args)

        self.assertEqual(captured['fps'], 8)
        self.assertEqual(captured['segment_frame_length'], 25)
        self.assertEqual(len(captured['driving_video']), 49)
        self.assertEqual(captured['driving_video_fps'], 30)
        self.assertTrue(warnings)
        self.assertIn('reduced to 25', warnings[0])

    def test_call_wan_kwargs(self):
        class Scheduler:
            def __init__(self):
                self.config = {}

        class Pipe:
            def __init__(self):
                self.scheduler = Scheduler()
                self.vae_scale_factor_temporal = 4
                self.vae_scale_factor_spatial = 8
                self.config = {}

            def register_to_config(self, **kwargs):
                self.config.update(kwargs)

        pipe = Pipe()
        captured = {}

        def invoke(wrapper, pipeline, kwargs):
            captured['kwargs'] = kwargs

            class Output:
                frames = [PIL.Image.new('RGB', (4, 4))]

            return Output()

        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('a fox runs')
        args.inference_steps = 30
        args.guidance_scale = 5
        args.video_fps = 16
        args.video_length = 2
        args.width = 832
        args.height = 480
        args.wan_low_noise_guidance_scale = 3.5
        args.wan_boundary_ratio = 0.9

        class Held:
            pipeline = pipe
            family = 'wan-t2v'

        class Wrapper:
            device = 'cpu'
            model_type = _pipelinewrapper.ModelType.WAN
            model_cpu_offload = False
            model_sequential_offload = False
            model_path = 'org/wan'
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
                    side_effect=lambda pipeline, mode, family: captured.__setitem__('mode', mode) or pipeline), \
                unittest.mock.patch.object(
                    _videopipelines._schedulers, 'load_scheduler'), \
                unittest.mock.patch.object(
                    _videopipelines, '_invoke', side_effect=invoke):
            frames, audio, rate, fps = _videopipelines._call_wan(Wrapper(), args)

        self.assertEqual(captured['mode'], 'wan-txt')
        self.assertEqual(captured['kwargs']['num_frames'], 33)
        self.assertEqual(captured['kwargs']['guidance_scale_2'], 3.5)
        self.assertEqual(pipe.config['boundary_ratio'], 0.9)
        self.assertEqual(fps, 16)
        self.assertEqual(len(frames), 1)
        self.assertIsNone(audio)

    def test_wan_vace_fits_control_clip_to_output_size(self):
        class Scheduler:
            def __init__(self):
                self.config = {}

        class Transformer:
            def __init__(self):
                self.config = unittest.mock.Mock(patch_size=(1, 2, 2))

        class Pipe:
            def __init__(self):
                self.scheduler = Scheduler()
                self.transformer = Transformer()
                self.vae_scale_factor_temporal = 4
                self.vae_scale_factor_spatial = 8
                self.config = {}

            def register_to_config(self, **kwargs):
                self.config.update(kwargs)

        pipe = Pipe()
        captured = {}

        def invoke(wrapper, pipeline, kwargs):
            captured['kwargs'] = kwargs

            class Output:
                frames = [PIL.Image.new('RGB', (4, 4))]

            return Output()

        clip = PIL.Image.new('RGB', (720, 480), (20, 40, 60))
        mask = PIL.Image.new('L', (720, 480), 255)
        args = _pipelinewrapper.DiffusionArguments()
        args.prompt = _prompt.Prompt('a traveler walks the trail')
        args.inference_steps = 4
        args.guidance_scale = 5
        args.video_fps = 8
        args.video_length = 6
        args.width = 832
        args.height = 480
        reference = PIL.Image.new('RGB', (400, 600))
        args.aspect_correct = True
        args.vace_video_frames = [clip]
        args.vace_mask_frames = [mask]
        args.vace_reference_images = [reference]

        class Held:
            pipeline = pipe
            family = 'wan-vace'

        class Wrapper:
            device = 'cpu'
            model_type = _pipelinewrapper.ModelType.WAN
            model_cpu_offload = False
            model_sequential_offload = False
            model_path = 'org/wan-vace'
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
                    side_effect=lambda pipeline, mode, family: captured.__setitem__('mode', mode) or pipeline), \
                unittest.mock.patch.object(
                    _videopipelines._schedulers, 'load_scheduler'), \
                unittest.mock.patch.object(
                    _videopipelines, '_invoke', side_effect=invoke):
            _videopipelines._call_wan(Wrapper(), args)

        self.assertEqual(captured['mode'], 'wan-vace')
        self.assertEqual(captured['kwargs']['width'], 832)
        self.assertEqual(captured['kwargs']['height'], 544)
        self.assertEqual(captured['kwargs']['video'][0].size, (832, 544))
        self.assertEqual(captured['kwargs']['mask'][0].size, (832, 544))
        self.assertEqual(captured['kwargs']['reference_images'][0].size, (400, 600))
        self.assertEqual(args.width, 832)
        self.assertEqual(args.height, 544)

        stretched = _pipelinewrapper.DiffusionArguments()
        stretched.aspect_correct = False
        stretched.vace_video_frames = [PIL.Image.new('RGB', (720, 480))]
        stretched.vace_mask_frames = [PIL.Image.new('L', (720, 480), 0)]
        fitted = _videopipelines.wan._fit_vace_canvas(stretched, 832, 480, 16)
        self.assertEqual(fitted, (832, 480))
        self.assertEqual(stretched.vace_video_frames[0].size, (832, 480))
        self.assertEqual(stretched.vace_mask_frames[0].size, (832, 480))

        pose = _pipelinewrapper.DiffusionArguments()
        pose.aspect_correct = True
        pose.wan_pose_video_frames = [PIL.Image.new('RGB', (720, 480))]
        pose.images = [PIL.Image.new('RGB', (400, 600))]
        pose.wan_face_video_frames = [PIL.Image.new('RGB', (128, 128))]
        pose.mask_video_frames = [PIL.Image.new('L', (720, 480), 255)]
        canvas = _videopipelines.wan._fit_output_media(
            pose, 'wan_pose_video_frames', [('mask_video_frames', True)],
            832, 480, 16, 'Wan-Animate pose clip')
        self.assertEqual(canvas, (832, 544))
        self.assertEqual(pose.wan_pose_video_frames[0].size, (832, 544))
        self.assertEqual(pose.mask_video_frames[0].size, (832, 544))
        self.assertEqual(pose.images[0].size, (400, 600))
        self.assertEqual(pose.wan_face_video_frames[0].size, (128, 128))

    def test_length_product_uses_shared_video_fields(self):
        config = _config(
            model_path='org/wan',
            model_type=_pipelinewrapper.ModelType.WAN,
            prompts=[_prompt.Prompt(), _prompt.Prompt()],
            video_lengths=[5.0, 2.0],
            video_fps=[16.0, 24.0])
        self.assertEqual(config.calculate_generation_steps(), 8)


if __name__ == '__main__':
    unittest.main()
