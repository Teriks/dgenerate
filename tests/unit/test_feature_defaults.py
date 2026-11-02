import unittest

import dgenerate.extras.regione.runtime as _regione
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.arguments as _arguments
import dgenerate.pipelinewrapper.featuredefaults as _featuredefaults
import dgenerate.prompt as _prompt
import dgenerate.renderloopconfig as _renderloopconfig


class TestFeatureDefaultsInStepText(unittest.TestCase):
    def _args(self):
        args = _arguments.DiffusionArguments()
        args.prompt = _prompt.Prompt()
        args.inference_steps = 20
        return args

    def test_sada_defaults_are_shown_and_the_arguments_stay_unset(self):
        args = self._args()
        args.sada = True
        args.sada_max_interval = 8
        text = args.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.FLUX)
        self.assertIn('SADA Max Downsample: 0', text)
        self.assertIn('SADA SX: 0', text)
        self.assertIn('SADA Max Interval: 8', text)
        self.assertIsNone(args.sada_sx)
        self.assertIsNone(args.sada_max_downsample)

    def test_omitted_features_stay_out_of_the_step_text(self):
        text = self._args().describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.FLUX)
        self.assertNotIn('SADA', text)
        self.assertNotIn('DyPE', text)
        self.assertNotIn('PAG Scale', text)

    def test_dype_and_pag_defaults_are_shown(self):
        args = self._args()
        args.dype = True
        args.pag_applied_layers = ['blocks.1']
        text = args.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.SD3)
        self.assertIn('DyPE Position Encoding Method: yarn', text)
        self.assertIn('DyPE YARN Magnitude Scale: constant', text)
        self.assertIn('DyPE Axis Scaling: shared', text)
        self.assertIn('PAG Scale: 3.0', text)
        self.assertIn('PAG Adaptive Scale: 0.0', text)
        self.assertIsNone(args.dype_method)
        self.assertIsNone(args.pag_scale)

    def test_qwen_regione_uses_its_own_defaults(self):
        args = self._args()
        args.regione = True
        text = args.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT)
        self.assertIn('RegionE Threshold: 0.8', text)
        self.assertIn('RegionE Cache Threshold: 0.03', text)

    def test_fill_defaults_on_render_loop_config(self):
        config = _renderloopconfig.RenderLoopConfig()
        config.model_type = _pipelinewrapper.ModelType.SD
        config.sada = True
        config.fill_defaults()
        self.assertIsNone(config.guidance_scales)
        self.assertIsNone(config.inference_steps)
        self.assertEqual(config.sada_sxs, [3])

        untouched = _renderloopconfig.RenderLoopConfig()
        untouched.model_type = _pipelinewrapper.ModelType.SD
        untouched.guidance_scales = [9.0]
        untouched.fill_defaults()
        self.assertEqual(untouched.guidance_scales, [9.0])

    def test_fill_defaults_regione_steps(self):
        config = _renderloopconfig.RenderLoopConfig()
        config.model_type = _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT
        config.regione = True
        config.fill_defaults()
        self.assertEqual(config.inference_steps, [28])
        self.assertEqual(config.regione_thresholds, [0.8])

    def test_unspecified_steps_follow_the_pipeline(self):
        import dgenerate.pipelinewrapper.wrapper as _wrapper

        args = self._args()
        args.inference_steps = None
        args.guidance_scale = None
        _wrapper._resolve_omitted_sampling(
            args, _pipelinewrapper.ModelType.SD, None)
        self.assertEqual(args.inference_steps, 50)
        self.assertEqual(args.guidance_scale, 7.5)

        regione = self._args()
        regione.inference_steps = None
        regione.regione = True
        regione.inference_steps = _regione.resolve_regione_inference_step_count(
            regione.inference_steps)
        self.assertEqual(regione.inference_steps, 28)

        omitted = self._args()
        omitted.inference_steps = None
        omitted.guidance_scale = None
        text = omitted.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.SD)
        self.assertIn('Inference Steps: auto', text)
        self.assertIn('Guidance Scale: auto', text)
        self.assertIsNone(omitted.inference_steps)
        self.assertIsNone(omitted.guidance_scale)

        class _Seed:
            def __init__(self, images, masks=None, single=False):
                self.images = images
                self.mask_images = masks
                self.is_single_spec = single

        img2img = self._args()
        img2img.image_seed_strength = None
        text = img2img.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.SD, image_seed=_Seed(['photo.png']))
        self.assertIn('Image Seed Strength: 0.8', text)
        self.assertIsNone(img2img.image_seed_strength)

        control = self._args()
        text = control.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.SD,
            image_seed=_Seed(['control.png'], single=True),
            control_guidance=True)
        self.assertNotIn('Image Seed Strength', text)
        args = self._args()
        text = args.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.WAN)
        self.assertIn('Video FPS: 16.0', text)
        self.assertIsNone(args.video_fps)

        text = args.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.UPSCALER_X4)
        self.assertIn('Upscaler Noise Level: 20', text)

        text = args.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.SDXL, refiner=True)
        self.assertIn('SDXL High Noise Fraction: 0.8', text)
        text = args.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.SDXL)
        self.assertNotIn('SDXL High Noise Fraction', text)

    def test_floyd_noise_and_wan_video_strength_are_shown(self):
        class _Seed:
            def __init__(self, images, masks=None):
                self.images = images
                self.mask_images = masks

        ifs = self._args()
        text = ifs.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.IFS, image_seed=_Seed(['low.png']))
        self.assertIn('Upscaler Noise Level: 250', text)
        self.assertIsNone(ifs.upscaler_noise_level)

        inpaint = self._args()
        text = inpaint.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.IFS,
            image_seed=_Seed(['low.png'], ['mask.png']))
        self.assertIn('Upscaler Noise Level: 0', text)
        self.assertIsNone(inpaint.upscaler_noise_level)

        img2img = self._args()
        text = img2img.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.IFS_IMG2IMG)
        self.assertIn('Upscaler Noise Level: 250', text)

        kept = self._args()
        kept.upscaler_noise_level = 10
        text = kept.describe_pipeline_wrapper_args(
            _pipelinewrapper.ModelType.IFS, image_seed=_Seed(['low.png']))
        self.assertIn('Upscaler Noise Level: 10', text)

        wan = self._args()
        wan.video_frames = ['frame-a', 'frame-b']
        text = wan.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.WAN)
        self.assertIn('Image Seed Strength: 0.8', text)
        self.assertIsNone(wan.image_seed_strength)

        explicit = self._args()
        explicit.video_frames = ['frame-a', 'frame-b']
        explicit.image_seed_strength = 0.4
        text = explicit.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.WAN)
        self.assertIn('Image Seed Strength: 0.4', text)

        still = self._args()
        still.images = ['still.png']
        text = still.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.WAN)
        self.assertNotIn('Image Seed Strength', text)

    def test_ras_end_stays_unset_until_steps_are_known(self):
        omitted = self._args()
        omitted.inference_steps = None
        omitted.ras = True
        _featuredefaults.apply_known_defaults(
            omitted, _pipelinewrapper.ModelType.SD3)
        self.assertIsNone(omitted.ras_end_step)
        self.assertEqual(omitted.ras_start_step, 4)
        text = omitted.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.SD3)
        self.assertNotIn('RAS End Step:', text)
        self.assertIn('Inference Steps: auto', text)

        known = self._args()
        known.inference_steps = 28
        known.ras = True
        text = known.describe_pipeline_wrapper_args(_pipelinewrapper.ModelType.SD3)
        self.assertIn('RAS End Step: 28', text)
        self.assertIsNone(known.ras_end_step)

        config = _renderloopconfig.RenderLoopConfig()
        config.model_type = _pipelinewrapper.ModelType.SD3
        config.ras = True
        config.fill_defaults()
        self.assertIsNone(config.ras_end_steps)
        self.assertIsNone(config.inference_steps)

        explicit = _renderloopconfig.RenderLoopConfig()
        explicit.model_type = _pipelinewrapper.ModelType.SD3
        explicit.ras = True
        explicit.inference_steps = [40]
        explicit.fill_defaults()
        self.assertEqual(explicit.ras_end_steps, [40])

    def test_ras_error_reset_steps_are_one_pattern(self):
        config = _renderloopconfig.RenderLoopConfig()
        config.model_path = 'org/sd3'
        config.model_type = _pipelinewrapper.ModelType.SD3
        config.prompts = [_prompt.Prompt('a cat')]
        config.ras = True
        config.check()
        config.fill_defaults()
        self.assertEqual(config.ras_error_reset_steps, [[12, 22]])
        self.assertEqual(config.calculate_generation_steps(), 1)
        step = next(config.iterate_diffusion_args())
        self.assertEqual(step.ras_error_reset_steps, [12, 22])

    def test_feature_defaults_do_not_split_into_extra_generations(self):
        cases = (
            (_pipelinewrapper.ModelType.SD, {'sada': True}),
            (_pipelinewrapper.ModelType.SD, {'pag': True}),
            (_pipelinewrapper.ModelType.FLUX, {'dype': True}),
            (_pipelinewrapper.ModelType.FLUX, {'tea_cache': True}),
            (_pipelinewrapper.ModelType.FLUX, {'spectrum': True}),
            (_pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, {'regione': True}),
            (_pipelinewrapper.ModelType.SD3, {'ras': True}),
            (_pipelinewrapper.ModelType.SD, {'deep_cache': True}),
            (_pipelinewrapper.ModelType.SD, {'inpaint_crop': True}),
            (_pipelinewrapper.ModelType.PIX2PIX, {}),
            (_pipelinewrapper.ModelType.S_CASCADE, {}),
            (_pipelinewrapper.ModelType.SDXL, {'sdxl_refiner_uri': 'refiner'}),
        )
        for model_type, extra in cases:
            config = _renderloopconfig.RenderLoopConfig()
            config.model_path = 'org/model'
            config.model_type = model_type
            config.prompts = [_prompt.Prompt('a cat')]
            for name, value in extra.items():
                setattr(config, name, value)
            config.fill_defaults()
            self.assertEqual(
                config.calculate_generation_steps(), 1, (model_type, extra))

        sada = _renderloopconfig.RenderLoopConfig()
        sada.model_type = _pipelinewrapper.ModelType.SD
        sada.sada = True
        sada.fill_defaults()
        self.assertEqual(sada.sada_acc_ranges, [(10, 47)])


if __name__ == '__main__':
    unittest.main()
