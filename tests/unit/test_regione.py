import os
import tempfile
import unittest

import dgenerate.extras.regione as _regione
import dgenerate.extras.regione.runtime as _regione_runtime
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.renderloopconfig as _renderloopconfig

# 1x1 PNG. RenderLoopConfig.check parses image seeds and requires a real file.
_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d4948445200000001000000010802000000907753'
    'de0000000c4944415408d763f8cfc000000301010018dd8db40000000049454e'
    '44ae426082')


def _config(**values):
    config = _renderloopconfig.RenderLoopConfig()
    for name, value in values.items():
        setattr(config, name, value)
    return config


class TestRegionE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        handle.write(_PNG)
        handle.close()
        cls.image_path = handle.name

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.image_path)

    def test_model_type_support(self):
        self.assertTrue(_pipelinewrapper.model_type_supports_regione('flux-kontext'))
        self.assertTrue(_pipelinewrapper.model_type_supports_regione('qwen-image-edit'))
        self.assertFalse(_pipelinewrapper.model_type_supports_regione('flux'))

    def test_pipeline_class_support(self):
        self.assertTrue(_regione.pipeline_class_supports_regione('FluxKontextPipeline'))
        self.assertTrue(_regione.pipeline_class_supports_regione('QwenImageEditPlusPipeline'))
        self.assertFalse(_regione.pipeline_class_supports_regione('FluxKontextInpaintPipeline'))

    def test_build_settings_uses_pipeline_defaults(self):
        settings = _regione.build_settings(
            'FluxKontextPipeline',
            warmup_step=None,
            post_step=None,
            refresh_step=None,
            threshold=None,
            cache_threshold=None,
            erosion_dilation=None)
        self.assertEqual(settings.threshold, 0.93)

    def test_config_requires_image_and_steps(self):
        config = _config(
            model_path='black-forest-labs/FLUX.1-Kontext-dev',
            model_type=_pipelinewrapper.ModelType.FLUX_KONTEXT,
            regione=True,
            inference_steps=[28],
            prompts=['edit the photo'],
        )
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

        config.image_seeds = [self.image_path]
        config.check()

    def test_config_rejects_tea_cache(self):
        config = _config(
            model_path='black-forest-labs/FLUX.1-Kontext-dev',
            model_type=_pipelinewrapper.ModelType.FLUX_KONTEXT,
            regione=True,
            tea_cache=True,
            inference_steps=[28],
            image_seeds=[self.image_path],
            prompts=['edit'],
        )
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config.check()

    def test_implied_regione_from_thresholds(self):
        config = _config(
            model_path='black-forest-labs/FLUX.1-Kontext-dev',
            model_type=_pipelinewrapper.ModelType.FLUX_KONTEXT,
            regione_thresholds=[0.9],
            inference_steps=[28],
            image_seeds=[self.image_path],
            prompts=['edit'],
        )
        config.check()

    def test_omitted_inference_steps_default_to_28(self):
        config = _config(
            model_path='Qwen/Qwen-Image-Edit',
            model_type=_pipelinewrapper.ModelType.QWEN_IMAGE_EDIT,
            regione=True,
            image_seeds=[self.image_path],
            prompts=['edit'],
        )
        config.check()
        self.assertEqual(config.inference_steps, [28])

    def test_default_inference_steps_coerced_to_28(self):
        default = _pipelinewrapper.constants.DEFAULT_INFERENCE_STEPS
        self.assertEqual(_regione.resolve_regione_inference_steps_list(None), [28])
        self.assertEqual(
            _regione.resolve_regione_inference_steps_list([default]), [28])
        self.assertEqual(
            _regione.resolve_regione_inference_step_count(default), 28)
        self.assertEqual(
            _regione.resolve_regione_inference_step_count(None), 28)

    def test_schedule_rejects_refresh_inside_warmup(self):
        with self.assertRaises(ValueError):
            _regione.validate_settings(_regione.RegionESettings(
                warmup_step=20,
                post_step=2,
                refresh_step=16,
                threshold=0.93,
                cache_threshold=0.04,
                erosion_dilation=True))

    def test_helper_does_not_mutate_shared_defaults(self):
        from dgenerate.extras.regione.upstream.tool.RegionE import RegionEHelper, config

        class FluxKontextPipeline:
            pass

        original = config['FluxKontextPipeline']['threshold']
        helper = RegionEHelper(FluxKontextPipeline())
        helper.set_params(threshold=0.11)
        self.assertEqual(config['FluxKontextPipeline']['threshold'], original)
        again = RegionEHelper(FluxKontextPipeline())
        self.assertEqual(again.config['threshold'], original)

    def test_restore_puts_back_pipeline_objects(self):
        class _Attn:
            def __init__(self):
                self.processor = 'original'

            def set_processor(self, processor):
                self.processor = processor

        class _Block:
            def __init__(self):
                self.attn = _Attn()

        class _Transformer:
            def __init__(self):
                self.transformer_blocks = [_Block()]
                self.single_transformer_blocks = []

            def forward(self):
                return 'original'

        class _Pipeline:
            def __init__(self):
                self.transformer = _Transformer()
                self.scheduler = 'original-scheduler'

        pipeline = _Pipeline()
        snapshot = _regione_runtime._snapshot_pipeline(pipeline)
        pipeline.__class__ = type('Replaced', (), {})
        pipeline.scheduler = 'replaced-scheduler'
        pipeline.transformer.forward = lambda: 'replaced'
        pipeline.transformer.transformer_blocks[0].attn.set_processor('replaced')
        _regione_runtime._restore_pipeline(pipeline, snapshot)

        self.assertIs(pipeline.__class__, _Pipeline)
        self.assertEqual(pipeline.scheduler, 'original-scheduler')
        self.assertEqual(pipeline.transformer.forward(), 'original')
        self.assertEqual(pipeline.transformer.transformer_blocks[0].attn.processor, 'original')


if __name__ == '__main__':
    unittest.main()
