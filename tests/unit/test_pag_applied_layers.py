import unittest

import dgenerate.arguments as _arguments
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.wrapper as _wrapper
import dgenerate.renderloopconfig as _renderloopconfig


class _FakePagPipeline:
    def __init__(self):
        self._pag_attn_processors = ('cfg-joint', 'uncond-joint')
        self.pag_applied_layers = ['blocks.1']
        self.processors = None

    def set_pag_applied_layers(self, layers, pag_attn_processors=None):
        self.pag_applied_layers = list(layers)
        self.processors = pag_attn_processors


class TestPagAppliedLayers(unittest.TestCase):
    def test_csv_layer_sets_are_tried_in_turn(self):
        parsed = _arguments._type_pag_applied_layers('mid, blocks.13')
        self.assertEqual(parsed, ['mid', 'blocks.13'])

        config = _renderloopconfig.RenderLoopConfig()
        config.model_path = 'stabilityai/stable-diffusion-3-medium-diffusers'
        config.model_type = _pipelinewrapper.ModelType.SD3
        config.pag_applied_layers = [
            _arguments._type_pag_applied_layers('blocks.13'),
            _arguments._type_pag_applied_layers('mid,blocks.1'),
        ]
        config.check()

        self.assertFalse(config.pag)
        self.assertTrue(config.is_pag_enabled())
        self.assertIsNone(config.pag_scales)
        self.assertIsNone(config.pag_adaptive_scales)

        generated = list(config.iterate_diffusion_args())
        self.assertEqual(
            [step.pag_applied_layers for step in generated],
            [['blocks.13'], ['mid', 'blocks.1']])
        self.assertTrue(all(step.pag_scale is None for step in generated))
        self.assertEqual(config.calculate_generation_steps(), len(generated))

    def test_implied_options_keep_generation_count_aligned(self):
        def assert_aligned(config):
            counted = config.calculate_generation_steps()
            produced = len(list(config.iterate_diffusion_args()))
            self.assertEqual(counted, produced)
            return counted

        layers = _renderloopconfig.RenderLoopConfig()
        layers.model_path = 'stabilityai/stable-diffusion-3-medium-diffusers'
        layers.model_type = _pipelinewrapper.ModelType.SD3
        layers.pag_applied_layers = [['blocks.13'], ['mid', 'blocks.1']]
        layers.pag_scales = [1.0, 2.0]
        layers.check()
        self.assertEqual(assert_aligned(layers), 4)
        self.assertFalse(layers.pag)

        crop = _renderloopconfig.RenderLoopConfig()
        crop.model_path = 'runwayml/stable-diffusion-v1-5'
        crop.model_type = _pipelinewrapper.ModelType.SD
        crop.inpaint_crop_paddings = [32, 64]
        self.assertEqual(assert_aligned(crop), 2)
        self.assertFalse(crop.inpaint_crop)
        self.assertTrue(crop.is_inpaint_crop_enabled())

    def test_feature_queries_do_not_write_the_parent_flag(self):
        fresh = _renderloopconfig.RenderLoopConfig()
        for method in (
                'is_pipeline_prompt_enhance_enabled',
                'is_cache_dit_enabled',
                'is_dype_enabled',
                'is_tea_cache_enabled',
                'is_spectrum_enabled',
                'is_regione_enabled',
                'is_ras_enabled',
                'is_deep_cache_enabled',
                'is_sdxl_refiner_deep_cache_enabled',
                'is_sada_enabled',
                'is_pag_enabled',
                'is_sdxl_refiner_pag_enabled',
                'is_inpaint_crop_enabled',
        ):
            self.assertFalse(getattr(fresh, method)(), method)

        cases = (
            ('pipeline_prompt_enhance_top_p', 0.9,
             'is_pipeline_prompt_enhance_enabled', 'pipeline_prompt_enhance'),
            ('cache_dit_config', 'cache.yaml', 'is_cache_dit_enabled', 'cache_dit'),
            ('dype_mscales', ['decay'], 'is_dype_enabled', 'dype'),
            ('tea_cache_rel_l1_thresholds', [0.4], 'is_tea_cache_enabled', 'tea_cache'),
            ('spectrum_weights', [0.5], 'is_spectrum_enabled', 'spectrum'),
            ('regione_thresholds', [0.1], 'is_regione_enabled', 'regione'),
            ('ras_start_steps', [4], 'is_ras_enabled', 'ras'),
            ('deep_cache_intervals', [3], 'is_deep_cache_enabled', 'deep_cache'),
            ('sdxl_refiner_deep_cache_intervals', [3],
             'is_sdxl_refiner_deep_cache_enabled', 'sdxl_refiner_deep_cache'),
            ('sada_max_intervals', [4], 'is_sada_enabled', 'sada'),
            ('pag_scales', [1.0], 'is_pag_enabled', 'pag'),
            ('sdxl_refiner_pag_scales', [1.0],
             'is_sdxl_refiner_pag_enabled', 'sdxl_refiner_pag'),
            ('inpaint_crop_paddings', [32], 'is_inpaint_crop_enabled', 'inpaint_crop'),
        )
        for attr, value, method, flag in cases:
            config = _renderloopconfig.RenderLoopConfig()
            setattr(config, attr, value)
            self.assertTrue(getattr(config, method)(), method)
            self.assertFalse(getattr(config, flag), flag)

    def test_apply_keeps_the_pipeline_attention_processors(self):
        pipeline = _FakePagPipeline()
        helper = _wrapper.DiffusionPipelineWrapper.__new__(
            _wrapper.DiffusionPipelineWrapper)
        apply = helper._apply_pag_applied_layers
        apply(pipeline, ['blocks.13'])
        self.assertEqual(pipeline.pag_applied_layers, ['blocks.13'])
        self.assertEqual(pipeline.processors, ('cfg-joint', 'uncond-joint'))

        apply(pipeline, None)
        self.assertEqual(pipeline.pag_applied_layers, ['blocks.1'])
        self.assertEqual(pipeline.processors, ('cfg-joint', 'uncond-joint'))

    def test_flow_image_models_reject_pag_layers(self):
        config = _renderloopconfig.RenderLoopConfig()
        config.model_path = 'Tongyi-MAI/Z-Image-Turbo'
        config.model_type = _pipelinewrapper.ModelType.Z_IMAGE
        config.pag_applied_layers = [['blocks.0']]
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('pag', str(raised.exception))

    def test_bare_pag_switch_arrives_as_the_default_scales(self):
        config = _renderloopconfig.RenderLoopConfig()
        config.model_path = 'stabilityai/stable-diffusion-3-medium-diffusers'
        config.model_type = _pipelinewrapper.ModelType.SD3
        config.pag = True
        generated = list(config.iterate_diffusion_args())
        self.assertEqual(generated[0].pag_scale, 3.0)
        self.assertEqual(generated[0].pag_adaptive_scale, 0.0)
        self.assertTrue(config.pag)


if __name__ == '__main__':
    unittest.main()
