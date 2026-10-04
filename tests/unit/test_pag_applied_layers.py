import unittest

import dgenerate.arguments as _arguments
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.constants as _constants
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

        self.assertTrue(config.pag)
        self.assertEqual(config.pag_scales, [_constants.DEFAULT_PAG_SCALE])
        self.assertEqual(config.pag_adaptive_scales, [_constants.DEFAULT_PAG_ADAPTIVE_SCALE])

        generated = list(config.iterate_diffusion_args())
        self.assertEqual(
            [step.pag_applied_layers for step in generated],
            [['blocks.13'], ['mid', 'blocks.1']])
        self.assertTrue(all(step.pag_scale == _constants.DEFAULT_PAG_SCALE for step in generated))

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


if __name__ == '__main__':
    unittest.main()
