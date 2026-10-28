import unittest
import unittest.mock

import diffusers

import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.pipelines as _pipelines
from dgenerate.pipelinewrapper.enums import DataType, ModelType
from dgenerate.pipelinewrapper.uris.controllorauri import ControlLoRAUri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidControlLoRAUriError


class TestControlLoRA(unittest.TestCase):
    def test_parse_scale(self):
        parsed = ControlLoRAUri.parse('repo/lora;scale=0.5;weight-name=a.safetensors')
        self.assertEqual(parsed.model, 'repo/lora')
        self.assertEqual(parsed.scale, 0.5)
        self.assertEqual(parsed.weight_name, 'a.safetensors')

    def test_parse_rejects_bad_scale(self):
        with self.assertRaises(InvalidControlLoRAUriError):
            ControlLoRAUri.parse('repo/lora;scale=nope')

    def test_flux_pipeline_class(self):
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX,
                _pipelinewrapper.PipelineType.TXT2IMG,
                control_lora_uris=['repo/lora']),
            diffusers.FluxControlPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX,
                _pipelinewrapper.PipelineType.IMG2IMG,
                control_lora_uris=['repo/lora']),
            diffusers.FluxControlImg2ImgPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX,
                _pipelinewrapper.PipelineType.INPAINT,
                control_lora_uris=['repo/lora']),
            diffusers.FluxControlInpaintPipeline)

    def test_sd_pipeline_class(self):
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.SD,
                _pipelinewrapper.PipelineType.TXT2IMG,
                control_lora_uris=['repo/lora']),
            diffusers.StableDiffusionControlNetPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.SDXL,
                _pipelinewrapper.PipelineType.TXT2IMG,
                control_lora_uris=['repo/lora']),
            diffusers.StableDiffusionXLControlNetPipeline)

    def test_exclusive_with_controlnet(self):
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.SDXL,
                _pipelinewrapper.PipelineType.TXT2IMG,
                controlnet_uris=['repo/cn'],
                control_lora_uris=['repo/lora'])

    def test_rejected_on_other_model_types(self):
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.SD3,
                _pipelinewrapper.PipelineType.TXT2IMG,
                control_lora_uris=['repo/lora'])

    def test_cache_key_ignores_control_lora_unless_it_changes_the_component(self):
        common = dict(
            model_path='repo/model',
            lora_uris=['org/style;scale=1'],
            lora_fuse_scale=1.0,
            revision=None,
            variant=None,
            subfolder=None,
            dtype=DataType.FLOAT16,
            original_config=None,
        )
        plain = _pipelines._get_component_cache_key(
            'unet', model_type=ModelType.SDXL, **common)
        with_control = _pipelines._get_component_cache_key(
            'unet', model_type=ModelType.SDXL,
            control_lora_uris=['org/control'], control_lora_fuse_scale=1.0,
            **common)
        self.assertEqual(plain, with_control)

        controlnet = _pipelines._get_component_cache_key(
            'controlnet', model_type=ModelType.SDXL,
            control_lora_uris=['org/control'], control_lora_fuse_scale=0.5,
            **common)
        self.assertNotEqual(plain, controlnet)
        self.assertIn('org/control', controlnet)
        self.assertNotIn('org/style', controlnet)

        other_style = dict(common)
        other_style.update(lora_uris=['org/other;scale=1'], lora_fuse_scale=0.25)
        shared = _pipelines._get_component_cache_key(
            'controlnet', model_type=ModelType.SDXL,
            control_lora_uris=['org/control'], control_lora_fuse_scale=0.5,
            **other_style)
        self.assertEqual(controlnet, shared)

        replaced_unet = _pipelines._get_component_cache_key(
            'controlnet', model_type=ModelType.SDXL,
            control_lora_uris=['org/control'], control_lora_fuse_scale=0.5,
            manual_component_uris={'unet': 'org/unet'},
            **common)
        self.assertNotEqual(controlnet, replaced_unet)

        flux = _pipelines._get_component_cache_key(
            'transformer', model_type=ModelType.FLUX,
            control_lora_uris=['org/control'], control_lora_fuse_scale=1.0,
            **common)
        flux_plain = _pipelines._get_component_cache_key(
            'transformer', model_type=ModelType.FLUX, **common)
        self.assertNotEqual(flux, flux_plain)

    def test_flux_control_and_style_loras_load_as_one_set(self):
        calls = []

        def load(pipeline, uris, fuse_scale=1.0, **kwargs):
            calls.append(([(u.model, u.scale, u.component) for u in uris], fuse_scale))

        style = _pipelinewrapper.uris.LoRAUri.parse('org/style;scale=0.5;component=transformer')
        with unittest.mock.patch.object(_pipelinewrapper.uris.LoRAUri, 'load_on_pipeline', side_effect=load):
            _pipelines._load_flux_control_and_style_loras(
                object(), ['org/control;scale=0.8'], 0.5, [style], 2.0)

        self.assertEqual(calls, [(
            [('org/control', 0.4, None), ('org/style', 1.0, 'transformer')], 1.0)])

    def test_fused_norm_layers_survive_unload(self):
        transformer = type('Transformer', (), {})()
        transformer._transformer_norm_layers = {'norm_q.weight': object()}
        pipeline = type('Pipe', (), {})()
        pipeline.transformer = transformer
        _pipelines._keep_fused_norm_layers(pipeline)
        self.assertIsNone(transformer._transformer_norm_layers)

    def test_flux_control_embedder_config_follows_widened_weights(self):
        class Config:
            in_channels = 64
            out_channels = 64
            patch_size = 1

        class Linear:
            def __init__(self, features):
                self.in_features = features
                self.out_features = features

        module = type('Module', (), {})()
        module.config = Config()
        module.x_embedder = Linear(128)
        module.proj_out = Linear(128)
        _pipelines._apply_flux_control_embedder_config(module)
        self.assertEqual(module.config.in_channels, 128)
        self.assertEqual(module.config.out_channels, 128)


if __name__ == '__main__':
    unittest.main()
