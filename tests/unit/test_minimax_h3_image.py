import types
import unittest

import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.minimaxh3image as _image
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.prompt as _prompt


def _args(**values):
    args = types.SimpleNamespace(
        scheduler_uri=None, sigmas=None, timesteps=None, inference_steps=None,
        guidance_scale=None, batch_size=None, output_latents=False)
    for name, value in values.items():
        setattr(args, name, value)
    return args


class TestMiniMaxH3Image(unittest.TestCase):
    def test_model_type_registration(self):
        self.assertEqual(
            _enums.get_model_type_enum('minimax-h3-image'),
            _enums.ModelType.MINIMAX_H3_IMAGE)
        self.assertIn('minimax-h3-image', _enums.supported_model_type_strings())
        self.assertTrue(_enums.model_type_is_minimax_h3_image('minimax-h3-image'))
        self.assertTrue(_enums.model_type_is_flow_image_family('minimax-h3-image'))
        self.assertTrue(_enums.model_type_image_is_condition('minimax-h3-image'))
        self.assertFalse(_enums.model_type_is_video_family('minimax-h3-image'))
        self.assertIn('minimax-h3-image', _pipelinewrapper.CACHE_DIT_MODEL_TYPES)
        schema = __import__(
            'dgenerate.pipelinewrapper.textencoderschema', fromlist=['text_encoder_schema']
        ).text_encoder_schema()
        self.assertEqual(schema['minimax-h3-image'][0]['name'], 'text_encoder')
        self.assertIn(
            'Qwen3VLForConditionalGeneration',
            schema['minimax-h3-image'][0]['encoders'])

    def test_prepare_call_keeps_the_distilled_schedule(self):
        pipeline_args = {
            'guidance_scale': 1.0,
            'num_inference_steps': 6,
            'num_images_per_prompt': 1,
        }
        user_args = _args(inference_steps=6, guidance_scale=1.0, batch_size=1)
        _image.prepare_call(pipeline_args, user_args, _prompt.Prompt('a fox'))
        self.assertEqual(pipeline_args['prompt'], 'a fox')
        self.assertNotIn('guidance_scale', pipeline_args)
        self.assertNotIn('num_inference_steps', pipeline_args)
        self.assertNotIn('num_images_per_prompt', pipeline_args)
        self.assertEqual(pipeline_args['output'], 'images')

    def test_images_come_out_of_a_pipeline_state(self):
        import dgenerate.pipelinewrapper.flux3 as _flux3

        image = object()
        state = type('PipelineState', (), {'values': {'images': [image]}})()
        self.assertEqual(_flux3.images_from_call(state), [image])
        self.assertEqual(_flux3.images_from_call([image]), [image])

    def test_prepare_call_rejects_a_different_schedule(self):
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _image.prepare_call(
                {}, _args(inference_steps=20), _prompt.Prompt('a fox'))
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _image.prepare_call(
                {}, _args(guidance_scale=3.5), _prompt.Prompt('a fox'))
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _image.prepare_call({}, _args(), _prompt.Prompt())

    def test_comfy_image_output_heads_split(self):
        import torch
        import dgenerate.pipelinewrapper.minimaxh3 as _minimax

        weight = torch.arange(576 * 4, dtype=torch.float32).reshape(576, 4)
        bias = torch.arange(576, dtype=torch.float32)
        converted = _minimax.convert_minimax_h3_image_checkpoint({
            'video_patch_proj.weight': torch.zeros(2, 2),
            'video_patch_proj.bias': torch.zeros(2),
            'final_layer.video_out.weight': weight,
            'final_layer.video_out.bias': bias,
        })
        self.assertIn('proj_in.weight', converted)
        self.assertNotIn('proj_out.weight', converted)
        self.assertEqual(tuple(converted['pdd_proj_out.0.weight'].shape), (96, 4))
        self.assertEqual(tuple(converted['pdd_proj_out.5.bias'].shape), (96,))
        self.assertTrue(torch.equal(converted['pdd_proj_out.0.weight'], weight[:96]))
        self.assertTrue(torch.equal(
            converted['pdd_proj_out.1.weight'], weight[:96] + weight[96:192]))
        self.assertTrue(torch.equal(
            converted['pdd_proj_out.1.bias'], bias[:96] + bias[96:192]))

        wide = torch.arange(576 * 5376, dtype=torch.float32).reshape(576, 5376)
        wide_converted = _minimax.convert_minimax_h3_image_checkpoint({
            'video_patch_proj.weight': torch.zeros(5376, 96),
            'final_layer.video_out.weight': wide,
        })
        self.assertEqual(tuple(wide_converted['pdd_proj_out.0.weight'].shape), (96, 5376))
        self.assertTrue(torch.equal(
            wide_converted['pdd_proj_out.0.weight'], wide[:96]))

    def test_image_comfy_export_fuses_output_heads(self):
        import torch
        import diffusers
        import dgenerate.pipelinewrapper.minimaxh3 as _minimax
        import dgenerate.pipelinewrapper.quant_skips as _quant_skips
        import dgenerate.quantize.comfykeys as _comfykeys

        config = {
            'num_attention_heads': 2,
            'attention_head_dim': 16,
            'hidden_size': 32,
            'num_layers': 1,
            'num_refiner_layers': 1,
            'ffn_dim': 32,
            'in_channels': 4,
            'patch_size': [1, 2, 2],
            'text_dim': 16,
            'freq_dim': 8,
            'time_embed_hidden_dim': 16,
            'time_embed_dim': 16,
            'rope_freq_dim': 16,
            'num_pdd_heads': 6,
        }
        model = diffusers.MiniMaxH3ImageTransformer2DModel.from_config(config)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()
                    if key.startswith('pdd_proj_out.')}
        exported, label = _comfykeys.comfy_state(
            model.state_dict(), {'_class_name': 'MiniMaxH3ImageTransformer2DModel'}, gguf=False)
        self.assertEqual(label, 'minimax_h3')
        self.assertIn('final_layer.video_out.weight', exported)
        self.assertNotIn('pdd_proj_out.0.weight', exported)
        restored = _minimax.convert_minimax_h3_image_checkpoint(exported)
        for key, value in original.items():
            self.assertIn(key, restored)
            self.assertTrue(torch.allclose(restored[key], value), key)
        skips = _quant_skips.architecture_skip_keys('MiniMaxH3ImageTransformer2DModel')
        self.assertIn('pdd_proj_out', skips)
        self.assertIn('proj_in', skips)

    def test_reference_images_may_differ_in_size(self):
        import dgenerate.mediainput as _mediainput

        uri = ('images: examples/media/man-fighting-pose.jpg, '
               'examples/media/earth.jpg')
        with self.assertRaises(_mediainput.ImageSeedSizeMismatchError):
            with next(_mediainput.iterate_image_seed(
                    uri, resize_resolution=(2048, 2048),
                    check_dimensions_match=True)) as seed:
                pass

        with next(_mediainput.iterate_image_seed(
                uri, resize_resolution=None, check_dimensions_match=False)) as seed:
            self.assertEqual(len(seed.images), 2)
            self.assertNotEqual(seed.images[0].size, seed.images[1].size)


if __name__ == '__main__':
    unittest.main()
