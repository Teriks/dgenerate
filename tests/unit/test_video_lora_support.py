import json
import os
import tempfile
import unittest

import safetensors.torch
import torch

import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines  # noqa: F401  installs video LoRA loaders


def _video_pipeline_classes():
    import diffusers
    from dgenerate.pipelinewrapper.videopipelines.lora import _VIDEO_LORA_PIPELINE_CLASSES
    classes = []
    for name in _VIDEO_LORA_PIPELINE_CLASSES:
        pipeline_class = getattr(diffusers, name, None)
        if pipeline_class is not None:
            classes.append(pipeline_class)
    return classes


class TestVideoLoraSupport(unittest.TestCase):
    def test_every_video_pipeline_can_load_a_lora(self):
        classes = _video_pipeline_classes()
        self.assertGreaterEqual(len(classes), 18)
        for pipeline_class in classes:
            self.assertTrue(
                callable(getattr(pipeline_class, 'load_lora_weights', None)),
                pipeline_class.__name__)
            self.assertTrue(
                callable(getattr(pipeline_class, 'lora_state_dict', None)),
                pipeline_class.__name__)
            self.assertTrue(
                _pipelines.pipeline_class_supports_lora(pipeline_class),
                pipeline_class.__name__)

    def test_peft_header_is_not_passed_through_as_an_empty_config(self):
        import diffusers

        handle, path = tempfile.mkstemp(suffix='.safetensors')
        os.close(handle)
        self.addCleanup(os.remove, path)
        safetensors.torch.save_file(
            {
                'transformer.blocks.0.attn.to_q.lora_A.weight': torch.ones(4, 8),
                'transformer.blocks.0.attn.to_q.lora_B.weight': torch.ones(8, 4),
            },
            path,
            metadata={'lora_adapter_metadata': json.dumps({
                'r': 4,
                'lora_alpha': 8,
                'alpha_pattern': {},
                'rank_pattern': {},
                'target_modules': ['to_q'],
                'peft_type': 'LORA',
            })},
        )
        state, metadata = diffusers.WanPipeline.lora_state_dict(
            path, return_lora_metadata=True)
        self.assertIsNone(metadata)
        self.assertTrue(torch.equal(
            state['transformer.blocks.0.attn.to_q.lora_B.weight'],
            torch.full((8, 4), 2.0)))

    def test_motif_loads_a_musubi_lora_and_a_peft_header(self):
        from diffusers.models.transformers.transformer_motif_video import (
            MotifVideoTransformer3DModel,
        )
        from dgenerate.pipelinewrapper.videopipelines.motif_lora import (
            MotifVideoLoraLoaderMixin,
        )

        model = MotifVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_single_layers=1,
            num_decoder_layers=0,
            text_embed_dim=16,
            rope_axes_dim=(2, 3, 3),
        )
        hidden = 16

        class _Pipeline(MotifVideoLoraLoaderMixin):
            def __init__(self, transformer):
                self.transformer = transformer
                self.hf_device_map = None
                self.components = {'transformer': transformer}

        handle, path = tempfile.mkstemp(suffix='.safetensors')
        os.close(handle)
        self.addCleanup(os.remove, path)
        safetensors.torch.save_file(
            {
                'lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight': torch.ones(4, hidden),
                'lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight': torch.ones(hidden, 4),
                'lora_unet_transformer_blocks_0_attn_to_q.alpha': torch.tensor(8.0),
                'lora_unet_single_transformer_blocks_0_proj_mlp.lora_down.weight': torch.ones(4, hidden),
                'lora_unet_single_transformer_blocks_0_proj_mlp.lora_up.weight': torch.ones(hidden * 4, 4),
                'lora_unet_single_transformer_blocks_0_proj_mlp.alpha': torch.tensor(4.0),
            },
            path,
            metadata={'lora_adapter_metadata': json.dumps({
                'r': 4,
                'lora_alpha': 16,
                'alpha_pattern': {},
                'rank_pattern': {},
                'target_modules': ['attn_to_q'],
                'peft_type': 'LORA',
            })},
        )
        _Pipeline(model).load_lora_weights(path, adapter_name='style')
        query = model.transformer_blocks[0].attn.to_q
        self.assertIn('style', model.peft_config)
        # Kohya alpha 8 / rank 4 wins. The PEFT header is not applied again.
        self.assertTrue(torch.equal(query.lora_B['style'].weight, torch.full((hidden, 4), 2.0)))
        self.assertEqual(query.scaling['style'], 1.0)
        self.assertTrue(hasattr(model.single_transformer_blocks[0].proj_mlp, 'lora_A'))
