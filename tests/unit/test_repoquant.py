import json
import os
import tempfile
import unittest

import torch

import dgenerate.quantize.quantexport as quantexport
import dgenerate.quantize.repoquant as repoquant
import dgenerate.subcommands.quantize as quantize_command


def _tiny_unet(folder):
    from diffusers import UNet2DModel

    UNet2DModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32,),
        down_block_types=('DownBlock2D',),
        up_block_types=('UpBlock2D',),
    ).save_pretrained(folder)


def _tiny_clip(folder):
    from transformers import CLIPTextConfig, CLIPTextModel

    CLIPTextModel(CLIPTextConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=8,
    )).save_pretrained(folder)


def _pipeline(root):
    os.makedirs(os.path.join(root, 'scheduler'))
    _tiny_unet(os.path.join(root, 'unet'))
    _tiny_clip(os.path.join(root, 'text_encoder'))
    with open(os.path.join(root, 'scheduler', 'scheduler_config.json'), 'w', encoding='utf-8') as handle:
        json.dump({'_class_name': 'EulerDiscreteScheduler'}, handle)
    with open(os.path.join(root, 'model_index.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            '_class_name': 'StableDiffusionPipeline',
            'unet': ['diffusers', 'UNet2DModel'],
            'text_encoder': ['transformers', 'CLIPTextModel'],
            'scheduler': ['diffusers', 'EulerDiscreteScheduler'],
        }, handle)


class TestRepoFormats(unittest.TestCase):
    def test_quantizer_uri_keeps_the_requested_config(self):
        spec = repoquant.parse_quantizer(
            'sdnq;type=uint4;group-size=32;quant-conv=true;use-svd=true')
        self.assertEqual(spec.folder, 'sdnq-uint4-g32-conv-svd')
        config = spec.config('diffusers')
        self.assertEqual(config.weights_dtype, 'uint4')
        self.assertEqual(config.group_size, 32)
        self.assertTrue(config.quant_conv)
        self.assertTrue(config.use_svd)
        bnb = repoquant.parse_quantizer(
            'bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=true;'
            'bits4-compute-dtype=bfloat16')
        self.assertEqual(bnb.folder, 'bnb-4-nf4-dq-compute-bfloat16')
        bnb_config = bnb.config('transformers')
        self.assertTrue(bnb_config.load_in_4bit)
        self.assertEqual(bnb_config.bnb_4bit_quant_type, 'nf4')
        self.assertTrue(bnb_config.bnb_4bit_use_double_quant)
        self.assertEqual(bnb_config.bnb_4bit_compute_dtype, torch.bfloat16)
        with self.assertRaises(quantexport.QuantExportError):
            repoquant.parse_quantizer('quanto')
        with self.assertRaises(quantexport.QuantExportError):
            quantexport.resolve_formats(['sdnq'])

    def test_sdnq_repo_keeps_the_pipeline_and_reloads(self):
        from diffusers import UNet2DModel
        from transformers import CLIPTextModel

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            _pipeline(root)
            destination = repoquant.export_pretrained_repo(
                root,
                quantexport.component_directories(root),
                repoquant.parse_quantizer('sdnq;type=uint4;group-size=64;use-svd=true'),
                output)
            self.assertTrue(os.path.isfile(os.path.join(destination, 'model_index.json')))
            self.assertTrue(os.path.isfile(
                os.path.join(destination, 'scheduler', 'scheduler_config.json')))
            for component in ('unet', 'text_encoder'):
                with open(os.path.join(destination, component, 'quantization_config.json'),
                          encoding='utf-8') as handle:
                    config = json.load(handle)
                self.assertEqual(config['quant_method'], 'sdnq')
                self.assertEqual(config['weights_dtype'], 'uint4')
                self.assertEqual(config['group_size'], 64)
                self.assertTrue(config['use_svd'])
            text = CLIPTextModel.from_pretrained(
                os.path.join(destination, 'text_encoder'), low_cpu_mem_usage=True)
            layer_names = {type(module).__name__ for module in text.modules()}
            self.assertTrue(any('SDNQ' in name for name in layer_names), layer_names)
            UNet2DModel.from_pretrained(os.path.join(destination, 'unet'), low_cpu_mem_usage=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'bitsandbytes prequant needs CUDA')
    def test_bnb_repo_reloads_without_a_quantizer_argument(self):
        from diffusers import UNet2DModel

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'unet')
            _tiny_unet(folder)
            with open(os.path.join(root, 'model_index.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    '_class_name': 'StableDiffusionPipeline',
                    'unet': ['diffusers', 'UNet2DModel'],
                }, handle)
            destination = repoquant.export_pretrained_repo(
                root, [('unet', folder)],
                repoquant.parse_quantizer('bnb;bits=4;bits4-quant-type=nf4'),
                output)
            loaded = UNet2DModel.from_pretrained(
                os.path.join(destination, 'unet'), device_map={'': 'cuda:0'})
            self.assertTrue(loaded.is_loaded_in_4bit)

    def test_subcommand_writes_the_repository(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            _pipeline(root)
            status = quantize_command.QuantizeSubCommand(
                loaded_by_name='quantize',
                plugin_module_paths=[],
                args=[root, '--quantizer', 'sdnq;type=int4;group-size=32', '-o', output],
            )()
            self.assertEqual(status, 0)
            config_path = os.path.join(output, 'sdnq-int4-g32', 'unet', 'quantization_config.json')
            with open(config_path, encoding='utf-8') as handle:
                config = json.load(handle)
            self.assertEqual(config['weights_dtype'], 'int4')
            self.assertEqual(config['group_size'], 32)
