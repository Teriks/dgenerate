import unittest
from unittest import mock

import torch

from dgenerate.extras.hidiffusion.hidiffusion import apply_hidiffusion, remove_hidiffusion


class DiffusionPipeline:
    pass


class _Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)


class _DownBlock(torch.nn.Module):
    def __init__(self, with_downsampler=False):
        super().__init__()
        if with_downsampler:
            self.downsamplers = torch.nn.ModuleList([_Holder()])


class _UpBlock(torch.nn.Module):
    def __init__(self, with_upsampler=False):
        super().__init__()
        if with_upsampler:
            self.upsamplers = torch.nn.ModuleList([_Holder()])


class _UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_upsamplers = 4
        self.down_blocks = torch.nn.ModuleList([
            _DownBlock(with_downsampler=True),
            _DownBlock(),
        ])
        self.up_blocks = torch.nn.ModuleList([
            torch.nn.Module(),
            torch.nn.Module(),
            _UpBlock(with_upsampler=True),
        ])


class _Pipe(DiffusionPipeline):
    def __init__(self):
        self.unet = _UNet()
        self.name_or_path = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        self._num_timesteps = 10


class StableDiffusionXLControlNetPipeline(DiffusionPipeline):
    pass


class _ControlNetPipe(StableDiffusionXLControlNetPipeline):
    def __init__(self):
        self.controlnet = object()
        self.unet = _UNet()
        self.name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'


class TestHiDiffusionCleanup(unittest.TestCase):

    def test_remove_restores_unet_and_drops_pipeline_ref(self):
        pipe = _Pipe()
        conv = pipe.unet.down_blocks[0].downsamplers[0].conv
        original_conv_cls = type(conv)
        original_block_cls = type(pipe.unet.down_blocks[1])
        original_upsamplers = pipe.unet.num_upsamplers

        apply_hidiffusion(pipe, apply_window_attn=False)
        self.assertEqual(pipe.unet.num_upsamplers, original_upsamplers + 12)
        self.assertIsNot(type(conv), original_conv_cls)
        self.assertIsNot(type(pipe.unet.down_blocks[1]), original_block_cls)
        self.assertTrue(pipe.unet._forward_pre_hooks)
        info = conv.info
        self.assertIs(info["pipeline"], pipe)
        self.assertIs(pipe.info, info)

        # Interrupted RAU-Net step: stride was swapped and never restored.
        conv.ori_stride = conv.stride
        conv.ori_padding = conv.padding
        conv.ori_dilation = conv.dilation
        conv.stride = (4, 4)
        conv.padding = (2, 2)
        conv.dilation = (2, 2)

        remove_hidiffusion(pipe)

        self.assertEqual(pipe.unet.num_upsamplers, original_upsamplers)
        self.assertIs(type(conv), original_conv_cls)
        self.assertIs(type(pipe.unet.down_blocks[1]), original_block_cls)
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (1, 1))
        self.assertEqual(conv.dilation, (1, 1))
        self.assertFalse(pipe.unet._forward_pre_hooks)
        self.assertNotIn("info", pipe.__dict__)
        self.assertNotIn("info", pipe.unet.__dict__)
        self.assertNotIn("info", conv.__dict__)
        self.assertNotIn("model", conv.__dict__)
        self.assertIsNone(info["pipeline"])
        self.assertEqual(info["hooks"], [])

        apply_hidiffusion(pipe, apply_window_attn=False)
        self.assertEqual(pipe.unet.num_upsamplers, original_upsamplers + 12)
        remove_hidiffusion(pipe)
        self.assertEqual(pipe.unet.num_upsamplers, original_upsamplers)

    def test_remove_restores_sdxl_controlnet_pipeline_class(self):
        pipe = _ControlNetPipe()
        original_pipe_cls = type(pipe)
        original_unet_cls = type(pipe.unet)

        apply_hidiffusion(pipe, apply_raunet=False, apply_window_attn=False)
        self.assertIsNot(type(pipe), original_pipe_cls)
        self.assertIsNot(type(pipe.unet), original_unet_cls)
        self.assertTrue(type(pipe).__dict__.get("_hidiffusion_patched"))
        self.assertTrue(type(pipe.unet).__dict__.get("_hidiffusion_patched"))

        remove_hidiffusion(pipe)

        self.assertIs(type(pipe), original_pipe_cls)
        self.assertIs(type(pipe.unet), original_unet_cls)
        self.assertNotIn("info", pipe.__dict__)
        self.assertNotIn("info", pipe.unet.__dict__)
        self.assertEqual(pipe.unet.num_upsamplers, 4)

    def test_downsampler_restores_stride_when_conv_raises(self):
        pipe = _Pipe()
        apply_hidiffusion(pipe, apply_window_attn=False)
        conv = pipe.unet.down_blocks[0].downsamplers[0].conv
        conv.info["size"] = (128, 128)

        import dgenerate.extras.hidiffusion.hidiffusion as hidiffusion

        with mock.patch.object(hidiffusion.F, "conv2d", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                conv.forward(torch.zeros(1, 1, 8, 8))

        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (1, 1))
        self.assertEqual(conv.dilation, (1, 1))
        self.assertEqual(conv.timestep, 0)

        conv.forward(torch.zeros(1, 1, 8, 8))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (1, 1))
        self.assertEqual(conv.dilation, (1, 1))
        self.assertEqual(conv.timestep, 1)

        remove_hidiffusion(pipe)
        self.assertEqual(conv.stride, (1, 1))
        self.assertNotIn("ori_stride", conv.__dict__)
        self.assertNotIn("timestep", conv.__dict__)

    def test_remove_is_safe_when_not_patched(self):
        pipe = _Pipe()
        remove_hidiffusion(pipe)
        self.assertEqual(pipe.unet.num_upsamplers, 4)
        self.assertNotIn("info", pipe.__dict__)
