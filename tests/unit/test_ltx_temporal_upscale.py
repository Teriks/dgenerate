import types
import unittest
from unittest import mock

import torch

from dgenerate.pipelinewrapper import DiffusionArguments
from dgenerate.pipelinewrapper.videopipelines import ltx


class LTXTemporalUpscaleTest(unittest.TestCase):
    def test_temporal_dfr_composes_two_spatial_passes_then_temporal_round(self):
        calls = []

        class DFR:
            scheduler = object()

            def __call__(self, prompt=None, conditions=None, width=None, height=None,
                         num_frames=None, frame_rate=None, output_type=None,
                         return_dict=None, generate_slots=None, sigmas=None,
                         latents=None, audio_latents=None, keyframes_latents=None,
                         keyframe_positions=None, reference_latents=None,
                         reference_downscale_factor=None, noise_scale=None,
                         generator=None, use_cross_timestep=True,
                         max_sequence_length=1024, **kwargs):
                pass

        class Temporal:
            def __call__(self, prompt=None, conditions=None, latents=None,
                         keyframes_latents=None, keyframe_positions=None,
                         audio_latents=None, height=None, width=None,
                         num_frames=None, frame_rate=None, source_seconds=None,
                         condition_num_frames=None, round_index=None, sigmas=None,
                         generator=None, use_cross_timestep=True,
                         output_type=None, return_dict=None,
                         max_sequence_length=1024):
                pass

        class Upsampler:
            config = types.SimpleNamespace(spatial_upsample=False)

            def to(self, *args, **kwargs):
                return self

        class Pipe:
            transformer = types.SimpleNamespace(
                config=types.SimpleNamespace(use_keyframes_abs_pos_embedding=True))
            vae_temporal_compression_ratio = 8

        class Wrapper:
            _dtype = None
            device = 'cpu'
            model_cpu_offload = False
            model_sequential_offload = False

        def make_output():
            return types.SimpleNamespace(
                frames=torch.zeros(1, 128, 3, 4, 4),
                audio=torch.zeros(1, 8, 10, 16),
                keyframes=torch.zeros(1, 128, 1, 4, 4),
                keyframe_positions=[8])

        def invoke(wrapper, pipeline, kwargs, user_args=None):
            calls.append((pipeline, kwargs))
            return make_output()

        args = DiffusionArguments()
        args.ltx_latent_upscale_mode = 'temporal'
        args.ltx_temporal_upscale_rounds = 1
        args.ltx_stg_scale = None
        args.ltx_modality_scale = None
        args.ltx_stage_guidance_scale = None
        args.ltx_stage_audio_guidance_scale = None
        args.ltx_stage_lora_uris = None
        args.ltx_stage_sigmas = None
        args.ltx_noise_scale = None
        args.reference_video_frames = None
        args.ltx_decode_timestep = None

        with mock.patch('diffusers.LTX2DFRPipeline.from_pipe', return_value=DFR()), \
                mock.patch('diffusers.LTX2DFRTemporalRefinePipeline.from_pipe',
                           return_value=Temporal()), \
                mock.patch.object(ltx._vp, '_invoke', side_effect=invoke), \
                mock.patch.object(ltx, '_upsample_ltx_latents',
                                  return_value=torch.zeros(1, 128, 3, 8, 8)) as upsample, \
                mock.patch.object(ltx, '_load_ltx_latent_upsampler', return_value=Upsampler()), \
                mock.patch.object(ltx, '_ltx_pixels_from_output', return_value=([], None, None)):
            result = ltx._ltx_temporal_upscale(
                Wrapper(), Pipe(), types.SimpleNamespace(), args,
                {'prompt': 'test', 'width': 768, 'height': 512, 'num_frames': 17,
                 'frame_rate': 24.0, 'guidance_scale': 1.0, 'generator': None},
                diffusion=True)

        self.assertEqual(result, ([], None, None))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1]['width'], 384)
        self.assertEqual(calls[0][1]['height'], 256)
        self.assertTrue(calls[0][1]['generate_slots'])
        upsample.assert_called_once()
        self.assertEqual(calls[1][1]['reference_downscale_factor'], 2)
        self.assertEqual(calls[1][1]['keyframe_positions'], [8])
        self.assertEqual(calls[2][1]['round_index'], 1)
        self.assertEqual(calls[2][1]['num_frames'], 17)
        self.assertEqual(calls[2][1]['frame_rate'], 24.0)


if __name__ == '__main__':
    unittest.main()
