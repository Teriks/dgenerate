"""Unit tests for pipeline sampling defaults."""

import unittest

import dgenerate.pipelinewrapper.arguments as _arguments
import dgenerate.pipelinewrapper.sampling as _sampling
from diffusers import WanPipeline


class PipelineSamplingDefaultsTests(unittest.TestCase):

    def test_wan_pipeline_omitted_steps_and_guidance(self):
        args = _arguments.DiffusionArguments()
        args.inference_steps = None
        args.guidance_scale = None
        steps = _sampling.resolved_inference_steps(WanPipeline, args)
        guidance = _sampling.resolved_guidance_scale(WanPipeline, args)
        self.assertEqual(steps, 50)
        self.assertEqual(guidance, 5.0)
        self.assertEqual(args.inference_steps, 50)
        self.assertEqual(args.guidance_scale, 5.0)

    def test_explicit_values_are_unchanged(self):
        args = _arguments.DiffusionArguments()
        args.inference_steps = 30
        args.guidance_scale = 4.0
        steps = _sampling.resolved_inference_steps(WanPipeline, args)
        guidance = _sampling.resolved_guidance_scale(WanPipeline, args)
        self.assertEqual(steps, 30)
        self.assertEqual(guidance, 4.0)


if __name__ == '__main__':
    unittest.main()
