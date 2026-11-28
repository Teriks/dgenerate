import functools
import unittest

import diffusers
import numpy
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps

import dgenerate.pipelinewrapper.arguments as _arguments
import dgenerate.pipelinewrapper.denoise_range as _denoise_range
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines.ltx.legacy as _ltx_legacy
import dgenerate.pipelinewrapper.videopipelines.schedules as _schedules
import dgenerate.pipelinewrapper.wrapper as _wrapper
import dgenerate.renderloopconfig as _renderloopconfig


def _flow_match():
    return diffusers.FlowMatchEulerDiscreteScheduler(shift=5.0)


def _stub_pipeline(pipeline_cls, scheduler):
    stub = type('Pipe', (pipeline_cls,), {'__setattr__': object.__setattr__})
    pipeline = object.__new__(stub)
    pipeline.scheduler = scheduler
    return pipeline


class TestForcedSchedules(unittest.TestCase):
    def test_forced_sigmas_replace_a_pipeline_sigma_list(self):
        sigmas = [1.0, 0.75, 0.5, 0.25]
        expected = _flow_match()
        expected.set_timesteps(sigmas=sigmas, device='cpu')

        scheduler = _flow_match()
        with _schedules._forced_scheduler_sigmas(scheduler, sigmas):
            timesteps, steps = retrieve_timesteps(
                scheduler, 30, 'cpu', sigmas=numpy.linspace(1.0, 1 / 30, 30))
        self.assertEqual(steps, 4)
        self.assertTrue(numpy.allclose(timesteps.numpy(), expected.timesteps.numpy()))
        self.assertNotIn('set_timesteps', scheduler.__dict__)

    def test_forced_sigmas_replace_a_step_count(self):
        sigmas = [1.0, 0.6, 0.2]
        scheduler = _flow_match()
        with _schedules._forced_scheduler_sigmas(scheduler, sigmas):
            scheduler.set_timesteps(30, device='cpu')
        self.assertEqual(len(scheduler.timesteps), 3)
        self.assertEqual(_schedules._sigma_step_count(_flow_match(), sigmas), 3)

    def test_forced_timesteps_pass_retrieve_timesteps(self):
        scheduler = _flow_match()
        with _schedules._forced_scheduler_timesteps(scheduler, [900, 500, 100]):
            timesteps, steps = retrieve_timesteps(scheduler, 30, 'cpu', timesteps=[1, 2, 3])
        self.assertEqual(steps, 3)
        self.assertEqual([int(value) for value in timesteps], [900, 500, 100])
        self.assertTrue(numpy.allclose(scheduler.sigmas[:3].numpy(), [0.9, 0.5, 0.1]))

    def _wrapper_for(self, pipeline):
        wrapper = object.__new__(_wrapper.PipelineWrapper)
        wrapper._forced_schedules = []
        wrapper._forced_active = set()
        wrapper._pipeline = pipeline
        return wrapper

    def test_denoise_range_slices_a_forced_schedule(self):
        sigmas = [1.0, 0.8, 0.6, 0.4]
        pipeline = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, _flow_match())
        wrapper = self._wrapper_for(pipeline)
        wrapper._forced_schedules.append((pipeline, 'sigmas', sigmas))

        with wrapper._forced_schedule_context(pipeline), \
                _denoise_range.denoise_range(pipeline, 0.5, 1.0), \
                wrapper._forced_schedule_context(pipeline):
            pipeline.scheduler.set_timesteps(30, device='cpu')
        expected = _flow_match()
        expected.set_timesteps(sigmas=sigmas, device='cpu')
        self.assertTrue(numpy.allclose(pipeline.scheduler.sigmas.numpy(), expected.sigmas[2:].numpy()))
        self.assertNotIn('set_timesteps', pipeline.scheduler.__dict__)

    def test_forced_schedule_forwards_mu_through_an_open_patch(self):
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        received = {}

        def patched(num_inference_steps=None, device=None, **kwargs):
            received.update(kwargs)
            return type(scheduler).set_timesteps(
                scheduler, num_inference_steps, device=device, **kwargs)

        scheduler.set_timesteps = patched
        with _schedules._forced_scheduler_sigmas(scheduler, [1.0, 0.5]):
            scheduler.set_timesteps(30, device='cpu', mu=0.5)
        self.assertEqual(received.get('mu'), 0.5)
        self.assertIs(scheduler.set_timesteps, patched)

    def test_wrapper_forces_sigmas_the_call_does_not_accept(self):
        pipeline = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, _flow_match())
        self.assertNotIn('sigmas', _wrapper.PipelineWrapper._call_argument_names(pipeline))
        wrapper = self._wrapper_for(pipeline)
        user_args = _arguments.GenerationArguments()
        user_args.sigmas = [1.0, 0.75, 0.5, 0.25]
        pipeline_args = {}
        wrapper._set_non_universal_pipeline_arg(
            pipeline, pipeline_args, user_args, 'sigmas', 'sigmas', '--sigmas',
            transform=functools.partial(
                _wrapper.PipelineWrapper._sigmas_eval, 'primary', pipeline, 4))
        self.assertNotIn('sigmas', pipeline_args)
        self.assertEqual(pipeline_args['num_inference_steps'], 4)

        with wrapper._forced_schedule_context(pipeline):
            pipeline.scheduler.set_timesteps(30, device='cpu')
        self.assertEqual(len(pipeline.scheduler.timesteps), 4)

        other = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, _flow_match())
        with wrapper._forced_schedule_context(other):
            other.scheduler.set_timesteps(30, device='cpu')
        self.assertEqual(len(other.scheduler.timesteps), 30)

    def test_wrapper_forces_timesteps_the_call_does_not_accept(self):
        pipeline = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, _flow_match())
        wrapper = self._wrapper_for(pipeline)
        user_args = _arguments.GenerationArguments()
        user_args.timesteps = [900, 600, 300]
        pipeline_args = {}
        wrapper._set_timesteps_pipeline_arg(pipeline_args, user_args)
        self.assertNotIn('timesteps', pipeline_args)
        self.assertEqual(pipeline_args['num_inference_steps'], 3)
        self.assertEqual(user_args.inference_steps, 3)

        with wrapper._forced_schedule_context(pipeline):
            pipeline.scheduler.set_timesteps(30, device='cpu')
        self.assertEqual([int(value) for value in pipeline.scheduler.timesteps], [900, 600, 300])

    def test_scheduler_without_the_schedule_still_raises(self):
        pipeline = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, diffusers.PNDMScheduler())
        wrapper = self._wrapper_for(pipeline)
        wrapper._model_type = _wrapper._enums.ModelType.KANDINSKY5_IMAGE
        wrapper._pipeline_type = _wrapper._enums.PipelineType.TXT2IMG
        user_args = _arguments.GenerationArguments()
        user_args.sigmas = [1.0, 0.5]
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            wrapper._set_non_universal_pipeline_arg(
                pipeline, {}, user_args, 'sigmas', 'sigmas', '--sigmas')
        self.assertEqual(wrapper._forced_schedules, [])

    def test_sigmas_record_the_step_count_that_runs(self):
        pipeline = _stub_pipeline(diffusers.Kandinsky5T2IPipeline, _flow_match())
        wrapper = self._wrapper_for(pipeline)
        user_args = _arguments.GenerationArguments()
        user_args.inference_steps = 30
        user_args.sigmas = [1.0, 0.75, 0.5, 0.25]
        pipeline_args = {'num_inference_steps': 30}
        wrapper._set_schedule_pipeline_args(pipeline_args, user_args)
        self.assertEqual(pipeline_args['num_inference_steps'], 4)
        self.assertEqual(user_args.inference_steps, 4)

    def test_legacy_ltx_condition_sigmas_match_the_other_modes(self):
        shifting = diffusers.FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True, base_shift=0.95, max_shift=2.05,
            base_image_seq_len=1024, max_image_seq_len=4096, shift_terminal=0.1)
        size = (704, 480, 121)
        sigmas = [1.0, 0.8, 0.5, 0.2]
        expected = _ltx_legacy._legacy_ltx_probe(shifting, sigmas, *size, 32, 8)

        unshifted = shifting.__class__.from_config({
            **shifting.config, 'use_dynamic_shifting': False, 'shift': 1.0, 'shift_terminal': None})
        unshifted.set_timesteps(
            sigmas=_ltx_legacy._legacy_ltx_shifted_sigmas(shifting, sigmas, *size), device='cpu')
        self.assertTrue(numpy.allclose(unshifted.sigmas.numpy(), expected.sigmas.numpy()))
        self.assertFalse(numpy.allclose(expected.sigmas[:-1].numpy(), sigmas))

    def test_upscalers_accept_timesteps(self):
        for model_type in ('upscaler-x2', 'upscaler-x4'):
            config = _renderloopconfig.RenderLoopConfig()
            config.model_type = model_type
            config.timesteps = [[900, 500, 100]]
            config._check_timesteps_requirements(lambda name: name)
        config = _renderloopconfig.RenderLoopConfig()
        config.model_type = 's-cascade'
        config.timesteps = [[900, 500, 100]]
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            config._check_timesteps_requirements(lambda name: name)

    def test_video_linspace_pipelines(self):
        self.assertTrue(_wrapper.model_type_uses_flow_default_sigmas('hyvideo-15'))
        self.assertTrue(_wrapper.model_type_uses_flow_default_sigmas('ltx-video'))
        self.assertFalse(_wrapper.model_type_uses_flow_default_sigmas('wan'))
        self.assertFalse(_wrapper.model_type_uses_flow_default_sigmas('ltx2'))


if __name__ == '__main__':
    unittest.main()
