import unittest

import diffusers

import dgenerate.pipelinewrapper.schedulers as _schedulers

# Pipeline classes the video loaders build, and the --model-type that offers
# their schedulers in the console. load_scheduler checks the class.
_LOADED_CLASSES = {
    'motif-video': ('MotifVideoPipeline', 'MotifVideoImage2VideoPipeline'),
    'wan-animate-2': ('WanAnimate2ModularPipeline', 'WanAnimate2DistilledModularPipeline'),
    'minimax-h3-video': ('MiniMaxH3ModularPipeline',),
    'wan': ('WanPipeline', 'WanImageToVideoPipeline'),
    'wan-animate': ('WanAnimatePipeline',),
    'hyvideo-15': ('HunyuanVideo15Pipeline', 'HunyuanVideo15ImageToVideoPipeline'),
    'ltx2': ('LTX2Pipeline', 'LTX2ImageToVideoPipeline', 'LTX2ConditionPipeline',
             'LTX2InContextPipeline'),
    'flux3': ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline'),
}


class TestSchedulerCompatibility(unittest.TestCase):
    def test_loaded_pipelines_accept_offered_schedulers(self):
        for model_type, class_names in _LOADED_CLASSES.items():
            offered = _schedulers.compatible_scheduler_names(model_type)
            self.assertTrue(offered, model_type)
            for class_name in class_names:
                pipeline_cls = getattr(diffusers, class_name, None)
                if pipeline_cls is None:
                    continue
                with self.subTest(model_type=model_type, pipeline=class_name):
                    accepted = {
                        scheduler.__name__
                        for scheduler in _schedulers.get_compatible_schedulers(pipeline_cls)}
                    self.assertLessEqual(set(offered), accepted)

    def test_piflow_rejects_custom_schedules(self):
        self.assertFalse(_schedulers.scheduler_accepts_custom_timesteps('PiflowScheduler'))
        self.assertFalse(_schedulers.scheduler_accepts_custom_sigmas('PiflowScheduler'))
        self.assertTrue(_schedulers.scheduler_accepts_custom_timesteps(
            diffusers.FlowMatchEulerDiscreteScheduler()))

    def test_load_scheduler_on_motif_video(self):
        stub = type('Pipe', (diffusers.MotifVideoPipeline,), {'__setattr__': object.__setattr__})
        pipeline = object.__new__(stub)
        pipeline.scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        _schedulers.load_scheduler(pipeline, 'FlowMatchEulerDiscreteScheduler;shift=3.0')
        self.assertEqual(pipeline.scheduler.config.shift, 3.0)


if __name__ == '__main__':
    unittest.main()
