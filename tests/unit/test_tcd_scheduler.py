import unittest
from unittest.mock import patch

import diffusers

import dgenerate.pipelinewrapper.schedulers as _schedulers


class TestTCDScheduler(unittest.TestCase):
    def test_eta_is_accepted_and_not_a_constructor_argument(self):
        args = _schedulers.parse_scheduler_uri(
            diffusers.TCDScheduler, 'TCDScheduler;eta=1.0')
        self.assertEqual(args['eta'], 1.0)
        self.assertNotIn('eta', diffusers.TCDScheduler.__init__.__annotations__)
        names = [
            scheduler.__name__
            for scheduler in _schedulers.get_compatible_schedulers(
                diffusers.StableDiffusionPipeline)
        ]
        self.assertIn('TCDScheduler', names)

    def test_eta_is_rejected_for_other_schedulers(self):
        with self.assertRaises(_schedulers.SchedulerArgumentError):
            _schedulers.parse_scheduler_uri(
                diffusers.DDIMScheduler, 'DDIMScheduler;eta=1.0')

    def test_load_stashes_eta_for_the_pipeline_call(self):
        pipeline = type('Pipe', (), {})()
        pipeline.scheduler = diffusers.DDIMScheduler()

        with patch.object(
                _schedulers, 'get_compatible_schedulers',
                return_value=[diffusers.TCDScheduler]):
            _schedulers.load_scheduler(pipeline, 'TCDScheduler;eta=1.0')

        self.assertIsInstance(pipeline.scheduler, diffusers.TCDScheduler)
        self.assertEqual(pipeline.scheduler._dgenerate_call_eta, 1.0)
        self.assertNotIn('eta', pipeline.scheduler.config)


if __name__ == '__main__':
    unittest.main()
