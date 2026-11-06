import unittest

from dgenerate.extras.ras.utils.ras_manager import ras_manager


class TestRasShortSchedule(unittest.TestCase):
    def test_error_reset_past_the_end_is_ignored(self):
        manager = ras_manager()
        manager.num_steps = 10
        manager.height = 1024
        manager.weight = 1024
        manager.skip_num_step = 0
        manager.skip_num_step_length = 0
        manager.scheduler_start_step = 4
        manager.error_reset_steps = [12, 22]
        manager.generate_skip_token_list()
        self.assertEqual(len(manager.skip_token_num_list), 10)
        self.assertEqual(manager.skip_token_num_list[:4], [0, 0, 0, 0])
        self.assertTrue(all(count > 0 for count in manager.skip_token_num_list[4:]))

    def test_error_reset_inside_a_28_step_run_is_kept(self):
        manager = ras_manager()
        manager.num_steps = 28
        manager.height = 1024
        manager.weight = 1024
        manager.skip_num_step = 0
        manager.skip_num_step_length = 0
        manager.scheduler_start_step = 4
        manager.error_reset_steps = [12, 22]
        manager.generate_skip_token_list()
        self.assertEqual(manager.skip_token_num_list[12], 0)
        self.assertEqual(manager.skip_token_num_list[22], 0)
        self.assertGreater(manager.skip_token_num_list[13], 0)


if __name__ == '__main__':
    unittest.main()
