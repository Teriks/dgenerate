import unittest

import dgenerate.batchprocess.configrunnerbuiltins as _builtins


class TestConfigRunnerBuiltinsFormatImageSeed(unittest.TestCase):
    def test_configrunner_registers_function(self):
        from dgenerate.batchprocess import ConfigRunner
        runner = ConfigRunner()
        self.assertIs(
            runner.template_functions['format_image_seed'],
            _builtins.format_image_seed)
        self.assertIs(
            runner.template_functions['format_image_processors'],
            _builtins.format_image_processors)

    def test_formats_ltx_conditions(self):
        self.assertEqual(
            _builtins.format_image_seed(
                seed_images='first.png',
                ltx_index=0,
                ltx_extra_conditions=[('last.png', -1, 0.8)]),
            'first.png;ltx-index=0 ++ last.png;ltx-index=-1;strength=0.8')

    def test_formats_wan_video_conditions(self):
        self.assertEqual(
            _builtins.format_image_seed(
                seed_images='character.png',
                wan_pose_video='pose.mp4',
                wan_face_video='face.mp4'),
            'character.png;wan-pose=pose.mp4;wan-face=face.mp4')

    def test_preserves_control_only_input(self):
        self.assertEqual(
            _builtins.format_image_seed(
                control_images='control.mp4', preserve_control=True),
            'control=control.mp4')

    def test_formats_image_processor_chains(self):
        format_processors = _builtins.format_image_processors
        self.assertEqual(format_processors([]), '')
        self.assertEqual(format_processors(['first']), 'first')
        self.assertEqual(format_processors([None, 'last']), ' + last')
        self.assertEqual(format_processors(['first', 'last']), 'first + last')
        self.assertEqual(
            format_processors(['first processor', 'last']),
            "'first processor' + last")
        self.assertEqual(format_processors(['first', None]), 'first + ')
        self.assertEqual(format_processors([None, None]), '')


if __name__ == '__main__':
    unittest.main()
