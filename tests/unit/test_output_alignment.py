import unittest
import unittest.mock
from types import SimpleNamespace

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.wrapper as _wrapper


class TestOutputAlignment(unittest.TestCase):
    def test_model_specific_output_alignment(self):
        cases = (
            (_enums.ModelType.S_CASCADE, 128),
            (_enums.ModelType.SD3, 16),
            (_enums.ModelType.SD3_PIX2PIX, 16),
            (_enums.ModelType.FLUX2, 16),
            (_enums.ModelType.Z_IMAGE, 16),
            (_enums.ModelType.QWEN_IMAGE, 16),
            (_enums.ModelType.SDXL, 8),
            (_enums.ModelType.FLUX, 8),
        )

        wrapper = _wrapper.DiffusionPipelineWrapper.__new__(
            _wrapper.DiffusionPipelineWrapper)
        for model_type, expected_alignment in cases:
            with self.subTest(model_type=model_type):
                wrapper._model_type = model_type
                self.assertEqual(wrapper._output_align(), expected_alignment)

    def test_aligned_user_dimensions_snap_to_alignment_minimum(self):
        wrapper = _wrapper.DiffusionPipelineWrapper.__new__(
            _wrapper.DiffusionPipelineWrapper)
        wrapper._model_type = _enums.ModelType.QWEN_IMAGE

        self.assertEqual(
            wrapper._aligned_user_dimensions(
                10, 33, 'txt2img'),
            (16, 32))
        self.assertEqual(
            wrapper._aligned_user_dimensions(
                1, None, 'txt2img'),
            (16, None))

    def test_txt2img_reports_both_aligned_dimensions_once(self):
        wrapper = _wrapper.DiffusionPipelineWrapper.__new__(
            _wrapper.DiffusionPipelineWrapper)
        wrapper._model_type = _enums.ModelType.SDXL
        user_args = SimpleNamespace(width=1001, height=1003)

        with unittest.mock.patch(
                'dgenerate.pipelinewrapper.wrapper._messages.warning') as warning:
            wrapper._set_pipeline_txt2img_defaults(user_args, {})

        warning.assert_called_once()
        self.assertIn('1001x1003 to 1000x1000', warning.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
