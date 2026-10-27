import unittest

import dgenerate.imageprocessors as _imageprocessors
import dgenerate.imageprocessors.exceptions as _exceptions


class TestFaceRestoreProcessor(unittest.TestCase):

    def test_registered(self):
        self.assertIn('face-restore', _imageprocessors.image_processor_names())

    def test_fidelity_out_of_range(self):
        with self.assertRaises(_exceptions.ImageProcessorArgumentError):
            _imageprocessors.create_image_processor(
                'face-restore;model=missing.pth;fidelity=1.5',
                device='cpu')

    def test_missing_facexlib_message(self):
        try:
            import facexlib  # noqa: F401
        except ImportError:
            with self.assertRaises(_exceptions.ImageProcessorArgumentError) as ctx:
                _imageprocessors.create_image_processor(
                    'face-restore;model=missing.pth',
                    device='cpu')
            self.assertIn('facexlib', str(ctx.exception).lower())
            return
        self.skipTest('facexlib is installed; ImportError path not exercised')


if __name__ == '__main__':
    unittest.main()
