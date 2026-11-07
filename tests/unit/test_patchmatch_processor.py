import os
import tempfile
import unittest
import unittest.mock

import PIL.Image
import numpy as np

import dgenerate.imageprocessors as _imageprocessors
import dgenerate.imageprocessors.exceptions as _exceptions


def _processor(uri):
    return _imageprocessors.create_image_processor(uri, device='cpu')


def _mask_file(mask: PIL.Image.Image) -> str:
    descriptor, path = tempfile.mkstemp(suffix='.png')
    os.close(descriptor)
    mask.save(path)
    return path


class TestPatchMatchProcessor(unittest.TestCase):

    def test_algorithm_options_are_in_the_schema(self):
        schema = _imageprocessors.ImageProcessorLoader().get_accepted_args_schema()
        self.assertEqual(
            schema['patchmatch']['algorithm']['options'],
            ['barnes', 'phil'])

    def test_unknown_algorithm(self):
        with self.assertRaises(_exceptions.ImageProcessorArgumentError) as ctx:
            _processor('patchmatch;mask=mask.png;algorithm=patchmatch')
        message = str(ctx.exception)
        self.assertIn('barnes', message)
        self.assertIn('phil', message)

    def test_algorithm_is_case_insensitive(self):
        processor = _processor('patchmatch;mask=mask.png;algorithm=Phil')
        self.assertEqual(processor._algorithm, 'phil')

    def test_patchmatch_rejects_even_patch_size(self):
        with self.assertRaises(_exceptions.ImageProcessorArgumentError) as ctx:
            _processor('patchmatch;mask=mask.png;patch-size=4')
        self.assertIn('odd', str(ctx.exception))

    def test_philippeau_accepts_even_patch_size(self):
        processor = _processor('patchmatch;mask=mask.png;algorithm=phil;patch-size=4')
        self.assertEqual(processor._patch_size, 4)

    def test_philippeau_rejects_patchmatch_only_options(self):
        class UnsupportedOption(Exception):
            def __init__(self, algorithm, options):
                self.algorithm = algorithm
                self.options = tuple(options)

        def inpaint(*_args, **kwargs):
            rejected = [
                name for name in kwargs
                if name not in ('patch_size', 'seed', 'algorithm')
            ]
            raise UnsupportedOption(kwargs['algorithm'], rejected)

        image = PIL.Image.new('RGB', (8, 8), (1, 2, 3))
        mask = PIL.Image.new('L', (8, 8), 0)
        mask_path = _mask_file(mask)
        try:
            with unittest.mock.patch(
                    'dgenerate.imageprocessors.patchmatch.patchmatch_cython.UnsupportedOption',
                    UnsupportedOption,
                    create=True), \
                    unittest.mock.patch(
                        'dgenerate.imageprocessors.patchmatch.patchmatch_cython.inpaint_pyramid',
                        side_effect=inpaint):
                processor = _processor(
                    f'patchmatch;mask={mask_path};algorithm=phil;'
                    'edge-weighted=true;em-loops=2'
                )
                with self.assertRaises(_exceptions.ImageProcessorArgumentError) as ctx:
                    processor.process(image)
            message = str(ctx.exception)
            self.assertIn('"edge-weighted"', message)
            self.assertIn('"em-loops"', message)
            self.assertIn('algorithm=phil', message)
        finally:
            image.close()
            mask.close()
            os.unlink(mask_path)

    def test_philippeau_accepts_patchmatch_defaults(self):
        processor = _processor(
            'patchmatch;mask=mask.png;algorithm=phil;'
            'em-loops=1;poisson-blend=true;hole-refine=0;mask-feather=0'
        )
        self.assertEqual(processor._algorithm, 'phil')
        self.assertEqual(processor._seed, 0)

    def test_inpaint_call_uses_selected_algorithm(self):
        image = PIL.Image.new('RGB', (32, 32), (10, 20, 30))
        mask = PIL.Image.new('L', (32, 32), 0)
        filled = np.zeros((32, 32, 3), dtype=np.uint8)

        mask_path = _mask_file(mask)
        try:
            with unittest.mock.patch(
                    'dgenerate.imageprocessors.patchmatch.patchmatch_cython.inpaint_pyramid',
                    return_value=filled) as inpaint:
                processor = _processor(
                    f'patchmatch;mask={mask_path};algorithm=phil;patch-size=4;seed=7'
                )
                result = processor.process(image)

            self.assertEqual(result.mode, 'RGB')
            self.assertEqual(result.size, (32, 32))
            args, kwargs = inpaint.call_args
            self.assertEqual(args[0].shape, (32, 32, 3))
            self.assertEqual(args[1].shape, (32, 32))
            self.assertTrue(args[1].dtype == np.bool_)
            self.assertEqual(kwargs, {
                'patch_size': 4,
                'seed': 7,
                'algorithm': 'phil',
            })
        finally:
            image.close()
            mask.close()
            os.unlink(mask_path)

    def test_patchmatch_forwards_quality_options(self):
        image = PIL.Image.new('RGB', (16, 16), (40, 40, 40))
        mask = PIL.Image.new('L', (16, 16), 255)
        filled = np.zeros((16, 16), dtype=np.uint8)

        mask_path = _mask_file(mask)
        try:
            with unittest.mock.patch(
                    'dgenerate.imageprocessors.patchmatch.patchmatch_cython.inpaint_pyramid',
                    return_value=filled) as inpaint:
                processor = _processor(
                    f'patchmatch;mask={mask_path};seed=3;em-loops=2;'
                    'edge-weighted=true;reverse-search=true;poisson-blend=false;'
                    'hole-refine=1;mask-feather=2.5'
                )
                processor.process(image)

            self.assertEqual(inpaint.call_args.kwargs, {
                'patch_size': 5,
                'seed': 3,
                'algorithm': 'barnes',
                'em_loops': 2,
                'edge_weighted': True,
                'reverse_search': True,
                'poisson_blend': False,
                'hole_refine': 1,
                'mask_feather': 2.5,
            })
        finally:
            image.close()
            mask.close()
            os.unlink(mask_path)

    def test_philippeau_converts_subject_to_rgb(self):
        image = PIL.Image.new('L', (8, 8), 12)
        mask = PIL.Image.new('L', (8, 8), 0)
        filled = np.zeros((8, 8, 3), dtype=np.uint8)

        mask_path = _mask_file(mask)
        try:
            with unittest.mock.patch(
                    'dgenerate.imageprocessors.patchmatch.patchmatch_cython.inpaint_pyramid',
                    return_value=filled) as inpaint:
                processor = _processor(
                    f'patchmatch;mask={mask_path};algorithm=phil'
                )
                processor._process(image)

            passed = inpaint.call_args.args[0]
            self.assertEqual(passed.shape, (8, 8, 3))
            self.assertEqual(inpaint.call_args.kwargs['seed'], 0)
            self.assertNotIn('em_loops', inpaint.call_args.kwargs)
        finally:
            image.close()
            mask.close()
            os.unlink(mask_path)


if __name__ == '__main__':
    unittest.main()
