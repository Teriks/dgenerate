import unittest

import dgenerate.mediainput as _mi


class TestImageSeedParser(unittest.TestCase):
    def test_file_not_found(self):
        with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
            _mi.parse_image_seed_uri('not_found')
        self.assertIn('not_found', str(e.exception))

        with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
            _mi.parse_image_seed_uri('examples/media/earth.jpg;not_found')
        self.assertIn('not_found', str(e.exception))

        with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
            _mi.parse_image_seed_uri('examples/media/earth.jpg;mask=not_found')
        self.assertIn('not_found', str(e.exception))

        with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
            _mi.parse_image_seed_uri('examples/media/earth.jpg;control=not_found')
        self.assertIn('not_found', str(e.exception))

        with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
            _mi.parse_image_seed_uri('examples/media/earth.jpg;floyd=not_found')
        self.assertIn('not_found', str(e.exception))

    def test_legacy_arguments(self):
        with self.assertRaises(_mi.ImageSeedArgumentError):
            # not aligned by 8
            _mi.parse_image_seed_uri('examples/media/earth.jpg;14')

        # fine
        d = _mi.parse_image_seed_uri('examples/media/earth.jpg;14', align=2).resize_resolution
        self.assertEqual(d, (14, 14))

        with self.assertRaises(_mi.ImageSeedArgumentError):
            # defined the resolution multiple times
            _mi.parse_image_seed_uri('examples/media/earth.jpg;512x512;512x512')

        parsed = _mi.parse_image_seed_uri(
            'examples/media/dog-on-bench.png')

        self.assertEqual(parsed.seed_path, 'examples/media/dog-on-bench.png')
        self.assertIsNone(parsed.mask_path)
        self.assertIsNone(parsed.resize_resolution)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/dog-on-bench.png;examples/media/dog-on-bench-mask.png')

        self.assertEqual(parsed.seed_path, 'examples/media/dog-on-bench.png')
        self.assertEqual(parsed.mask_path, 'examples/media/dog-on-bench-mask.png')
        self.assertIsNone(parsed.resize_resolution)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/dog-on-bench.png;512x512;examples/media/dog-on-bench-mask.png')

        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertEqual(parsed.seed_path, 'examples/media/dog-on-bench.png')
        self.assertEqual(parsed.mask_path, 'examples/media/dog-on-bench-mask.png')

        parsed = _mi.parse_image_seed_uri('examples/media/dog-on-bench.png;examples/media/dog-on-bench-mask.png;1024')

        self.assertEqual(parsed.resize_resolution, (1024, 1024))
        self.assertEqual(parsed.seed_path, 'examples/media/dog-on-bench.png')
        self.assertEqual(parsed.mask_path, 'examples/media/dog-on-bench-mask.png')

    def test_uri_arguments(self):
        with self.assertRaises(_mi.ImageSeedArgumentError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=garbage')

        with self.assertRaises(_mi.ImageSeedArgumentError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;aspect=garbage')

        with self.assertRaises(_mi.ImageSeedArgumentError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;frame-start=garbage')

        with self.assertRaises(_mi.ImageSeedArgumentError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;frame-end=garbage')

        with self.assertRaises(_mi.ImageSeedArgumentError):
            # mutually exclusive arguments
            _mi.parse_image_seed_uri(
                'examples/media/earth.jpg;floyd=examples/media/earth.jpg;control=examples/media/earth.jpg')

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'control=examples/media/horse2.jpeg;resize=512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.seed_path, 'examples/media/earth.jpg')
        self.assertEqual(parsed.mask_path, 'examples/media/horse1.jpg')
        self.assertEqual(parsed.control_path, 'examples/media/horse2.jpeg')
        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'control=examples/media/horse2.jpeg, "examples/media/beach.jpg";'
            'resize=1024x512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.seed_path, 'examples/media/earth.jpg')
        self.assertEqual(parsed.mask_path, 'examples/media/horse1.jpg')
        self.assertSequenceEqual(parsed.control_path, ['examples/media/horse2.jpeg', 'examples/media/beach.jpg'])
        self.assertEqual(parsed.resize_resolution, (1024, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'floyd=examples/media/horse2.jpeg;resize=512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.seed_path, 'examples/media/earth.jpg')
        self.assertEqual(parsed.mask_path, 'examples/media/horse1.jpg')
        self.assertEqual(parsed.floyd_path, 'examples/media/horse2.jpeg')
        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'floyd=examples/media/horse2.jpeg;resize=1024x512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.seed_path, 'examples/media/earth.jpg')
        self.assertEqual(parsed.mask_path, 'examples/media/horse1.jpg')
        self.assertEqual(parsed.floyd_path, 'examples/media/horse2.jpeg')
        self.assertEqual(parsed.resize_resolution, (1024, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        with self.assertRaises(_mi.ImageSeedArgumentError):
            # not aligned by 8, the default requirement
            _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=14')

        with self.assertRaises(_mi.ImageSeedArgumentError):
            # not aligned by 8, the default requirement
            _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=3')

        # fine
        d = _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=14', align=1).resize_resolution
        self.assertEqual(d, (14, 14))

        # fine
        d = _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=14', align=2).resize_resolution
        self.assertEqual(d, (14, 14))

        # fine
        d = _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=3', align=1).resize_resolution
        self.assertEqual(d, (3, 3))

        # fine
        d = _mi.parse_image_seed_uri('examples/media/earth.jpg;resize=3', align=3).resize_resolution
        self.assertEqual(d, (3, 3))

    def test_uri_syntax_errors(self):
        with self.assertRaises(_mi.ImageSeedParseError):
            # stray colon
            _mi.parse_image_seed_uri('examples/media/earth.jpg;')

        with self.assertRaises(_mi.ImageSeedParseError):
            # stray colon
            _mi.parse_image_seed_uri('examples/media/earth.jpg;examples/media/earth.jpg;')

        with self.assertRaises(_mi.ImageSeedParseError):
            # stray colon
            _mi.parse_image_seed_uri('examples/media/earth.jpg;;examples/media/earth.jpg')

        with self.assertRaises(_mi.ImageSeedParseError):
            # unterminated string
            _mi.parse_image_seed_uri('"examples/media/earth.jpg;')

        with self.assertRaises(_mi.ImageSeedParseError):
            # unterminated string
            _mi.parse_image_seed_uri('examples/media/earth.jpg"')

        with self.assertRaises(_mi.ImageSeedParseError):
            # unterminated string
            _mi.parse_image_seed_uri("'examples/media/earth.jpg;")

        with self.assertRaises(_mi.ImageSeedParseError):
            # unterminated string
            _mi.parse_image_seed_uri("examples/media/earth.jpg'")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control= ")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=examples/media/earth.jpg,")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=,examples/media/earth.jpg")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=examples/media/earth.jpg,,examples/media/earth.jpg")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=,")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;mask= ")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;mask=")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;floyd= ")


if __name__ == '__main__':
    unittest.main()

