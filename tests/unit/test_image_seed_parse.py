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

    def test_adapter_quoting(self):
        seed = _mi.parse_image_seed_uri(
            'images: "examples/media/earth.jpg", "examples/media/beach.jpg"; '
            'adapter=examples/media/earth.jpg + "examples/media/earth.jpg"|resize=512')

        self.assertEqual(seed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
        self.assertEqual(seed.adapter_images[0][0].path, 'examples/media/earth.jpg')
        self.assertEqual(seed.adapter_images[0][1].resize, (512, 512))

        seed = _mi.parse_image_seed_uri('adapter: examples/media/earth.jpg + "examples/media/earth.jpg"|resize=512')

        self.assertEqual(seed.adapter_images[0][0].path, 'examples/media/earth.jpg')
        self.assertEqual(seed.adapter_images[0][1].resize, (512, 512))

        seed = _mi.parse_image_seed_uri(
            'images: "examples/media/earth.jpg", "examples/media/beach.jpg"; '
            'adapter=examples/media/earth.jpg, "examples/media/earth.jpg"|resize=512')

        self.assertEqual(seed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
        self.assertEqual(seed.adapter_images[0][0].path, 'examples/media/earth.jpg')
        self.assertEqual(seed.adapter_images[1][0].resize, (512, 512))

        seed = _mi.parse_image_seed_uri('adapter: examples/media/earth.jpg, "examples/media/earth.jpg"|resize=512')

        self.assertEqual(seed.adapter_images[0][0].path, 'examples/media/earth.jpg')
        self.assertEqual(seed.adapter_images[1][0].resize, (512, 512))

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

        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png'])
        self.assertIsNone(parsed.mask_images)
        self.assertIsNone(parsed.resize_resolution)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/dog-on-bench.png;examples/media/dog-on-bench-mask.png')

        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        self.assertIsNone(parsed.resize_resolution)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/dog-on-bench.png;512x512;examples/media/dog-on-bench-mask.png')

        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])

        parsed = _mi.parse_image_seed_uri('examples/media/dog-on-bench.png;examples/media/dog-on-bench-mask.png;1024')

        self.assertEqual(parsed.resize_resolution, (1024, 1024))
        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])

        parsed = _mi.parse_image_seed_uri(
            'images: examples/media/dog-on-bench.png, examples/media/earth.jpg;examples/media/dog-on-bench-mask.png;1024')

        self.assertEqual(parsed.resize_resolution, (1024, 1024))
        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png', 'examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images,
                         ['examples/media/dog-on-bench-mask.png', 'examples/media/dog-on-bench-mask.png'])

        parsed = _mi.parse_image_seed_uri(
            'images: examples/media/dog-on-bench.png, examples/media/earth.jpg;examples/media/dog-on-bench-mask.png, examples/media/dog-on-bench.png;512')

        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertEqual(parsed.images, ['examples/media/dog-on-bench.png', 'examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images,
                         ['examples/media/dog-on-bench-mask.png', 'examples/media/dog-on-bench.png'])

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

        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/horse1.jpg'])
        self.assertEqual(parsed.control_images, ['examples/media/horse2.jpeg'])
        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'control=examples/media/horse2.jpeg, "examples/media/beach.jpg";'
            'resize=1024x512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/horse1.jpg'])
        self.assertSequenceEqual(parsed.control_images, ['examples/media/horse2.jpeg', 'examples/media/beach.jpg'])
        self.assertEqual(parsed.resize_resolution, (1024, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'floyd=examples/media/horse2.jpeg;resize=512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/horse1.jpg'])
        self.assertEqual(parsed.floyd_image, 'examples/media/horse2.jpeg')
        self.assertEqual(parsed.resize_resolution, (512, 512))
        self.assertFalse(parsed.aspect_correct)
        self.assertEqual(parsed.frame_start, 5)
        self.assertEqual(parsed.frame_end, 10)

        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask=examples/media/horse1.jpg;'
            'floyd=examples/media/horse2.jpeg;resize=1024x512;aspect=false;frame-start=5;frame-end=10')

        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/horse1.jpg'])
        self.assertEqual(parsed.floyd_image, 'examples/media/horse2.jpeg')
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
            _mi.parse_image_seed_uri(
                "examples/media/earth.jpg;control=examples/media/earth.jpg,,examples/media/earth.jpg")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;control=,")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;mask= ")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;mask=")

        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;floyd= ")

    def test_latents_parsing(self):
        """Test the latents syntax parsing functionality"""
        import tempfile
        import torch
        import os

        # Create temporary tensor files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test tensor files
            latent1_path = os.path.join(temp_dir, "latent1.pt")
            latent2_path = os.path.join(temp_dir, "latent2.pt")
            latent3_path = os.path.join(temp_dir, "latent3.safetensors")
            
            # Create actual tensor files
            test_tensor1 = torch.randn(4, 64, 64)
            test_tensor2 = torch.randn(4, 32, 32)
            torch.save(test_tensor1, latent1_path)
            torch.save(test_tensor2, latent2_path)
            torch.save(test_tensor2, latent3_path)  # safetensors file with .pt content for simplicity

            # Test basic latents-only syntax
            parsed = _mi.parse_image_seed_uri(f'latents: {latent1_path}')
            self.assertIsNone(parsed.images)
            self.assertEqual(parsed.latents, [latent1_path])
            self.assertFalse(parsed.is_single_spec)
            self.assertIsNone(parsed.mask_images)
            self.assertIsNone(parsed.control_images)
            self.assertIsNone(parsed.adapter_images)

            # Test multiple latents
            parsed = _mi.parse_image_seed_uri(f'latents: {latent1_path}, {latent2_path}')
            self.assertIsNone(parsed.images)
            self.assertEqual(parsed.latents, [latent1_path, latent2_path])
            self.assertFalse(parsed.is_single_spec)

            # Test latents with keyword argument (single latent)
            parsed = _mi.parse_image_seed_uri(f'examples/media/earth.jpg;latents={latent1_path}')
            self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
            self.assertEqual(parsed.latents, [latent1_path])
            self.assertFalse(parsed.is_single_spec)

            # Test multiple images with multiple latents
            parsed = _mi.parse_image_seed_uri(
                f'images: examples/media/earth.jpg, examples/media/beach.jpg;latents={latent1_path}, {latent2_path}')
            self.assertEqual(parsed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
            self.assertEqual(parsed.latents, [latent1_path, latent2_path])
            self.assertTrue(parsed.multi_image_mode)
            self.assertFalse(parsed.is_single_spec)

            # Test latents with control images
            parsed = _mi.parse_image_seed_uri(f'latents: {latent1_path};control=examples/media/earth.jpg')
            self.assertIsNone(parsed.images)
            self.assertEqual(parsed.latents, [latent1_path])
            self.assertEqual(parsed.control_images, ['examples/media/earth.jpg'])

            # Test latents with mixed keyword arguments
            parsed = _mi.parse_image_seed_uri(
                f'examples/media/earth.jpg;mask=examples/media/dog-on-bench-mask.png;'
                f'latents={latent1_path};resize=512x512;aspect=false')
            self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
            self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
            self.assertEqual(parsed.latents, [latent1_path])
            self.assertEqual(parsed.resize_resolution, (512, 512))
            self.assertFalse(parsed.aspect_correct)

    def test_latents_error_cases(self):
        """Test error cases for latents parsing"""
        import tempfile
        import torch
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a real tensor file
            latent_path = os.path.join(temp_dir, "latent.pt")
            torch.save(torch.randn(4, 64, 64), latent_path)
            
            # Create a non-tensor file
            non_tensor_path = os.path.join(temp_dir, "not_tensor.txt")
            with open(non_tensor_path, 'w') as f:
                f.write("not a tensor")

            # Test non-existent latent file
            with self.assertRaises(_mi.ImageSeedFileNotFoundError) as e:
                _mi.parse_image_seed_uri('latents: nonexistent.pt')
            self.assertIn('nonexistent.pt', str(e.exception))

            # Test non-tensor file in latents syntax (legacy)
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'latents: {non_tensor_path}')
            self.assertIn('must be a tensor file', str(e.exception))

            # Test non-tensor file with latents keyword
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'examples/media/earth.jpg;latents={non_tensor_path}')
            self.assertIn('must be a tensor file', str(e.exception))

            # Test latents with floyd (should be incompatible)
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'latents: {latent_path};floyd=examples/media/earth.jpg')
            self.assertIn('latent tensors not supported with floyd', str(e.exception))

            # Test latents with floyd via keyword argument
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'examples/media/earth.jpg;latents={latent_path};floyd=examples/media/beach.jpg')
            self.assertIn('latent tensors not supported with floyd', str(e.exception))

            # Test empty latents argument
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri('examples/media/earth.jpg;latents=')
            self.assertIn('missing assignment value', str(e.exception))

            # Test stray comma in latents
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'examples/media/earth.jpg;latents={latent_path},')
            self.assertIn('Missing latent tensor definition', str(e.exception))

    def test_latents_with_resize_restrictions(self):
        """Test that latents-only mode rejects resize/mask arguments"""
        import tempfile
        import torch
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            latent_path = os.path.join(temp_dir, "latent.pt")
            torch.save(torch.randn(4, 64, 64), latent_path)

            # Test latents-only with resize (should fail in legacy mode)
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'latents: {latent_path};512x512')
            self.assertIn('Cannot use resize resolution', str(e.exception))

            # Test latents-only with mask (should fail in legacy mode)
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'latents: {latent_path};examples/media/dog-on-bench-mask.png')
            self.assertIn('Cannot use', str(e.exception))
            self.assertIn('inpaint mask', str(e.exception))

    def test_latents_file_extensions(self):
        """Test that various tensor file extensions are recognized"""
        import tempfile
        import torch
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different tensor file extensions
            for ext in ['.pt', '.pth', '.safetensors']:
                latent_path = os.path.join(temp_dir, f"latent{ext}")
                torch.save(torch.randn(4, 64, 64), latent_path)

                # Should parse successfully
                parsed = _mi.parse_image_seed_uri(f'latents: {latent_path}')
                self.assertEqual(parsed.latents, [latent_path])
                self.assertIsNone(parsed.images)

            # Test non-tensor extension should fail
            bad_path = os.path.join(temp_dir, "latent.jpg")
            with open(bad_path, 'w') as f:
                f.write("fake")
                
            with self.assertRaises(_mi.ImageSeedParseError) as e:
                _mi.parse_image_seed_uri(f'latents: {bad_path}')
            self.assertIn('must be a tensor file', str(e.exception))


if __name__ == '__main__':
    unittest.main()
