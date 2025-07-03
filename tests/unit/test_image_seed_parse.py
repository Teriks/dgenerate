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

    def test_mask_quote_handling(self):
        """Test quote handling for mask images in legacy and modern syntax"""
        
        # Test single mask with double quotes (legacy syntax)
        parsed = _mi.parse_image_seed_uri('examples/media/earth.jpg;"examples/media/dog-on-bench-mask.png"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])  # Quotes should be removed
        
        # Test single mask with single quotes (legacy syntax)
        parsed = _mi.parse_image_seed_uri("examples/media/earth.jpg;'examples/media/dog-on-bench-mask.png'")
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])  # Quotes should be removed
        
        # Test multiple masks with quotes (legacy syntax)
        parsed = _mi.parse_image_seed_uri(
            'images: examples/media/earth.jpg, examples/media/beach.jpg;'
            '"examples/media/dog-on-bench-mask.png", "examples/media/horse1.jpg"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png', 'examples/media/horse1.jpg'])
        
        # Test mask with keyword argument and quotes
        parsed = _mi.parse_image_seed_uri('examples/media/earth.jpg;mask="examples/media/dog-on-bench-mask.png"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        
        # Test multiple masks with keyword argument
        parsed = _mi.parse_image_seed_uri(
            'images: examples/media/earth.jpg, examples/media/beach.jpg;'
            'mask="examples/media/dog-on-bench-mask.png", "examples/media/horse1.jpg"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png', 'examples/media/horse1.jpg'])
    
    def test_mask_url_quote_handling(self):
        """Test quote handling for URLs as mask images"""
        
        # Use HTTP URLs instead of file:// to avoid file existence checks
        url1 = "http://example.com/mask1.png"
        url2 = "https://example.com/mask2.png"
        
        # Test single URL mask with quotes (legacy syntax)
        parsed = _mi.parse_image_seed_uri(f'examples/media/earth.jpg;"{url1}"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, [url1])  # Quotes should be removed
        
        # Test multiple URL masks
        parsed = _mi.parse_image_seed_uri(
            f'images: examples/media/earth.jpg, examples/media/beach.jpg;'
            f'"{url1}", "{url2}"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg', 'examples/media/beach.jpg'])
        self.assertEqual(parsed.mask_images, [url1, url2])
    
    def test_mixed_quote_styles(self):
        """Test mixing quote styles in image seed URIs"""
        
        # Test double quotes for image, single quotes for mask
        parsed = _mi.parse_image_seed_uri('"examples/media/earth.jpg";\'examples/media/dog-on-bench-mask.png\'')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        
        # Test with escape sequences in quotes
        parsed = _mi.parse_image_seed_uri(r'"examples/media/earth.jpg";"examples/media/dog-on-bench-mask.png"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
    
    def test_empty_and_edge_case_quotes(self):
        """Test edge cases with quotes"""
        
        # Test empty quotes should fail
        with self.assertRaises(_mi.ImageSeedFileNotFoundError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;""')
        
        with self.assertRaises(_mi.ImageSeedFileNotFoundError):
            _mi.parse_image_seed_uri("examples/media/earth.jpg;''")
        
        # Test mismatched quotes (should be treated as unterminated string)
        with self.assertRaises(_mi.ImageSeedParseError):
            _mi.parse_image_seed_uri('examples/media/earth.jpg;"examples/media/mask.png\'')
    
    def test_dimension_after_quoted_mask(self):
        """Test dimension specification after quoted mask"""
        
        # Legacy syntax with quoted mask followed by dimensions
        parsed = _mi.parse_image_seed_uri('examples/media/earth.jpg;"examples/media/dog-on-bench-mask.png";512x512')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        self.assertEqual(parsed.resize_resolution, (512, 512))
        
        # Modern syntax with resize keyword
        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;mask="examples/media/dog-on-bench-mask.png";resize=512x512')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        self.assertEqual(parsed.resize_resolution, (512, 512))
    
    def test_control_image_quote_handling(self):
        """Test quote handling for control images"""
        
        # Test single control image with quotes
        parsed = _mi.parse_image_seed_uri('examples/media/earth.jpg;control="examples/media/horse2.jpeg"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.control_images, ['examples/media/horse2.jpeg'])
        
        # Test multiple control images with quotes
        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;control="examples/media/horse2.jpeg", "examples/media/beach.jpg"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.control_images, ['examples/media/horse2.jpeg', 'examples/media/beach.jpg'])
        
        # Test control images in single spec mode
        parsed = _mi.parse_image_seed_uri('"examples/media/horse2.jpeg", "examples/media/beach.jpg"')
        self.assertEqual(parsed.images, ['examples/media/horse2.jpeg', 'examples/media/beach.jpg'])
        self.assertTrue(parsed.is_single_spec)  # Should be ambiguous single spec
        self.assertIsNone(parsed.control_images)  # Control images not set in single spec mode
    
    def test_floyd_image_quote_handling(self):
        """Test quote handling for floyd images"""
        
        parsed = _mi.parse_image_seed_uri('examples/media/earth.jpg;floyd="examples/media/mountain.png"')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.floyd_image, 'examples/media/mountain.png')
    
    def test_adapter_image_quote_handling(self):
        """Test quote handling for adapter images"""
        
        # Already tested in test_adapter_quoting, but let's add a few more cases
        
        # Test adapter with pipe arguments and quotes
        parsed = _mi.parse_image_seed_uri(
            'examples/media/earth.jpg;adapter="examples/media/mountain.png"|resize=512|aspect=false')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.adapter_images[0][0].path, 'examples/media/mountain.png')
        self.assertEqual(parsed.adapter_images[0][0].resize, (512, 512))
        self.assertFalse(parsed.adapter_images[0][0].aspect)
    
    def test_complex_quote_scenarios(self):
        """Test complex scenarios with multiple quoted components"""
        
        # Test all components quoted
        parsed = _mi.parse_image_seed_uri(
            '"examples/media/earth.jpg";mask="examples/media/dog-on-bench-mask.png";'
            'control="examples/media/horse2.jpeg";resize=512x512')
        self.assertEqual(parsed.images, ['examples/media/earth.jpg'])
        self.assertEqual(parsed.mask_images, ['examples/media/dog-on-bench-mask.png'])
        self.assertEqual(parsed.control_images, ['examples/media/horse2.jpeg'])
        self.assertEqual(parsed.resize_resolution, (512, 512))
        
        # For tests that need multiple non-existent files, use HTTP URLs
        parsed = _mi.parse_image_seed_uri(
            'images: "http://example.com/img1.jpg", "http://example.com/img2.jpg";'
            'mask="http://example.com/mask1.png", "http://example.com/mask2.png";'
            'control="http://example.com/control.png"')
        self.assertEqual(parsed.images, ['http://example.com/img1.jpg', 'http://example.com/img2.jpg'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask1.png', 'http://example.com/mask2.png'])
        self.assertEqual(parsed.control_images, ['http://example.com/control.png'])
    
    def test_quote_handling_with_temp_files(self):
        """Test quote handling with URLs and simple paths"""
        
        # Test legacy syntax with quoted URLs
        parsed = _mi.parse_image_seed_uri('"http://example.com/image.png";"http://example.com/mask.png"')
        self.assertEqual(parsed.images, ['http://example.com/image.png'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask.png'])
        
        # Test with single quotes
        parsed = _mi.parse_image_seed_uri("'http://example.com/image.png';'http://example.com/mask.png'")
        self.assertEqual(parsed.images, ['http://example.com/image.png'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask.png'])
        
        # Test modern syntax with quotes
        parsed = _mi.parse_image_seed_uri(
            '"http://example.com/image.png";mask="http://example.com/mask.png";control="http://example.com/control.png"')
        self.assertEqual(parsed.images, ['http://example.com/image.png'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask.png'])
        self.assertEqual(parsed.control_images, ['http://example.com/control.png'])
        
        # Test without quotes (should work for URLs)
        parsed = _mi.parse_image_seed_uri('http://example.com/image.png;http://example.com/mask.png')
        self.assertEqual(parsed.images, ['http://example.com/image.png'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask.png'])

    def test_quote_stripping_edge_cases(self):
        """Test edge cases of quote stripping behavior"""
        
        # Test that quotes are properly stripped
        parsed = _mi.parse_image_seed_uri(r'http://example.com/img.jpg;"http://example.com/mask.png"')
        self.assertEqual(parsed.images, ['http://example.com/img.jpg'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask.png'])  # Quotes stripped
        
        # Test mixed quoting in multi-mask scenario
        parsed = _mi.parse_image_seed_uri(
            'images: http://example.com/img1.jpg, http://example.com/img2.jpg;'
            '"http://example.com/mask1.png", http://example.com/mask2.png')
        self.assertEqual(parsed.images, ['http://example.com/img1.jpg', 'http://example.com/img2.jpg'])
        self.assertEqual(parsed.mask_images, ['http://example.com/mask1.png', 'http://example.com/mask2.png'])

    @unittest.mock.patch('os.path.exists')
    def test_windows_paths_with_quotes(self, mock_exists):
        """Test quote handling with Windows-style paths"""

        mock_exists.return_value = True

        img_path = r"C:\\hello world\\test image.png"
        mask_path = r"C:\\hello world\\test mask.png"
        control_path = r"C:\\hello world\\test control.png"

        img_path_parsed = r"C:\hello world\test image.png"
        mask_path_parsed = r"C:\hello world\test mask.png"
        control_path_parsed = r"C:\hello world\test control.png"


        # Test legacy syntax with quoted paths (using escaped backslashes)
        parsed = _mi.parse_image_seed_uri(f'"{img_path}";"{mask_path}"')
        self.assertEqual(parsed.images, [img_path_parsed])
        self.assertEqual(parsed.mask_images, [mask_path_parsed])

        # Test with single quotes
        parsed = _mi.parse_image_seed_uri(f"'{img_path}';'{mask_path}'")
        self.assertEqual(parsed.images, [img_path_parsed])
        self.assertEqual(parsed.mask_images, [mask_path_parsed])

        # Test modern syntax
        parsed = _mi.parse_image_seed_uri(
            f'"{img_path}";mask="{mask_path}";control="{control_path}"')
        self.assertEqual(parsed.images, [img_path_parsed])
        self.assertEqual(parsed.mask_images, [mask_path_parsed])
        self.assertEqual(parsed.control_images, [control_path_parsed])

    @unittest.mock.patch('os.path.exists')
    def test_quote_handling_edge_cases_with_mock(self, mock_exists):
        """Test various quote handling edge cases without file system dependencies"""
        
        mock_exists.return_value = True

        # Test mixed quoting styles
        parsed = _mi.parse_image_seed_uri('"image.png";\'mask.png\'')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.mask_images, ['mask.png'])
        
        # Test empty quotes
        mock_exists.return_value = False  # Empty string won't exist
        with self.assertRaises(_mi.ImageSeedFileNotFoundError):
            _mi.parse_image_seed_uri('image.png;""')

        # Test unquoted path with spaces should fail
        with self.assertRaises(_mi.ImageSeedFileNotFoundError):
            # Spaces in unquoted path will cause tokenization issues
            _mi.parse_image_seed_uri('C:\\hello world\\image.png;C:\\hello world\\mask.png')

        # Test quotes with special characters
        mock_exists.return_value = True
        special_path = r"C:\\test\\file (1) [2] {3}.png"
        special_path_parsed = r"C:\test\file (1) [2] {3}.png"
        
        parsed = _mi.parse_image_seed_uri(f'"{special_path}";"{special_path}"')
        self.assertEqual(parsed.images, [special_path_parsed])
        self.assertEqual(parsed.mask_images, [special_path_parsed])
    
    @unittest.mock.patch('os.path.exists')
    def test_escape_sequences_in_quotes(self, mock_exists):
        """Test that escape sequences are properly handled in quoted strings"""
        
        mock_exists.return_value = True
        
        # Test newline escape
        path_with_newline = r"C:\\test\\file\nname.png"
        path_with_newline_parsed = "C:\\test\\file\nname.png"  # \n should be converted to actual newline
        
        parsed = _mi.parse_image_seed_uri(f'"{path_with_newline}"')
        self.assertEqual(parsed.images, [path_with_newline_parsed])
        
        # Test tab escape
        path_with_tab = r"C:\\test\\file\tname.png"
        path_with_tab_parsed = "C:\\test\\file\tname.png"  # \t should be converted to actual tab
        
        parsed = _mi.parse_image_seed_uri(f'image.png;"{path_with_tab}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.mask_images, [path_with_tab_parsed])
        
        # Test escaped quote within quotes
        path_with_quote = r"C:\\test\\file\"name.png"
        path_with_quote_parsed = r'C:\test\file"name.png'
        
        parsed = _mi.parse_image_seed_uri(f'"{path_with_quote}"')
        self.assertEqual(parsed.images, [path_with_quote_parsed])
    
    @unittest.mock.patch('os.path.exists')
    def test_multi_image_quote_handling(self, mock_exists):
        """Test quote handling with multiple images syntax"""
        
        mock_exists.return_value = True
        
        # Test images: syntax with quoted Windows paths
        img1 = r"C:\\folder\\image1.png"
        img2 = r"C:\\folder\\image2.png"
        mask1 = r"C:\\folder\\mask1.png"
        mask2 = r"C:\\folder\\mask2.png"
        
        img1_parsed = r"C:\folder\image1.png"
        img2_parsed = r"C:\folder\image2.png"
        mask1_parsed = r"C:\folder\mask1.png"
        mask2_parsed = r"C:\folder\mask2.png"
        
        # Test with mixed quotes
        parsed = _mi.parse_image_seed_uri(
            f'images: "{img1}", \'{img2}\';"{mask1}", "{mask2}"')
        self.assertEqual(parsed.images, [img1_parsed, img2_parsed])
        self.assertEqual(parsed.mask_images, [mask1_parsed, mask2_parsed])
        
        # Test single mask for multiple images
        parsed = _mi.parse_image_seed_uri(
            f'images: "{img1}", "{img2}";"{mask1}"')
        self.assertEqual(parsed.images, [img1_parsed, img2_parsed])
        # Single mask should be duplicated
        self.assertEqual(parsed.mask_images, [mask1_parsed, mask1_parsed])
    
    @unittest.mock.patch('os.path.exists') 
    def test_network_paths_and_unc(self, mock_exists):
        """Test handling of UNC paths and network paths"""
        
        mock_exists.return_value = True
        
        # Test UNC path
        unc_path = r"\\\\server\\share\\folder\\image.png"
        unc_path_parsed = r"\\server\share\folder\image.png"
        
        parsed = _mi.parse_image_seed_uri(f'"{unc_path}"')
        self.assertEqual(parsed.images, [unc_path_parsed])
        
        # Test UNC with mask
        unc_mask = r"\\\\server\\share\\folder\\mask.png"
        unc_mask_parsed = r"\\server\share\folder\mask.png"
        
        parsed = _mi.parse_image_seed_uri(f'"{unc_path}";"{unc_mask}"')
        self.assertEqual(parsed.images, [unc_path_parsed])
        self.assertEqual(parsed.mask_images, [unc_mask_parsed])
    
    @unittest.mock.patch('os.path.exists')
    def test_keyword_argument_escape_sequences(self, mock_exists):
        """Test escape sequence expansion in keyword arguments"""
        
        mock_exists.return_value = True
        
        # Test mask keyword with Windows path
        win_path = r"C:\\Users\\test\\image.png"
        win_mask = r"C:\\Users\\test\\mask.png"
        win_control = r"C:\\Users\\test\\control.png"
        
        win_path_parsed = r"C:\Users\test\image.png"
        win_mask_parsed = r"C:\Users\test\mask.png"
        win_control_parsed = r"C:\Users\test\control.png"
        
        # Test mask keyword with double quotes
        parsed = _mi.parse_image_seed_uri(f'"{win_path}";mask="{win_mask}"')
        self.assertEqual(parsed.images, [win_path_parsed])
        self.assertEqual(parsed.mask_images, [win_mask_parsed])
        
        # Test mask keyword with single quotes  
        parsed = _mi.parse_image_seed_uri(f"'{win_path}';mask='{win_mask}'")
        self.assertEqual(parsed.images, [win_path_parsed])
        self.assertEqual(parsed.mask_images, [win_mask_parsed])
        
        # Test multiple keywords with escape sequences
        parsed = _mi.parse_image_seed_uri(
            f'"{win_path}";mask="{win_mask}";control="{win_control}"')
        self.assertEqual(parsed.images, [win_path_parsed])
        self.assertEqual(parsed.mask_images, [win_mask_parsed])
        self.assertEqual(parsed.control_images, [win_control_parsed])
        
        # Test floyd keyword
        floyd_path = r"C:\\floyd\\stage1.png"
        floyd_path_parsed = r"C:\floyd\stage1.png"
        
        parsed = _mi.parse_image_seed_uri(f'"{win_path}";floyd="{floyd_path}"')
        self.assertEqual(parsed.images, [win_path_parsed])
        self.assertEqual(parsed.floyd_image, floyd_path_parsed)
    
    @unittest.mock.patch('os.path.exists')
    def test_special_escape_sequences_in_keywords(self, mock_exists):
        """Test special escape sequences like newline, tab in keyword arguments"""
        
        mock_exists.return_value = True
        
        # Test newline in mask path
        path_with_newline = r"image\nname.png"
        path_with_newline_parsed = "image\nname.png"  # Actual newline
        
        parsed = _mi.parse_image_seed_uri(f'image.png;mask="{path_with_newline}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.mask_images, [path_with_newline_parsed])
        
        # Test tab in control path
        path_with_tab = r"image\tname.png" 
        path_with_tab_parsed = "image\tname.png"  # Actual tab
        
        parsed = _mi.parse_image_seed_uri(f'image.png;control="{path_with_tab}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.control_images, [path_with_tab_parsed])
        
        # Test escaped quotes in paths
        path_with_quote = r"image\"name.png"
        path_with_quote_parsed = 'image"name.png'
        
        parsed = _mi.parse_image_seed_uri(f'image.png;mask="{path_with_quote}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.mask_images, [path_with_quote_parsed])
    
    @unittest.mock.patch('os.path.exists')
    def test_keyword_lists_with_escape_sequences(self, mock_exists):
        """Test escape sequences in comma-separated list keyword arguments"""
        
        mock_exists.return_value = True
        
        # Test control list with Windows paths
        control1 = r"C:\\control\\image1.png"
        control2 = r"C:\\control\\image2.png"
        control1_parsed = r"C:\control\image1.png"
        control2_parsed = r"C:\control\image2.png"
        
        parsed = _mi.parse_image_seed_uri(
            f'image.png;control="{control1}", "{control2}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.control_images, [control1_parsed, control2_parsed])
        
        # Test mask list with escape sequences
        mask1 = r"mask\n1.png"
        mask2 = r"mask\t2.png"
        mask1_parsed = "mask\n1.png"  # Actual newline
        mask2_parsed = "mask\t2.png"  # Actual tab
        
        parsed = _mi.parse_image_seed_uri(
            f'images: img1.png, img2.png;mask="{mask1}", "{mask2}"')
        self.assertEqual(parsed.images, ['img1.png', 'img2.png'])
        self.assertEqual(parsed.mask_images, [mask1_parsed, mask2_parsed])
    
    @unittest.mock.patch('os.path.exists')
    def test_adapter_keyword_escape_sequences(self, mock_exists):
        """Test escape sequences in adapter keyword arguments"""
        
        mock_exists.return_value = True
        
        # Test adapter with Windows path
        adapter_path = r"C:\\adapters\\style.png"
        adapter_path_parsed = r"C:\adapters\style.png"
        
        parsed = _mi.parse_image_seed_uri(f'image.png;adapter="{adapter_path}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.adapter_images[0][0].path, adapter_path_parsed)
        
        # Test adapter with pipe arguments containing escape sequences
        adapter_with_args = r"C:\\adapters\\style.png"
        parsed = _mi.parse_image_seed_uri(
            f'image.png;adapter="{adapter_with_args}"|resize=512|aspect=false')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.adapter_images[0][0].path, adapter_path_parsed)
        self.assertEqual(parsed.adapter_images[0][0].resize, (512, 512))
        self.assertFalse(parsed.adapter_images[0][0].aspect)
    
    @unittest.mock.patch('os.path.exists')
    def test_latents_keyword_escape_sequences(self, mock_exists):
        """Test escape sequences in latents keyword arguments"""
        
        mock_exists.return_value = True
        
        # Test latents with Windows path
        latent_path = r"C:\\models\\latent.pt"
        latent_path_parsed = r"C:\models\latent.pt"
        
        parsed = _mi.parse_image_seed_uri(f'image.png;latents="{latent_path}"')
        self.assertEqual(parsed.images, ['image.png'])
        self.assertEqual(parsed.latents, [latent_path_parsed])
        
        # Test multiple latents with escape sequences
        latent1 = r"C:\\models\\latent1.pt"
        latent2 = r"C:\\models\\latent2.safetensors"
        latent1_parsed = r"C:\models\latent1.pt"
        latent2_parsed = r"C:\models\latent2.safetensors"
        
        parsed = _mi.parse_image_seed_uri(
            f'images: img1.png, img2.png;latents="{latent1}", "{latent2}"')
        self.assertEqual(parsed.images, ['img1.png', 'img2.png'])
        self.assertEqual(parsed.latents, [latent1_parsed, latent2_parsed])


if __name__ == '__main__':
    unittest.main()
