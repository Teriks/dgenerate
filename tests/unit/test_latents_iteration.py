import unittest
import tempfile

import safetensors.torch
import torch
import os

import dgenerate.mediainput as _mi


class TestLatentsIteration(unittest.TestCase):
    """Test iteration functionality with latents"""

    def setUp(self):
        """Set up temporary tensor files for testing"""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test tensor files
        self.latent1_path = os.path.join(self.temp_dir.name, "latent1.pt")
        self.latent2_path = os.path.join(self.temp_dir.name, "latent2.pt")
        self.latent3_path = os.path.join(self.temp_dir.name, "latent3.safetensors")
        
        # Create actual tensor data
        self.test_tensor1 = torch.randn(4, 64, 64)
        self.test_tensor2 = torch.randn(4, 32, 32)
        
        torch.save(self.test_tensor1, self.latent1_path)
        torch.save(self.test_tensor2, self.latent2_path)
        safetensors.torch.save_file({'samples': self.test_tensor2}, self.latent3_path)

    def tearDown(self):
        """Clean up temporary files"""
        self.temp_dir.cleanup()

    def test_iterate_latents_only(self):
        """Test iterating over latents-only image seed"""
        uri = f'latents: {self.latent1_path}'
        
        seeds = list(_mi.iterate_image_seed(uri))
        self.assertEqual(len(seeds), 1)
        
        seed = seeds[0]
        self.assertIsNone(seed.images)
        self.assertIsNotNone(seed.latents)
        self.assertEqual(len(seed.latents), 1)
        self.assertTrue(torch.is_tensor(seed.latents[0]))
        self.assertEqual(seed.latents[0].shape, self.test_tensor1.shape)
        self.assertFalse(seed.is_animation_frame)

    def test_iterate_multiple_latents(self):
        """Test iterating over multiple latents"""
        uri = f'latents: {self.latent1_path}, {self.latent2_path}'
        
        seeds = list(_mi.iterate_image_seed(uri))
        self.assertEqual(len(seeds), 1)
        
        seed = seeds[0]
        self.assertIsNone(seed.images)
        self.assertIsNotNone(seed.latents)
        self.assertEqual(len(seed.latents), 2)
        
        # Check that both tensors are loaded correctly
        self.assertTrue(torch.is_tensor(seed.latents[0]))
        self.assertTrue(torch.is_tensor(seed.latents[1]))
        self.assertEqual(seed.latents[0].shape, self.test_tensor1.shape)
        self.assertEqual(seed.latents[1].shape, self.test_tensor2.shape)

    def test_iterate_images_with_latents(self):
        """Test iterating over images combined with latents"""
        uri = f'examples/media/earth.jpg;latents={self.latent1_path}'
        
        seeds = list(_mi.iterate_image_seed(uri))
        self.assertEqual(len(seeds), 1)
        
        seed = seeds[0]
        self.assertIsNotNone(seed.images)
        self.assertIsNotNone(seed.latents)
        self.assertEqual(len(seed.images), 1)
        self.assertEqual(len(seed.latents), 1)
        
        # Check that image is loaded as PIL Image
        from PIL import Image
        self.assertIsInstance(seed.images[0], Image.Image)
        
        # Check that latent is loaded as tensor
        self.assertTrue(torch.is_tensor(seed.latents[0]))
        self.assertEqual(seed.latents[0].shape, self.test_tensor1.shape)

    def test_iterate_multiple_images_with_multiple_latents(self):
        """Test iterating over multiple images with multiple latents"""
        uri = (f'images: examples/media/earth.jpg, examples/media/earth.jpg;'
               f'latents={self.latent1_path}, {self.latent2_path}')
        
        seeds = list(_mi.iterate_image_seed(uri))
        self.assertEqual(len(seeds), 1)
        
        seed = seeds[0]
        self.assertIsNotNone(seed.images)
        self.assertIsNotNone(seed.latents)
        self.assertEqual(len(seed.images), 2)
        self.assertEqual(len(seed.latents), 2)
        
        # Check that all items are loaded correctly
        from PIL import Image
        for img in seed.images:
            self.assertIsInstance(img, Image.Image)
        
        for latent in seed.latents:
            self.assertTrue(torch.is_tensor(latent))

    def test_iterate_latents_with_control(self):
        """Test iterating over latents with control images"""
        uri = f'latents: {self.latent1_path};control=examples/media/earth.jpg'
        
        seeds = list(_mi.iterate_image_seed(uri))
        self.assertEqual(len(seeds), 1)
        
        seed = seeds[0]
        self.assertIsNone(seed.images)
        self.assertIsNotNone(seed.latents)
        self.assertIsNotNone(seed.control_images)
        self.assertEqual(len(seed.latents), 1)
        self.assertEqual(len(seed.control_images), 1)
        
        # Check types
        from PIL import Image
        self.assertTrue(torch.is_tensor(seed.latents[0]))
        self.assertIsInstance(seed.control_images[0], Image.Image)

    def test_latents_no_image_processing(self):
        """Test that latents are loaded as-is without image processing"""
        # Create a custom image processor to verify it's not called on tensors
        class MockImageProcessor:
            def __init__(self):
                self.called = False
                
            def process(self, image):
                self.called = True
                return image
        
        processor = MockImageProcessor()
        uri = f'latents: {self.latent1_path}'
        
        # This should not raise an error even though tensors can't be processed as images
        seeds = list(_mi.iterate_image_seed(uri, seed_image_processor=processor))
        seed = seeds[0]
        
        # Verify the processor was not called (tensors bypass image processing)
        self.assertFalse(processor.called)
        self.assertIsNotNone(seed.latents)
        self.assertTrue(torch.is_tensor(seed.latents[0]))

    def test_latents_tensor_file_formats(self):
        """Test that different tensor file formats are handled correctly"""
        for path in [self.latent1_path, self.latent2_path, self.latent3_path]:
            uri = f'latents: {path}'
            
            seeds = list(_mi.iterate_image_seed(uri))
            self.assertEqual(len(seeds), 1)
            
            seed = seeds[0]
            self.assertIsNotNone(seed.latents)
            self.assertEqual(len(seed.latents), 1)
            self.assertTrue(torch.is_tensor(seed.latents[0]))

    def test_latents_with_image_seed_context(self):
        """Test that latents maintain proper context when used with ImageSeed"""
        uri = f'latents: {self.latent1_path}'
        
        with next(_mi.iterate_image_seed(uri)) as seed:
            # Test that context manager works
            self.assertIsNotNone(seed.latents)
            self.assertTrue(torch.is_tensor(seed.latents[0]))
            
            # Test URI tracking
            self.assertEqual(seed.uri, uri)
            
            # Test animation frame info
            self.assertFalse(seed.is_animation_frame)
            self.assertIsNone(seed.total_frames)
            self.assertIsNone(seed.fps)


if __name__ == '__main__':
    unittest.main() 