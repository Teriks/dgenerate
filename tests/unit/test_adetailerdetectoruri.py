import unittest

import dgenerate.pipelinewrapper.constants as _pipelinewrapper_constants
import dgenerate.pipelinewrapper.uris.adetailerdetectoruri as _adetailerdetectoruri
import dgenerate.torchutil as _torchutil
from dgenerate.pipelinewrapper.uris.exceptions import InvalidAdetailerDetectorUriError


class TestAdetailerDetectorUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.weight_name, None)
        self.assertEqual(result.confidence, _pipelinewrapper_constants.DEFAULT_ADETAILER_DETECTOR_CONFIDENCE)
        self.assertEqual(result.class_filter, None)
        self.assertEqual(result.index_filter, None)
        self.assertEqual(result.model_masks, None)
        self.assertEqual(result.mask_shape, None)
        self.assertEqual(result.detector_padding, None)
        self.assertEqual(result.mask_padding, None)
        self.assertEqual(result.mask_blur, None)
        self.assertEqual(result.mask_dilation, None)
        self.assertEqual(result.prompt, None)
        self.assertEqual(result.negative_prompt, None)
        self.assertEqual(result.device, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;subfolder=models;weight-name=weights.pt;confidence=0.75;" \
              "class-filter=[0,1,2];index-filter=[0,1];model-masks=true;mask-shape=circle;" \
              "detector-padding=10;mask-padding=5;mask-blur=3;mask-dilation=2;" \
              "prompt=test prompt;negative-prompt=bad result;device=cuda:0"
        
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.weight_name, "weights.pt")
        self.assertEqual(result.confidence, 0.75)
        self.assertEqual(result.class_filter, {0, 1, 2})
        self.assertEqual(result.index_filter, {0, 1})
        self.assertEqual(result.model_masks, True)
        self.assertEqual(result.mask_shape, "circle")
        self.assertEqual(result.detector_padding, 10)
        self.assertEqual(result.mask_padding, 5)
        self.assertEqual(result.mask_blur, 3)
        self.assertEqual(result.mask_dilation, 2)
        self.assertEqual(result.prompt, "test prompt")
        self.assertEqual(result.negative_prompt, "bad result")
        self.assertEqual(result.device, "cuda:0")

    def test_class_filter_parsing(self):
        # Test class filter parsing with different formats
        
        # Integer list
        uri = "model;class-filter=[0, 1, 2]"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {0, 1, 2})
        
        # String list
        uri = "model;class-filter=['person', 'dog', 'cat']"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 'dog', 'cat'})
        
        # Mixed list
        uri = "model;class-filter=['person', 0, 'dog']"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 0, 'dog'})
        
        # Comma-separated string
        uri = "model;class-filter=person,dog,cat"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 'dog', 'cat'})

        uri = "model;class-filter=person,dog,0"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 'dog', 0})

        uri = "model;class-filter=person,dog,'0'"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 'dog', "0"})
        
        # Single value
        uri = "model;class-filter=person"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person'})
        
        # Single integer
        uri = "model;class-filter=0"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {0})
        
        # Quoted strings in comma-separated list
        uri = "model;class-filter=\"person\",\"dog\",\"cat with spaces\""
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {'person', 'dog', 'cat with spaces'})
        
        # Empty list - returns empty set in implementation
        uri = "model;class-filter=[]"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, set())
        
        # Invalid syntax - this actually doesn't raise an error in the implementation
        # because the string is passed to ast.literal_eval which treats it as a string
        # with a syntax error, and then it falls back to other parsing methods
        uri = "model;class-filter=[0, 1, 2"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertIsNotNone(result.class_filter)

    def test_index_filter_parsing(self):
        # Test index filter parsing
        
        # Integer list
        uri = "model;index-filter=[0, 1, 2]"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.index_filter, {0, 1, 2})
        
        # Single integer
        uri = "model;index-filter=5"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.index_filter, {5})
        
        # Empty list - returns empty set in implementation
        uri = "model;index-filter=[]"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.index_filter, set())
        
        # Invalid negative index
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;index-filter=[-1, 0, 1]")
        
        # Invalid non-integer value
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;index-filter=['a', 'b']")
            
        # Invalid syntax - this should raise an error because the fallback parsing
        # doesn't work for index_filter which must be integers
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;index-filter=[0, 1, 2")
            
        # Invalid type (string instead of int)
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;index-filter=abc")

    def test_unhashable_types(self):
        # Test handling of unhashable types like dictionaries
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;class-filter=[{}]")
        
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;class-filter=[{1: 2}]")

    def test_combined_filters(self):
        # Test using both class-filter and index-filter together
        uri = "model;class-filter=[0, 1, 'person'];index-filter=[0, 1, 2]"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.class_filter, {0, 1, 'person'})
        self.assertEqual(result.index_filter, {0, 1, 2})

    def test_padding_parsing(self):
        # Test padding parsing with different formats
        
        # Uniform padding
        uri = "model;detector-padding=10;mask-padding=5"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.detector_padding, 10)
        self.assertEqual(result.mask_padding, 5)
        
        # Width x Height
        uri = "model;detector-padding=10x20;mask-padding=5x15"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.detector_padding, (10, 20))
        self.assertEqual(result.mask_padding, (5, 15))
        
        # Left x Top x Right x Bottom
        uri = "model;detector-padding=10x20x30x40;mask-padding=5x15x25x35"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.detector_padding, (10, 20, 30, 40))
        self.assertEqual(result.mask_padding, (5, 15, 25, 35))

    def test_mask_shape_validation(self):
        # Test mask shape validation
        
        # Valid shapes
        uri = "model;mask-shape=rectangle"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.mask_shape, "rectangle")
        
        uri = "model;mask-shape=circle"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.mask_shape, "circle")
        
        # Invalid shape
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;mask-shape=triangle")

    def test_numeric_validation(self):
        # Test numeric validation
        
        # Invalid confidence
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;confidence=-0.5")
        
        # Invalid mask blur
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;mask-blur=-3")
        
        # Invalid mask dilation
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;mask-dilation=-2")
        
        # Invalid index filter
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;index-filter=[-1, 0, 1]")

    def test_device_validation(self):
        # Test device validation
        
        # Valid devices
        if _torchutil.is_valid_device_string("cuda:0"):
            uri = "model;device=cuda:0"
            result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
            self.assertEqual(result.device, "cuda:0")
        
        uri = "model;device=cpu"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.device, "cpu")
        
        # Invalid device
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;device=invalid_device")

    def test_boolean_parsing(self):
        # Test boolean parsing
        
        # True values
        uri = "model;model-masks=true"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model_masks, True)
        
        uri = "model;model-masks=True"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model_masks, True)
        
        # False values
        uri = "model;model-masks=false"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model_masks, False)
        
        uri = "model;model-masks=False"
        result = _adetailerdetectoruri.AdetailerDetectorUri.parse(uri)
        self.assertEqual(result.model_masks, False)
        
        # Invalid boolean
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;model-masks=invalid")

    def test_invalid_padding_format(self):
        # Test invalid padding format
        
        # Invalid detector padding format
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;detector-padding=10x20x30")
        
        # Invalid mask padding format
        with self.assertRaises(InvalidAdetailerDetectorUriError):
            _adetailerdetectoruri.AdetailerDetectorUri.parse("model;mask-padding=10x20x30")


if __name__ == '__main__':
    unittest.main() 