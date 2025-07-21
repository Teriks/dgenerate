import unittest

import dgenerate.pipelinewrapper.constants as _pipelinewrapper_constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.controlneturi as _controlneturi
from dgenerate.pipelinewrapper.uris.exceptions import InvalidControlNetUriError


class TestControlNetUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.start, 0.0)
        self.assertEqual(result.end, 1.0)
        self.assertEqual(result.mode, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.model_type, _enums.ModelType.SD)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;variant=fp16;subfolder=models;" \
              "scale=0.8;start=0.2;end=0.9;dtype=float16"
        
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.scale, 0.8)
        self.assertEqual(result.start, 0.2)
        self.assertEqual(result.end, 0.9)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)

    def test_scale_parsing(self):
        # Test scale parsing
        
        # Single float value
        uri = "model;scale=0.75"
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.scale, 0.75)

    def test_start_end_validation(self):
        # Test start/end validation
        
        # Valid start/end
        uri = "model;start=0.2;end=0.8"
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.start, 0.2)
        self.assertEqual(result.end, 0.8)
        
        # Invalid start (not a float)
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;start=abc")
        
        # Invalid end (not a float)
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;end=abc")
        
        # Invalid start (greater than end)
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;start=0.8;end=0.5")

    def test_scale_validation(self):
        # Test scale validation
        
        # Invalid scale (not a float)
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;scale=invalid")

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "model;dtype=float16"
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "model;dtype=float32"
        result = _controlneturi.ControlNetUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;dtype=invalid_dtype")

    def test_mode_parsing_for_flux(self):
        # Test mode parsing for Flux model type
        
        # String mode (flux)
        uri = "model;mode=canny"
        result = _controlneturi.ControlNetUri.parse(uri, model_type=_enums.ModelType.FLUX)
        self.assertIsInstance(result.mode, int)
        
        # Integer mode
        uri = "model;mode=2"
        result = _controlneturi.ControlNetUri.parse(uri, model_type=_enums.ModelType.FLUX)
        self.assertEqual(result.mode, 2)
        
        # Invalid mode for TORCH model type
        with self.assertRaises(InvalidControlNetUriError):
            _controlneturi.ControlNetUri.parse("model;mode=canny", model_type=_enums.ModelType.SD)

    def test_mode_parsing_for_sdxl(self):
        # Test mode parsing for SDXL model type
        
        # String mode (sdxl)
        uri = "model;mode=openpose"
        result = _controlneturi.ControlNetUri.parse(uri, model_type=_enums.ModelType.SDXL)
        self.assertIsInstance(result.mode, int)
        
        # Integer mode
        uri = "model;mode=2"
        result = _controlneturi.ControlNetUri.parse(uri, model_type=_enums.ModelType.SDXL)
        self.assertEqual(result.mode, 2)


if __name__ == '__main__':
    unittest.main() 