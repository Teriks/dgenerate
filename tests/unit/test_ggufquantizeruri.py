import unittest

from dgenerate.hfhub import is_gguf_model, is_single_file_model_load
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTransformerUriError
from dgenerate.pipelinewrapper.uris.transformeruri import TransformerUri
from dgenerate.pipelinewrapper.uris.util import UnknownQuantizerName, get_quantizer_uri_class


class TestGGUFTransformer(unittest.TestCase):

    def test_gguf_is_not_a_quantizer_backend(self):
        with self.assertRaises(UnknownQuantizerName):
            get_quantizer_uri_class('gguf')

    def test_gguf_path_detection(self):
        self.assertTrue(is_gguf_model('model.gguf'))
        self.assertTrue(is_gguf_model(
            'https://huggingface.co/org/repo/blob/main/unet/model.gguf'))
        self.assertTrue(is_single_file_model_load('model.gguf'))
        self.assertFalse(is_gguf_model('black-forest-labs/FLUX.1-dev'))

    def test_gguf_transformer_uri_omits_quantizer(self):
        parsed = TransformerUri.parse('model.gguf')
        self.assertEqual(parsed.model, 'model.gguf')
        self.assertFalse(parsed.quantizer)

    def test_gguf_transformer_rejects_quantizer_uri(self):
        with self.assertRaises(InvalidTransformerUriError):
            TransformerUri.parse('model.gguf;quantizer=gguf')
        with self.assertRaises(InvalidTransformerUriError):
            TransformerUri.parse('model.gguf;quantizer=bnb;bits=4')


if __name__ == '__main__':
    unittest.main()
