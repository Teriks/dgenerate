import unittest

from dgenerate.pipelinewrapper.uris.exceptions import InvalidSDNQQuantizerUriError
from dgenerate.pipelinewrapper.uris.sdnqquantizeruri import SDNQQuantizerUri


class TestSDNQQuantizerUri(unittest.TestCase):

    def test_basic_parsing(self):
        result = SDNQQuantizerUri.parse('sdnq')
        self.assertEqual(result.type, 'int8')
        self.assertEqual(result.group_size, 0)
        self.assertFalse(result.quant_conv)
        self.assertFalse(result.quantized_matmul)
        self.assertFalse(result.quantized_matmul_conv)

    def test_full_options_parsing(self):
        uri = 'sdnq;type=int4;group-size=8;quant-conv=true;quantized-matmul=true;quantized-matmul-conv=true'
        result = SDNQQuantizerUri.parse(uri)
        self.assertEqual(result.type, 'int4')
        self.assertEqual(result.group_size, 8)
        self.assertTrue(result.quant_conv)
        self.assertTrue(result.quantized_matmul)
        self.assertTrue(result.quantized_matmul_conv)

    def test_type_validation(self):
        for dtype in SDNQQuantizerUri._valid_weight_dtypes:
            self.assertEqual(SDNQQuantizerUri.parse(f'sdnq;type={dtype}').type, dtype)

        with self.assertRaises(InvalidSDNQQuantizerUriError) as ctx:
            SDNQQuantizerUri.parse('sdnq;type=not-a-dtype')
        self.assertIn('must be one of', str(ctx.exception))

    def test_group_size_validation(self):
        with self.assertRaises(InvalidSDNQQuantizerUriError):
            SDNQQuantizerUri.parse('sdnq;group-size=-1')

    def test_to_config_uses_pypi_sdnq(self):
        import sdnq

        config = SDNQQuantizerUri.parse('sdnq;type=uint4;group-size=32').to_config()
        self.assertIsInstance(config, sdnq.SDNQConfig)
        self.assertEqual(config.weights_dtype, 'uint4')
        self.assertEqual(config.group_size, 32)

        tf_config = SDNQQuantizerUri.parse('sdnq;type=int8').to_transformers_config()
        self.assertIsInstance(tf_config, sdnq.SDNQConfig)

    def test_pypi_accepts_every_uri_dtype(self):
        import sdnq.common

        missing = set(SDNQQuantizerUri._valid_weight_dtypes) - set(sdnq.common.accepted_weight_dtypes)
        self.assertFalse(missing, f'PyPI sdnq is missing URI dtypes: {sorted(missing)}')


if __name__ == '__main__':
    unittest.main()
