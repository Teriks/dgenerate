import unittest

import dgenerate.console.quantizermapselect as qms


class TestQuantizerMapSelect(unittest.TestCase):
    def test_parse_and_format_quantizer_map(self):
        self.assertEqual(
            qms.parse_quantizer_map_text('text_encoder text_encoder_2'),
            ['text_encoder', 'text_encoder_2'])
        self.assertEqual(
            qms.parse_quantizer_map_text('text_encoder,transformer'),
            ['text_encoder', 'transformer'])
        self.assertEqual(
            qms.format_quantizer_map_text(['transformer', 'text_encoder']),
            'transformer text_encoder')

    def test_options_follow_model_type(self):
        self.assertEqual(
            qms.quantizer_map_options_for_model_type('qwen-image-edit'),
            ['transformer', 'text_encoder'])
        self.assertEqual(
            qms.quantizer_map_options_for_model_type('sd3'),
            ['transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'controlnet'])
        self.assertIn('transformer_2', qms.quantizer_map_options_for_model_type('wan'))
        self.assertNotIn('vae', qms.QUANTIZER_MAP_SUBMODULES)
        self.assertNotIn('vae', qms.quantizer_map_options_for_model_type('wan'))
        self.assertEqual(
            qms.quantizer_map_options_for_model_type(''),
            list(qms.QUANTIZER_MAP_SUBMODULES))
