import tempfile
import unittest

from dgenerate.extras.DistillT5.models.T5_encoder import (
    T5EncoderWithProjection,
    T5ProjectionConfig,
)


def _tiny_config():
    return T5ProjectionConfig(
        vocab_size=32,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        relative_attention_num_buckets=8,
        project_in_dim=16,
        out_dim=8,
    )


class TestDistillT5EncoderPostInit(unittest.TestCase):

    def test_constructor_sets_tied_weight_keys(self):
        model = T5EncoderWithProjection(_tiny_config())
        self.assertIsInstance(model.all_tied_weights_keys, dict)

    def test_from_pretrained_roundtrip(self):
        model = T5EncoderWithProjection(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = T5EncoderWithProjection.from_pretrained(directory)
        self.assertIsInstance(loaded.all_tied_weights_keys, dict)
        self.assertEqual(
            loaded.final_projection[0].weight.shape,
            model.final_projection[0].weight.shape,
        )


if __name__ == '__main__':
    unittest.main()
