import types
import unittest

from dgenerate.extras.compel.embeddings_provider import clip_final_layer_norm
from dgenerate.extras.sd_embed.embedding_funcs import clip_text_tower


class TestClipTextTower(unittest.TestCase):

    def test_transformers_5_clip_text_model_has_no_nested_tower(self):
        encoder = types.SimpleNamespace(layers=['layer'])
        text_encoder = types.SimpleNamespace(encoder=encoder, final_layer_norm='norm')
        self.assertIs(clip_text_tower(text_encoder), text_encoder)
        self.assertIs(clip_final_layer_norm(text_encoder), 'norm')

    def test_nested_text_model_is_still_used(self):
        encoder = types.SimpleNamespace(layers=['layer'])
        nested = types.SimpleNamespace(encoder=encoder, final_layer_norm='nested-norm')
        text_encoder = types.SimpleNamespace(text_model=nested)
        self.assertIs(clip_text_tower(text_encoder), nested)
        self.assertIs(clip_final_layer_norm(text_encoder), 'nested-norm')


if __name__ == '__main__':
    unittest.main()
