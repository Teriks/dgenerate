import os
import unittest

from dgenerate.assistant.catalog import DEFAULT_EMBED_MODEL
from dgenerate.assistant.index import INDEX_VERSION, SHIPPED_INDEX, Index, choose_index, shipped_index_path


class TestSupportedAssistantModels(unittest.TestCase):

    def test_unknown_model_is_rejected(self):
        from dgenerate.assistant.cli import create_parser
        parser = create_parser('assistant')
        parser.parse_args(['--model', 'some/other/model.gguf', 'a cat'])
        self.assertEqual(parser.return_code, 2)
        parser.parse_args(['--embed-model', 'custom.gguf', 'a cat'])
        self.assertEqual(parser.return_code, 2)

    def test_a_listed_model_is_accepted(self):
        from dgenerate.assistant.catalog import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
        from dgenerate.assistant.cli import create_parser
        parser = create_parser('assistant')
        args = parser.parse_args(['--model', DEFAULT_CHAT_MODEL, '--embed-model', DEFAULT_EMBED_MODEL, 'a cat'])
        self.assertIsNone(parser.return_code)
        self.assertEqual(args.model, DEFAULT_CHAT_MODEL)
        self.assertEqual(args.embed_model, DEFAULT_EMBED_MODEL)


class TestChooseIndex(unittest.TestCase):

    def test_shipped_index_matches_the_default_embedder(self):
        meta = Index.read_meta(SHIPPED_INDEX)
        self.assertIsNotNone(meta)
        self.assertEqual(meta['version'], INDEX_VERSION)
        self.assertEqual(meta['embed_model'], os.path.basename(DEFAULT_EMBED_MODEL))

    def test_default_embedder_file_is_index_npz(self):
        self.assertEqual(
            os.path.normcase(shipped_index_path(os.path.basename(DEFAULT_EMBED_MODEL))),
            os.path.normcase(SHIPPED_INDEX))

    def test_other_embedder_file_is_named_for_the_model(self):
        path = shipped_index_path('Qwen3-Embedding-4B-Q8_0.gguf')
        self.assertTrue(path.endswith('Qwen3-Embedding-4B-Q8_0.npz'))

    def test_packaged_embedders_have_indexes(self):
        from dgenerate.assistant.catalog import EMBED_MODELS
        for spec, _size, _note in EMBED_MODELS:
            basename = os.path.basename(spec)
            meta = Index.read_meta(shipped_index_path(basename))
            self.assertIsNotNone(meta, basename)
            self.assertEqual(meta['embed_model'], basename)
            self.assertEqual(meta['version'], INDEX_VERSION)

    def test_default_embedder_uses_the_shipped_index(self):
        shipped = {'version': INDEX_VERSION, 'embed_model': 'Qwen3-Embedding-0.6B-Q8_0.gguf'}
        self.assertEqual(choose_index('Qwen3-Embedding-0.6B-Q8_0.gguf', shipped, None, 'abc'), 'shipped')

    def test_other_embedder_uses_a_matching_cache(self):
        shipped = {'version': INDEX_VERSION, 'embed_model': 'Qwen3-Embedding-0.6B-Q8_0.gguf'}
        cache = {'version': INDEX_VERSION, 'embed_model': 'Qwen3-Embedding-4B-Q8_0.gguf', 'fingerprint': 'abc'}
        self.assertEqual(
            choose_index('Qwen3-Embedding-4B-Q8_0.gguf', shipped, cache, 'abc'), 'cache')

    def test_stale_cache_is_rebuilt(self):
        shipped = {'version': INDEX_VERSION, 'embed_model': 'Qwen3-Embedding-0.6B-Q8_0.gguf'}
        cache = {'version': INDEX_VERSION, 'embed_model': 'Qwen3-Embedding-4B-Q8_0.gguf', 'fingerprint': 'old'}
        self.assertEqual(
            choose_index('Qwen3-Embedding-4B-Q8_0.gguf', shipped, cache, 'new'), 'build')

    def test_cache_without_a_repo_is_kept(self):
        cache = {'version': INDEX_VERSION, 'embed_model': 'custom.gguf', 'fingerprint': 'old'}
        self.assertEqual(choose_index('custom.gguf', None, cache, None), 'cache')


if __name__ == '__main__':
    unittest.main()
