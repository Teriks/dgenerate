import types
import unittest

from dgenerate.promptupscalers.magicpromptupscaler import (
    _attn_implementation_for_config,
)


class TestMagicPromptAttn(unittest.TestCase):
    def test_sliding_window_uses_eager(self):
        config = types.SimpleNamespace(sliding_window=262144)
        self.assertEqual(_attn_implementation_for_config(config), 'eager')

    def test_no_sliding_window_keeps_default(self):
        config = types.SimpleNamespace()
        self.assertIsNone(_attn_implementation_for_config(config))
        config.sliding_window = None
        self.assertIsNone(_attn_implementation_for_config(config))


if __name__ == '__main__':
    unittest.main()
