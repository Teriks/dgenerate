import unittest

import dgenerate
from transformers import CLIPConfig, CLIPVisionConfig

from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deprecated.stable_diffusion_safe.safety_checker import (
    SafeStableDiffusionSafetyChecker,
)


def _tiny_clip_config():
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        projection_dim=16,
    )
    return CLIPConfig(vision_config=vision, projection_dim=16, text_config={'hidden_size': 32})


class TestSafetyCheckerPostInit(unittest.TestCase):

    def test_if_safety_checker_has_tied_weight_keys(self):
        checker = IFSafetyChecker(_tiny_clip_config())
        self.assertIsInstance(checker.all_tied_weights_keys, dict)

    def test_safe_stable_diffusion_checker_has_tied_weight_keys(self):
        checker = SafeStableDiffusionSafetyChecker(_tiny_clip_config())
        self.assertIsInstance(checker.all_tied_weights_keys, dict)


if __name__ == '__main__':
    unittest.main()
