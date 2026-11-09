import json
import re
import unittest
from pathlib import Path

import dgenerate.console.formentries.torchcompileentry as tce

_RECIPES = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes'


def _tags(text):
    found = []
    for line in text.splitlines():
        match = re.fullmatch(r'@(\w+)\[(\{.*\})\]', line.strip())
        if match:
            found.append((match.group(1), json.loads(match.group(2))))
    return found


class TestTorchCompileMapSelect(unittest.TestCase):
    def test_parse_collapses_prompt_enhancer_alias(self):
        self.assertEqual(
            tce.parse_torch_compile_text('prompt_enhancer, vae'),
            ['prompt_upsampler', 'vae'])
        self.assertEqual(
            tce.format_torch_compile_text(['prompt_enhancer', 'transformer']),
            'transformer prompt_upsampler')
        self.assertNotIn('prompt_enhancer', tce.TORCH_COMPILE_SUBMODULES)

    def test_options_follow_model_type(self):
        flux3 = tce.torch_compile_options_for_model_type('flux3')
        self.assertEqual(
            flux3,
            ['transformer', 'vae', 'text_encoder', 'prompt_upsampler'])
        self.assertEqual(
            tce.torch_compile_options_for_model_type('sd'),
            ['unet', 'controlnet', 'vae', 'text_encoder'])
        ltx2 = tce.torch_compile_options_for_model_type('ltx2')
        self.assertIn('connectors', ltx2)
        self.assertIn('prompt_upsampler', ltx2)
        self.assertIn('vae', ltx2)
        wan = tce.torch_compile_options_for_model_type('wan')
        self.assertIn('transformer_2', wan)
        self.assertIn('image_encoder', wan)
        self.assertNotIn('prompt_upsampler', wan)
        self.assertEqual(
            tce.torch_compile_options_for_model_type(''),
            list(tce.TORCH_COMPILE_SUBMODULES))

    def test_recipes_expose_torch_compile_map(self):
        model_re = re.compile(r'--model-type\s+([A-Za-z0-9_.-]+)')
        seen = 0
        for path in sorted(_RECIPES.glob('*.recipe')):
            text = path.read_text(encoding='utf-8')
            tags = _tags(text)
            self.assertNotIn('@torchcompilemap[', text, path.name)
            compile_switches = [
                config for name, config in tags
                if name == 'switch' and config.get('arg') == '--torch-compile'
            ]
            self.assertEqual(compile_switches, [], path.name)
            current_model = None
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith('@'):
                    declared = model_re.search(stripped)
                    if declared:
                        current_model = declared.group(1)
                    continue
                match = re.fullmatch(r'@(\w+)\[(\{.*\})\]', stripped)
                if match is None or match.group(1) != 'torchcompile':
                    continue
                config = json.loads(match.group(2))
                self.assertNotIn('enabled-when', config, path.name)
                if current_model:
                    self.assertEqual(config.get('model-type'), current_model, path.name)
                else:
                    quantizer = next(
                        item for item_name, item in tags if item_name == 'quantizer')
                    self.assertEqual(
                        config.get('model-type'), quantizer.get('model-type'), path.name)
                self.assertTrue(config.get('model-type'), path.name)
                seen += 1
        self.assertGreater(seen, 200)
        self.assertEqual(tce._TorchCompileEntry.NAME, 'torchcompile')
