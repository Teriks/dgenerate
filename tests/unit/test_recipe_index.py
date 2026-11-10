import json
import unittest
from pathlib import Path

_RECIPES = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes'


class TestRecipeIndex(unittest.TestCase):
    def test_index_matches_recipe_files(self):
        index = json.loads((_RECIPES / 'index.json').read_text(encoding='utf-8'))
        on_disk = {path.name for path in _RECIPES.glob('*.recipe')}

        self.assertIsInstance(index, list)
        self.assertEqual(len(index), len(set(index)))
        self.assertEqual(set(index), on_disk)

        for name in on_disk:
            lines = (_RECIPES / name).read_text(encoding='utf-8').splitlines()
            self.assertTrue(lines[0].startswith('title:'), name)
            self.assertFalse(lines[0].split(':', 1)[1].strip() == '', name)
            self.assertFalse(any(line.startswith('order:') for line in lines[:3]), name)

        self.assertEqual(index[-4:], [
            'stable-diffusion-upscaler-x2.recipe',
            'stable-diffusion-upscaler-x4.recipe',
            'generic-image-process.recipe',
            'generic-image-process-to-directory.recipe',
        ])

    def test_get_recipes_follows_index_order(self):
        import dgenerate.console.resources as resources

        resources._RECIPES.clear()
        loaded = resources.get_recipes()
        index = json.loads((_RECIPES / 'index.json').read_text(encoding='utf-8'))
        titles = []
        for name in index:
            first = (_RECIPES / name).read_text(encoding='utf-8').splitlines()[0]
            titles.append(first.split(':', 1)[1].strip())
        self.assertEqual(list(loaded), titles)

    def test_filter_prefixes_come_from_recipe_titles(self):
        from dgenerate.console.recipesform import recipe_filter_prefixes

        titles = []
        for path in sorted(_RECIPES.glob('*.recipe')):
            first = path.read_text(encoding='utf-8').splitlines()[0]
            titles.append(first.split(':', 1)[1].strip())

        prefixes = recipe_filter_prefixes(titles)
        self.assertIn('Stable Diffusion 1.5', prefixes)
        self.assertIn('Flux', prefixes)
        self.assertIn('Flux Schnell', prefixes)

        for prefix in prefixes:
            matched = [
                title for title in titles
                if title.lower().startswith(prefix.lower())
            ]
            self.assertGreaterEqual(len(matched), 2, prefix)


if __name__ == '__main__':
    unittest.main()
