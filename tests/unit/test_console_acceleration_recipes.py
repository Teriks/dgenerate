import json
import re
import unittest
from pathlib import Path

try:
    import tkinter as tk
    _root = tk.Tk()
    _root.withdraw()
    _root.destroy()
except Exception:
    tk = None

_RECIPES = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes'

_CHILD_ARGS = (
    '--spectrum-weights',
    '--spectrum-orders',
    '--spectrum-lambdas',
    '--spectrum-warmup-steps',
    '--spectrum-window-sizes',
    '--spectrum-flex-windows',
    '--sada-max-downsamples',
    '--sada-sxs',
    '--sada-sys',
    '--sada-acc-ranges',
    '--sada-lagrange-terms',
    '--sada-lagrange-ints',
    '--sada-lagrange-steps',
    '--sada-max-fixes',
    '--sada-max-intervals',
    '--ras-index-fusion',
    '--ras-sample-ratios',
    '--ras-high-ratios',
    '--ras-starvation-scales',
    '--ras-error-reset-steps',
    '--ras-metrics',
    '--ras-start-steps',
    '--ras-end-steps',
    '--ras-skip-num-steps',
    '--ras-skip-num-step-lengths',
    '--regione-warmup-steps',
    '--regione-post-steps',
    '--regione-refresh-steps',
    '--regione-thresholds',
    '--regione-cache-thresholds',
)


def _tags(text):
    found = []
    for line in text.splitlines():
        match = re.fullmatch(r'@(\w+)\[(\{.*\})\]', line.strip())
        if match:
            found.append((match.group(1), json.loads(match.group(2))))
    return found


class TestConsoleAccelerationRecipes(unittest.TestCase):
    def test_optimizations_are_single_controls(self):
        for path in sorted(_RECIPES.glob('*.recipe')):
            tags = _tags(path.read_text(encoding='utf-8'))
            by_name = {}
            for name, config in tags:
                by_name.setdefault(name, []).append(config)
                self.assertNotIn(config.get('arg'), _CHILD_ARGS, path.name)

            for config in by_name.get('sada', []):
                self.assertEqual(config.get('id'), 'sada', path.name)
                self.assertIn(config.get('preset'), ('sd', 'sdxl', 'flux'), path.name)
                self.assertTrue(config.get('divider-after'), path.name)
            self.assertLessEqual(len(by_name.get('sada', [])), 1, path.name)

            for config in by_name.get('spectrum', []):
                self.assertEqual(config.get('id'), 'sp', path.name)
                self.assertTrue(config.get('divider-after'), path.name)
            self.assertLessEqual(len(by_name.get('spectrum', [])), 1, path.name)

            for config in by_name.get('ras', []):
                self.assertEqual(config.get('id'), 'ras', path.name)
                self.assertTrue(config.get('divider-after'), path.name)
                if '3.5' in path.name:
                    self.assertNotIn('index-fusion', config, path.name)
                else:
                    self.assertTrue(config.get('index-fusion'), path.name)
            self.assertLessEqual(len(by_name.get('ras', [])), 1, path.name)

            for config in by_name.get('regione', []):
                self.assertEqual(config.get('id'), 're', path.name)
                self.assertIn(
                    config.get('preset'), ('flux-kontext', 'qwen-edit'), path.name)
                if 'qwen-image-edit' in path.name:
                    self.assertEqual(config.get('preset'), 'qwen-edit', path.name)
                elif 'flux-kontext' in path.name:
                    self.assertEqual(config.get('preset'), 'flux-kontext', path.name)
                self.assertTrue(config.get('divider-after'), path.name)
            self.assertLessEqual(len(by_name.get('regione', [])), 1, path.name)

            teacache = [
                config for name, config in tags
                if name == 'checkboxwithfloatarg' and config.get('help-arg') == '--tea-cache'
            ]
            if teacache:
                self.assertEqual(len(teacache), 1, path.name)
                self.assertNotIn(
                    '--tea-cache-rel-l1-thresholds',
                    [config.get('arg') for _name, config in tags],
                    path.name)


class _Form:
    def get_entry_by_id(self, entry_id):
        return None


@unittest.skipIf(tk is None, 'Tk is not available')
class TestOptimizationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def _make(self, entry_class, config):
        frame = tk.Frame(self.root)
        return entry_class(
            form=_Form(),
            master=frame,
            row=0,
            config=config,
            placeholder='SLOT')

    def test_unchanged_values_are_omitted(self):
        from dgenerate.console.formentries.rasentry import _RASEntry
        from dgenerate.console.formentries.regioneentry import _RegionEEntry
        from dgenerate.console.formentries.sadaentry import _SADAEntry
        from dgenerate.console.formentries.spectrumentry import _SpectrumEntry

        spectrum = self._make(_SpectrumEntry, {'id': 'sp'})
        spectrum.enabled_var.set(True)
        self.assertEqual(spectrum.template('SLOT'), '--spectrum')
        spectrum.weight_var.set('0.9')
        self.assertEqual(spectrum.template('SLOT'), '--spectrum-weights 0.9')
        spectrum.enabled_var.set(False)
        self.assertEqual(spectrum.template('keep SLOT here'), 'keep  here')

        sada = self._make(_SADAEntry, {'id': 'sada', 'preset': 'flux'})
        sada.enabled_var.set(True)
        self.assertEqual(sada.template('SLOT'), '--sada')
        sada.downsample_var.set('2')
        self.assertEqual(sada.template('SLOT'), '--sada-max-downsamples 2')

        ras = self._make(_RASEntry, {'id': 'ras', 'index-fusion': True})
        ras.enabled_var.set(True)
        self.assertEqual(ras.template('SLOT'), '--ras')
        ras.index_fusion_var.set(True)
        ras.end_var.set('20')
        self.assertEqual(
            ras.template('SLOT'),
            '--ras-index-fusion\n--ras-end-steps 20')

        plain = self._make(_RASEntry, {'id': 'ras'})
        self.assertIsNone(plain.index_fusion_var)

        regione = self._make(_RegionEEntry, {'id': 're', 'preset': 'qwen-edit'})
        regione.enabled_var.set(True)
        self.assertEqual(regione.template('SLOT'), '--regione')
        regione.threshold_var.set('0.9')
        self.assertEqual(regione.template('SLOT'), '--regione-thresholds 0.9')

    def test_sada_preset_is_required(self):
        from dgenerate.console.formentries.sadaentry import _SADAEntry

        with self.assertRaises(RuntimeError):
            self._make(_SADAEntry, {'id': 'sada'})

    def test_regione_preset_is_required(self):
        from dgenerate.console.formentries.regioneentry import _RegionEEntry

        with self.assertRaises(RuntimeError):
            self._make(_RegionEEntry, {'id': 're'})


if __name__ == '__main__':
    unittest.main()
