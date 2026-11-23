import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class ThemeScrollbarTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_dark_editor_colors_the_trough_and_thumb(self):
        import tkinter.ttk as ttk

        from dgenerate.console.themescrollbar import HORIZONTAL_STYLE, VERTICAL_STYLE, apply, scrollbar

        bar = scrollbar(_ROOT, orient=tk.VERTICAL)
        self.addCleanup(bar.destroy)
        apply(_ROOT, '#282c34', '#abb2bf')

        style = ttk.Style(_ROOT)
        self.assertEqual(str(bar.cget('style')), VERTICAL_STYLE)
        self.assertEqual(style.lookup(VERTICAL_STYLE, 'troughcolor'), '#282c34')
        self.assertNotEqual(style.lookup(VERTICAL_STYLE, 'background'), '#282c34')
        self.assertEqual(style.lookup(HORIZONTAL_STYLE, 'troughcolor'), '#282c34')
        self.assertEqual(int(style.lookup(VERTICAL_STYLE, 'gripsize')), 0)
        self.assertEqual(int(style.lookup(HORIZONTAL_STYLE, 'gripsize')), 0)
        self.assertEqual(int(style.lookup(VERTICAL_STYLE, 'gripcount')), 0)
        self.assertEqual(int(style.lookup(HORIZONTAL_STYLE, 'gripcount')), 0)

    def test_light_editor_uses_a_lighter_thumb_than_a_dark_editor(self):
        import tkinter.ttk as ttk

        from dgenerate.console.themescrollbar import VERTICAL_STYLE, apply

        apply(_ROOT, '#ffffff', '#000000')
        light_thumb = ttk.Style(_ROOT).lookup(VERTICAL_STYLE, 'background')
        apply(_ROOT, '#282c34', '#abb2bf')
        dark_thumb = ttk.Style(_ROOT).lookup(VERTICAL_STYLE, 'background')
        self.assertNotEqual(light_thumb, dark_thumb)
