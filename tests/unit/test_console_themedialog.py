import platform
import unittest

try:
    import tkinter as tk
    import tkinter.ttk as ttk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class ThemeDialogTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_other_platforms_leave_the_native_theme(self):
        from dgenerate.console import themedialog

        canvas = tk.Canvas(_ROOT, background='#ffffff', highlightthickness=0)
        self.addCleanup(canvas.destroy)
        style = ttk.Style(_ROOT)
        before = style.theme_use()
        themedialog._CUSTOM = False
        try:
            themedialog.apply(_ROOT, '#282a36', '#f8f8f2', '#44475a', '#f8f8f2')
        finally:
            themedialog._CUSTOM = platform.system() in ('Windows', 'Linux', 'Darwin')
        self.assertEqual(style.theme_use(), before)
        self.assertEqual(str(canvas.cget('background')).lower(), '#ffffff')

    def test_editor_theme_recolors_dialogs(self):
        if platform.system() not in ('Windows', 'Linux', 'Darwin'):
            self.skipTest('dialog theming is windows, linux, and macos')
        from dgenerate.console import themedialog, themescrollbar

        white = tk.Canvas(_ROOT, background='white', highlightthickness=0)
        card = tk.Frame(_ROOT, background='#f7f7f7')
        viewer = tk.Canvas(_ROOT, background='black', highlightthickness=0)
        sash = tk.Frame(_ROOT, background='#808080')
        self.addCleanup(white.destroy)
        self.addCleanup(card.destroy)
        self.addCleanup(viewer.destroy)
        self.addCleanup(sash.destroy)

        themedialog.apply(_ROOT, '#282a36', '#f8f8f2', '#44475a', '#f8f8f2')

        style = ttk.Style(_ROOT)
        self.assertEqual(style.theme_use(), 'clam')
        self.assertEqual(style.lookup('TFrame', 'background'), '#282a36')
        self.assertEqual(style.lookup('TLabel', 'foreground'), '#f8f8f2')
        self.assertEqual(int(style.lookup('TButton', 'width')), 0)
        padding = style.lookup('TButton', 'padding')
        if isinstance(padding, str):
            padding = padding.split()
        self.assertEqual([int(value) for value in padding[:2]], [4, 2])
        self.assertEqual(style.lookup('TEntry', 'fieldbackground'), '#282a36')
        disabled_field = style.lookup('TEntry', 'fieldbackground', ['disabled'])
        self.assertNotEqual(disabled_field, '#282a36')
        red, green, blue = _ROOT.winfo_rgb(disabled_field)
        self.assertLess(red, _ROOT.winfo_rgb('#282a36')[0])
        entry = tk.Entry(_ROOT)
        self.addCleanup(entry.destroy)
        self.assertNotEqual(str(entry.cget('disabledbackground')).lower(), 'systembuttonface')
        self.assertEqual(str(white.cget('background')).lower(), '#282a36')
        self.assertEqual(str(card.cget('background')).lower(), '#282a36')
        self.assertEqual(str(viewer.cget('background')), 'black')
        self.assertEqual(str(sash.cget('background')).lower(), '#808080')

        later = tk.Canvas(_ROOT, highlightthickness=0)
        self.addCleanup(later.destroy)
        self.assertEqual(str(later.cget('background')).lower(), '#282a36')

        bar = themescrollbar.scrollbar(_ROOT)
        self.addCleanup(bar.destroy)
        themescrollbar.apply(_ROOT, '#282a36', '#f8f8f2')
        self.assertEqual(style.lookup(themescrollbar.VERTICAL_STYLE, 'troughcolor'), '#282a36')
        self.assertEqual(int(style.lookup('Vertical.TScrollbar', 'gripcount')), 0)
        self.assertEqual(int(style.lookup('Vertical.TScrollbar', 'gripsize')), 0)

        themedialog.apply(
            _ROOT, '#ffffff', '#000000', '#cce8ff', '#000000',
            window_background='#f2f2f2', window_chrome='#e3e3e3')
        self.assertEqual(style.lookup('TFrame', 'background'), '#f2f2f2')
        self.assertEqual(style.lookup('TEntry', 'fieldbackground'), '#ffffff')
        self.assertEqual(style.lookup('TEntry', 'bordercolor'), '#e3e3e3')

        themedialog.apply(_ROOT, '#f8f8f2', '#282a36', '#44475a', '#282a36')
        self.assertEqual(style.lookup('TFrame', 'background'), '#f8f8f2')
        self.assertEqual(str(white.cget('background')).lower(), '#f8f8f2')
        self.assertEqual(str(viewer.cget('background')), 'black')
        self.assertEqual(str(sash.cget('background')).lower(), '#808080')

    def test_viewport_border_uses_the_editor_edge(self):
        from dgenerate.console import appearance, themedialog

        themedialog.apply(
            _ROOT, '#ffffff', '#000000', '#cce8ff', '#000000',
            window_background='#112233', window_chrome='#445566')
        frame = tk.Frame(_ROOT)
        self.addCleanup(frame.destroy)
        appearance.configure_viewport(frame)
        self.assertEqual(str(frame.cget('highlightbackground')).lower(), '#445566')
        self.assertEqual(str(frame.cget('background')).lower(), '#112233')

    def test_aqua_system_colors_step_aside_after_the_editor_theme(self):
        from dgenerate.console import appearance, themedialog

        themedialog.apply(_ROOT, '#282a36', '#f8f8f2', '#44475a', '#f8f8f2')
        frame = tk.Frame(_ROOT, background='#abcdef')
        self.addCleanup(frame.destroy)
        seen = []
        original_configure = appearance._configure
        original_aqua = appearance.is_aqua

        def spy(widget, **options):
            seen.append(options)
            return original_configure(widget, **options)

        appearance.is_aqua = lambda widget: True
        appearance._configure = spy
        try:
            appearance.install(frame)
        finally:
            appearance.is_aqua = original_aqua
            appearance._configure = original_configure
        self.assertEqual(seen, [])
        self.assertEqual(str(frame.cget('background')).lower(), '#abcdef')
