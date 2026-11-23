import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None

_AYU = {
    'editor': {
        'bg': '#0a0e14',
        'fg': '#b3b1ad',
        'select_bg': '#1b2733',
        'inactive_select_bg': '#1b2733',
    },
    'general': {
        'comment': '#626a73',
        'error': '#ff3333',
        'escape': '#b3b1ad',
        'keyword': '#ff7700',
        'name': '#ff8f40',
        'string': '#95e6cb',
        'punctuation': '#b3b1ad',
    },
}


class SyntaxHighlighterTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_a_dark_theme_without_selection_text_stays_readable_after_map(self):
        from pygments.lexers.special import TextLexer

        from dgenerate.console.syntaxhighlighter import TkTextHighlighter

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        parent.withdraw()
        text = tk.Text(parent)
        text.pack()
        highlighter = TkTextHighlighter(text, TextLexer())
        highlighter.set_theme(_AYU)
        text.configure(selectbackground='#ff0000', selectforeground='#000000')
        parent.deiconify()
        parent.update()
        self.assertEqual(str(text.cget('selectbackground')).lower(), '#1b2733')
        self.assertEqual(str(text.cget('selectforeground')).lower(), '#b3b1ad')
