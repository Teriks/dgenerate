import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class WordWrapScrollbarTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_code_view_hides_the_horizontal_bar_while_wrapped(self):
        from dgenerate.console.codeview import DgenerateCodeView

        view = DgenerateCodeView(_ROOT)
        self.addCleanup(view.destroy)
        view.enable_word_wrap()
        self.assertEqual(view.x_scrollbar.winfo_manager(), '')
        self.assertEqual(str(view.text.cget('wrap')), 'word')

        view.disable_word_wrap()
        self.assertEqual(view.x_scrollbar.winfo_manager(), 'grid')
        self.assertEqual(str(view.text.cget('wrap')), 'none')
        self.assertEqual(view.y_scrollbar.grid_info()['rowspan'], 2)

    def test_scrolled_text_hides_the_horizontal_bar_while_wrapped(self):
        from dgenerate.console.scrolledtext import ScrolledText

        view = ScrolledText(_ROOT)
        self.addCleanup(view.destroy)
        self.assertEqual(view.x_scrollbar.winfo_manager(), '')

        view.disable_word_wrap()
        self.assertEqual(view.x_scrollbar.winfo_manager(), 'pack')
        view.enable_word_wrap()
        self.assertEqual(view.x_scrollbar.winfo_manager(), '')
        self.assertEqual(str(view.text.cget('wrap')), 'word')
