import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class ValueEntryTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_one_value_stays_a_text_field(self):
        from dgenerate.console.textentry import TextEntry, value_entry

        variable = tk.StringVar(master=_ROOT, value='kept')
        widget = value_entry(_ROOT, variable, ['kept'])
        self.addCleanup(widget.destroy)
        self.assertIsInstance(widget, TextEntry)
        self.assertEqual(variable.get(), 'kept')

    def test_several_values_are_an_editable_dropdown(self):
        import tkinter.ttk as ttk

        from dgenerate.console.textentry import value_entry

        variable = tk.StringVar(master=_ROOT, value='q4')
        widget = value_entry(_ROOT, variable, ['q4', '', 'q4', 'q8'])
        self.addCleanup(widget.destroy)
        self.assertIsInstance(widget, ttk.Combobox)
        self.assertEqual(str(widget.cget('state')), 'normal')
        self.assertEqual(list(widget.cget('values')), ['q4', 'q8'])
        self.assertEqual(variable.get(), 'q4')

    def test_recipe_label_is_shown_and_the_value_is_kept(self):
        from dgenerate.console.textentry import value_entry

        q4 = 'https://huggingface.co/Teriks/Kandinsky-5.0-T2V-Pro-sft-5s-GGUF/blob/main/kandinsky5Pro_t2v_sft_5s-Q4_K_S.gguf'
        q8 = 'https://huggingface.co/Teriks/Kandinsky-5.0-T2V-Pro-sft-5s-GGUF/blob/main/kandinsky5Pro_t2v_sft_5s-Q8_0.gguf'
        variable = tk.StringVar(master=_ROOT, value=q4)
        widget = value_entry(_ROOT, variable, [
            {'label': 'Q4_K_S', 'value': q4},
            {'label': 'Q8_0', 'value': q8},
        ])
        self.addCleanup(widget.destroy)
        self.assertEqual(list(widget.cget('values')), ['Q4_K_S', 'Q8_0'])
        self.assertEqual(widget.get(), 'Q4_K_S')
        self.assertEqual(variable.get(), q4)

        widget.set('Q8_0')
        self.assertEqual(variable.get(), q8)

        variable.set(r'C:\models\custom.gguf')
        self.assertEqual(widget.get(), r'C:\models\custom.gguf')

        variable.set(q4)
        widget.selection_range(0, tk.END)
        widget.event_generate('<<Copy>>')
        self.assertEqual(widget.clipboard_get(), q4)

    def test_plain_suggestions_are_not_renamed(self):
        from dgenerate.console.textentry import value_entry

        variable = tk.StringVar(master=_ROOT, value='alpha')
        widget = value_entry(_ROOT, variable, ['alpha', 'beta'])
        self.addCleanup(widget.destroy)
        self.assertEqual(list(widget.cget('values')), ['alpha', 'beta'])
        self.assertEqual(widget.get(), 'alpha')

    def test_tip_shows_a_clipped_value_and_a_hidden_url(self):
        from dgenerate.console.textentry import TextEntry, _plain_tip_text, value_entry

        top = tk.Toplevel(_ROOT)
        top.overrideredirect(True)
        top.geometry('90x90+-2500+-2500')
        self.addCleanup(top.destroy)
        long = 'https://huggingface.co/org/repo/blob/main/model-Q4_K_S.gguf'
        entry = TextEntry(top)
        entry.pack(fill='x')
        entry.insert(0, long)
        top.update()
        self.assertEqual(_plain_tip_text(entry), long)
        entry.delete(0, tk.END)
        entry.insert(0, 'Q4')
        top.update()
        self.assertEqual(_plain_tip_text(entry), '')

        variable = tk.StringVar(master=top, value=long)
        labeled = value_entry(top, variable, [
            {'label': 'Q4_K_S', 'value': long},
            {'label': 'Q8_0', 'value': long + '.q8'},
        ])
        labeled.pack(fill='x')
        top.update()
        self.assertEqual(labeled._tip_text(), long)


if __name__ == '__main__':
    unittest.main()
