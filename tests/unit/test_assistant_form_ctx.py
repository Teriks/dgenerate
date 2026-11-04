import json
import tempfile
import tkinter.ttk as ttk
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None

from dgenerate.assistant.catalog import format_context_vram
from dgenerate.console.assistantform import _DEFAULT_CTX, _MAX_CTX, ctx_value


def _named_buttons(widget, names: set[str]) -> list:
    found = []
    for child in widget.winfo_children():
        if isinstance(child, ttk.Button) and str(child.cget('text')) in names:
            found.append(child)
        found.extend(_named_buttons(child, names))
    return found


class TestContextValue(unittest.TestCase):
    def test_accepts_a_native_window(self):
        self.assertEqual(ctx_value('65536'), 65536)
        self.assertEqual(ctx_value(str(_MAX_CTX)), _MAX_CTX)

    def test_invalid_text_uses_the_default(self):
        self.assertEqual(ctx_value(''), _DEFAULT_CTX)
        self.assertEqual(ctx_value('nope'), _DEFAULT_CTX)
        self.assertEqual(ctx_value('0'), _DEFAULT_CTX)
        self.assertEqual(ctx_value(str(_MAX_CTX + 1)), _DEFAULT_CTX)

    def test_context_vram_reads_in_whole_units(self):
        self.assertEqual(format_context_vram(1024), '+64 MiB')
        self.assertEqual(format_context_vram(_DEFAULT_CTX), '+2 GiB')
        self.assertEqual(format_context_vram(_MAX_CTX), '+16 GiB')


@unittest.skipIf(_ROOT is None, 'Tk is not available')
class TestGenerateCodeContextField(unittest.TestCase):
    def test_command_passes_the_field_as_ctx(self):
        from dgenerate.console.assistantform import _AssistantForm

        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / 'console_settings.json'
            with patch('dgenerate.console.assistantform._settings_path', return_value=settings):
                form = _AssistantForm(
                    populate=lambda _text: None,
                    dgenerate_exe='dgenerate',
                    get_cwd=lambda: None,
                    get_offline=lambda: False,
                    get_editor=lambda: '',
                    master=_ROOT)
                try:
                    form.withdraw()
                    form.update_idletasks()
                    self.assertEqual(form.minsize()[0], form.winfo_reqwidth())
                    edge = form.winfo_rootx() + form.winfo_reqwidth()
                    buttons = _named_buttons(form, {'Default', 'Max'})
                    self.assertEqual(sorted(b.cget('text') for b in buttons), ['Default', 'Max'])
                    for button in buttons:
                        self.assertLessEqual(button.winfo_rootx() + button.winfo_reqwidth(), edge)
                    form._ctx_var.set('65536')
                    command = form._command()
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                finally:
                    form.destroy()
        self.assertEqual(command[command.index('--ctx') + 1], '65536')
        self.assertEqual(saved['assistant_ctx'], 65536)

    def test_a_narrow_saved_size_is_widened_to_fit_the_buttons(self):
        from dgenerate.console.assistantform import _AssistantForm

        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / 'console_settings.json'
            with patch('dgenerate.console.assistantform._settings_path', return_value=settings):
                form = _AssistantForm(
                    populate=lambda _text: None,
                    dgenerate_exe='dgenerate',
                    get_cwd=lambda: None,
                    get_offline=lambda: False,
                    get_editor=lambda: '',
                    master=_ROOT,
                    size=(400, 420))
                try:
                    form.update_idletasks()
                    # A saved width of 400 cannot pull the minimum under the content.
                    self.assertGreaterEqual(form.minsize()[0], form.winfo_reqwidth())
                    self.assertGreater(form.minsize()[0], 400)
                    self.assertLess(form.minsize()[0], 720)
                finally:
                    form.destroy()

    def test_default_and_max_buttons_set_the_window(self):
        from dgenerate.console.assistantform import _AssistantForm

        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / 'console_settings.json'
            with patch('dgenerate.console.assistantform._settings_path', return_value=settings):
                form = _AssistantForm(
                    populate=lambda _text: None,
                    dgenerate_exe='dgenerate',
                    get_cwd=lambda: None,
                    get_offline=lambda: False,
                    get_editor=lambda: '',
                    master=_ROOT)
                try:
                    form.withdraw()
                    form._ctx_var.set('8192')
                    form._set_max_ctx()
                    self.assertEqual(form._ctx_var.get(), str(_MAX_CTX))
                    form._set_default_ctx()
                    self.assertEqual(form._ctx_var.get(), str(_DEFAULT_CTX))
                    self.assertEqual(form._ctx_vram_var.get(), '+2 GiB')
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                    form._ctx_var.set('1024')
                    self.assertEqual(form._ctx_vram_var.get(), '+64 MiB')
                    form._ctx_var.set(str(_MAX_CTX))
                    self.assertEqual(form._ctx_vram_var.get(), '+16 GiB')
                finally:
                    form.destroy()
        self.assertEqual(saved['assistant_ctx'], _DEFAULT_CTX)
