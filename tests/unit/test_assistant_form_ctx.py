import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None

from dgenerate.console.assistantform import _DEFAULT_CTX, _MAX_CTX, ctx_value


class TestContextValue(unittest.TestCase):
    def test_accepts_a_native_window(self):
        self.assertEqual(ctx_value('65536'), 65536)
        self.assertEqual(ctx_value(str(_MAX_CTX)), _MAX_CTX)

    def test_invalid_text_uses_the_default(self):
        self.assertEqual(ctx_value(''), _DEFAULT_CTX)
        self.assertEqual(ctx_value('nope'), _DEFAULT_CTX)
        self.assertEqual(ctx_value('0'), _DEFAULT_CTX)
        self.assertEqual(ctx_value(str(_MAX_CTX + 1)), _DEFAULT_CTX)


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
                    form._ctx_var.set('65536')
                    command = form._command()
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                finally:
                    form.destroy()
        self.assertEqual(command[command.index('--ctx') + 1], '65536')
        self.assertEqual(saved['assistant_ctx'], 65536)

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
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                finally:
                    form.destroy()
        self.assertEqual(saved['assistant_ctx'], _DEFAULT_CTX)
