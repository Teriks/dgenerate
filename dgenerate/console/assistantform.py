# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
import pathlib
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance
import typing

import psutil

import dgenerate.assistant.catalog as _catalog
import dgenerate.console.combobox as _combobox
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.scrolledtext as _scrolledtext
import dgenerate.console.terminaltext as _terminaltext
import dgenerate.console.themetext as _themetext
import dgenerate.console.util as _util
import dgenerate.files as _files
import dgenerate.textprocessing as _textprocessing

_dialog_state = _util.DialogState(save_position=True, save_size=True)

_POLL_MS = 100

_SETTINGS_CHAT = 'assistant_chat_model'
_SETTINGS_EMBED = 'assistant_embed_model'
_SETTINGS_THINK = 'assistant_think'
_SETTINGS_EFFORT = 'assistant_reasoning_effort'
_SETTINGS_EDIT = 'assistant_editor_in_context'
_EFFORT_LABELS = {'low': 'Low', 'medium': 'Medium', 'xhigh': 'Extra high'}

_PROMPT_HISTORY_FILE = 'assistant_prompt_history.json'
_DEFAULT_PROMPT_HISTORY = 500


def _settings_path() -> pathlib.Path:
    return pathlib.Path.home() / '.dgenerate' / 'console_settings.json'


def _saved_model(key: str, default: str) -> str:
    try:
        with _settings_path().open(encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, str) and value.strip() else default


def _saved_bool(key: str, default: bool = False) -> bool:
    try:
        with _settings_path().open(encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, bool) else default


def _remember(key: str, value) -> None:
    path = _settings_path()
    try:
        with path.open(encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[key] = value
    try:
        path.parent.mkdir(exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(data, handle)
    except OSError:
        pass


def _model_name(spec: str) -> str:
    return os.path.splitext(os.path.basename(spec))[0]


def _prompt_history_path() -> pathlib.Path:
    return pathlib.Path.home() / '.dgenerate' / _PROMPT_HISTORY_FILE


def _max_prompt_history_from_environ() -> int:
    raw = os.environ.get('DGENERATE_ASSISTANT_MAX_HISTORY', _DEFAULT_PROMPT_HISTORY)
    try:
        value = int(raw)
    except ValueError:
        print(
            f'WARNING: environmental variable DGENERATE_ASSISTANT_MAX_HISTORY '
            f'set to invalid value: "{raw}", defaulting to {_DEFAULT_PROMPT_HISTORY}',
            file=sys.stderr)
        return _DEFAULT_PROMPT_HISTORY
    if value < 0:
        print(
            f'WARNING: environmental variable DGENERATE_ASSISTANT_MAX_HISTORY '
            f'set to invalid value: "{raw}", defaulting to {_DEFAULT_PROMPT_HISTORY}. '
            f'Value must be greater than or equal to "0"',
            file=sys.stderr)
        return _DEFAULT_PROMPT_HISTORY
    return value


def _request_intro_task(editing: bool) -> str:
    return (
        'Describe the change to make to the config in the editor.'
        if editing else
        'Describe what the config script should do.')


_INTRO_HISTORY = (
    'Previous prompts are saved, use the up and down arrow keys '
    'on the first and last line to recall them.')


class _AssistantForm(tk.Toplevel):
    """
    Runs ``dgenerate --sub-command assistant`` on a plain language request
    and hands the resulting config to ``populate``.
    """

    def __init__(self,
                 populate: typing.Callable[[str], None],
                 dgenerate_exe: str,
                 get_cwd: typing.Callable[[], str | None],
                 get_offline: typing.Callable[[], bool],
                 get_editor: typing.Callable[[], str],
                 master=None,
                 position: tuple[int, int] = None,
                 size: tuple[int, int] = None):
        super().__init__(master)
        self.title('Generate Code')
        self.configure(padx=5, pady=5)
        self.transient(master)

        self._populate = populate
        self._dgenerate_exe = dgenerate_exe
        self._get_cwd = get_cwd
        self._get_offline = get_offline
        self._get_editor = get_editor
        self._edit_file: str | None = None

        self._process: subprocess.Popen | None = None
        self._stderr_queue: queue.Queue = queue.Queue()
        self._stdout_chunks: list[bytes] = []
        self._readers: list[threading.Thread] = []

        self._max_prompt_history = _max_prompt_history_from_environ()
        self._prompt_history: list[str] = []
        self._prompt_history_index = 0
        self._prompt_history_draft = ''
        self._load_prompt_history()

        self._create_widgets()
        self._theme_text_boxes()
        _themetext.listen(self._theme_text_boxes)

        self.minsize(560, 420)
        _appearance.install(self)
        _util.position_toplevel(master, self, size=size if size else (720, 520), position=position)

        self._request_text.text.focus_set()

    def _create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._intro = ttk.Frame(self)
        self._intro.grid(row=0, column=0, sticky=tk.EW)
        self._intro.grid_columnconfigure(0, weight=1)
        self._intro_task = ttk.Label(
            self._intro, anchor=tk.W, justify=tk.LEFT,
            text=_request_intro_task(False))
        self._intro_task.grid(row=0, column=0, sticky=tk.EW)
        self._intro_history = ttk.Label(
            self._intro, anchor=tk.W, justify=tk.LEFT, text=_INTRO_HISTORY)
        self._intro_history.grid(row=1, column=0, sticky=tk.EW)
        self.bind('<Configure>', self._fit_intro, add='+')

        self._request_text = _scrolledtext.ScrolledText(self)
        self._request_text.text.configure(height=6)
        self._request_text.grid(row=1, column=0, sticky=tk.NSEW, pady=(4, 8))
        self._request_text.text.bind('<Control-Return>', lambda e: (self._generate(), 'break')[1])
        self._request_text.text.bind('<Up>', self._show_previous_prompt)
        self._request_text.text.bind('<Down>', self._show_next_prompt)
        self._request_context_menu = self._create_request_context_menu()
        self._request_text.text.bind(
            '<Button-3>',
            lambda e: self._show_request_context_menu(e))
        if platform.system() == 'Darwin':
            self._request_text.text.bind(
                '<Control-x>', lambda e: (self._request_cut(), 'break')[1])
            self._request_text.text.bind(
                '<Control-c>', lambda e: (self._request_copy(), 'break')[1])
            self._request_text.text.bind(
                '<Control-v>', lambda e: (self._request_paste(), 'break')[1])
            self._request_text.text.bind(
                '<Control-a>', lambda e: (self._request_select_all(), 'break')[1])

        options = ttk.LabelFrame(self, text='Options', padding=(8, 6))
        options.grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))
        options.grid_columnconfigure(1, weight=1)

        ttk.Label(options, text='Chat model').grid(row=0, column=0, sticky=tk.E, padx=(0, 8), pady=2)

        self._model_specs: list[str] = []
        self._model_labels: list[str] = []
        for spec, size, note in _catalog.CHAT_MODELS:
            self._add_model(spec, f'{_model_name(spec)}, {size:.1f} GB, {note}')

        self._model_var = tk.StringVar()
        self._model_combo = _combobox.ComboBox(options, textvariable=self._model_var,
                                               values=self._model_labels)
        self._model_combo.grid(row=0, column=1, sticky=tk.EW, pady=2)

        chosen = _saved_model(_SETTINGS_CHAT, _catalog.DEFAULT_CHAT_MODEL)
        if chosen not in self._model_specs:
            chosen = _catalog.DEFAULT_CHAT_MODEL
        self._model_var.set(self._model_labels[self._model_specs.index(chosen)])
        self._model_var.trace_add('write', lambda *_: _remember(_SETTINGS_CHAT, self._selected_model()))

        ttk.Label(options, text='Embedding model').grid(row=1, column=0, sticky=tk.E, padx=(0, 8), pady=2)

        self._embed_specs: list[str] = []
        self._embed_labels: list[str] = []
        for spec, size, note in _catalog.EMBED_MODELS:
            self._add_embed_model(spec, f'{_model_name(spec)}, {size:.1f} GB, {note}')

        self._embed_var = tk.StringVar()
        self._embed_combo = _combobox.ComboBox(options, textvariable=self._embed_var,
                                               values=self._embed_labels)
        self._embed_combo.grid(row=1, column=1, sticky=tk.EW, pady=2)

        embed_chosen = _saved_model(_SETTINGS_EMBED, _catalog.DEFAULT_EMBED_MODEL)
        if embed_chosen not in self._embed_specs:
            embed_chosen = _catalog.DEFAULT_EMBED_MODEL
        self._embed_var.set(self._embed_labels[self._embed_specs.index(embed_chosen)])
        self._embed_var.trace_add('write', lambda *_: _remember(_SETTINGS_EMBED, self._selected_embed_model()))

        reason = ttk.Frame(options)
        reason.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        self._think_var = tk.BooleanVar(value=_saved_bool(_SETTINGS_THINK))
        self._think_check = ttk.Checkbutton(
            reason, text='Reason before answering (slower)', variable=self._think_var)
        self._think_check.grid(row=0, column=0, sticky=tk.W)
        self._think_var.trace_add('write', self._remember_think)
        ttk.Label(reason, text='Effort').grid(row=0, column=1, sticky=tk.E, padx=(18, 6))
        self._effort_var = tk.StringVar()
        self._effort_combo = _combobox.ComboBox(
            reason, textvariable=self._effort_var, width=12,
            values=[_EFFORT_LABELS[name] for name in _catalog.REASONING_EFFORTS])
        self._effort_combo.grid(row=0, column=2, sticky=tk.W)
        saved_effort = _saved_model(_SETTINGS_EFFORT, _catalog.DEFAULT_REASONING_EFFORT)
        if saved_effort not in _catalog.REASONING_EFFORTS:
            saved_effort = _catalog.DEFAULT_REASONING_EFFORT
        self._effort_var.set(_EFFORT_LABELS[saved_effort])
        self._effort_var.trace_add('write', lambda *_: _remember(_SETTINGS_EFFORT, self._selected_effort()))
        self._set_effort_enabled()

        self._edit_var = tk.BooleanVar(value=_saved_bool(_SETTINGS_EDIT))
        self._edit_check = ttk.Checkbutton(
            options, text='Editor in context (edit mode)', variable=self._edit_var)
        self._edit_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        self._edit_var.trace_add('write', self._sync_edit_mode)

        ttk.Label(self, text='Progress', anchor=tk.W).grid(row=3, column=0, sticky=tk.EW)
        self._log_text = _scrolledtext.ScrolledText(self)
        self._log_text.disable_word_wrap()
        self._log_text.text.configure(height=8, state=tk.DISABLED)
        self._log_text.grid(row=4, column=0, sticky=tk.NSEW, pady=(2, 8))
        self._log_terminal = _terminaltext.TerminalText(self._log_text.text)

        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0)
        self._generate_button = ttk.Button(buttons, text='Generate', command=self._generate)
        self._generate_button.pack(side=tk.LEFT, padx=5)
        self._cancel_button = ttk.Button(buttons, text='Cancel', command=self._cancel, state=tk.DISABLED)
        self._cancel_button.pack(side=tk.LEFT, padx=5)
        self._sync_edit_mode()

    def _fit_intro(self, event):
        if event.widget is self:
            wrap = max(event.width - 24, 200)
            self._intro_task.configure(wraplength=wrap)
            self._intro_history.configure(wraplength=wrap)

    def _load_prompt_history(self):
        if self._max_prompt_history == 0:
            self._prompt_history_index = 0
            return
        history_path = _prompt_history_path()
        try:
            with history_path.open(encoding='utf-8') as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self._prompt_history = []
            self._prompt_history_index = 0
            return
        if not isinstance(loaded, list):
            self._prompt_history = []
            self._prompt_history_index = 0
            return
        self._prompt_history = [
            item for item in loaded
            if isinstance(item, str)
        ][-self._max_prompt_history:]
        self._prompt_history_index = len(self._prompt_history)

    def _save_prompt_history(self):
        if self._max_prompt_history == 0:
            return
        history_path = _prompt_history_path()
        try:
            history_path.parent.mkdir(exist_ok=True)
            with history_path.open('w', encoding='utf-8') as handle:
                json.dump(self._prompt_history[-self._max_prompt_history:], handle)
        except OSError:
            pass

    def _append_prompt_history(self, prompt: str):
        if self._max_prompt_history == 0:
            return
        if self._prompt_history:
            if self._prompt_history[-1] != prompt:
                self._prompt_history.append(prompt)
        else:
            self._prompt_history.append(prompt)
        self._prompt_history_index = len(self._prompt_history)
        self._save_prompt_history()

    def _browsing_prompt_history(self) -> bool:
        return self._prompt_history_index < len(self._prompt_history)

    def _prompt_on_first_line(self) -> bool:
        return int(self._request_text.text.index('insert').split('.')[0]) == 1

    def _prompt_on_last_line(self) -> bool:
        insert_line = int(self._request_text.text.index('insert').split('.')[0])
        end_line = int(self._request_text.text.index('end-1c').split('.')[0])
        return insert_line == end_line

    def _set_request_prompt(self, text: str):
        text_widget = self._request_text.text
        text_widget.replace('1.0', tk.END, text)
        text_widget.mark_set('insert', tk.END)
        text_widget.see(tk.END)

    def _show_previous_prompt(self, event):
        if event.state & 0x0004:
            return
        if str(self._request_text.text.cget('state')) == tk.DISABLED:
            return 'break'
        if not self._browsing_prompt_history() and not self._prompt_on_first_line():
            return
        if self._prompt_history_index == len(self._prompt_history):
            self._prompt_history_draft = self._request_text.text.get('1.0', 'end-1c')
        if self._prompt_history_index > 0:
            self._prompt_history_index -= 1
            self._set_request_prompt(self._prompt_history[self._prompt_history_index])
        return 'break'

    def _show_next_prompt(self, event):
        if event.state & 0x0004:
            return
        if str(self._request_text.text.cget('state')) == tk.DISABLED:
            return 'break'
        if not self._browsing_prompt_history() and not self._prompt_on_last_line():
            return
        if self._prompt_history_index < len(self._prompt_history) - 1:
            self._prompt_history_index += 1
            self._set_request_prompt(self._prompt_history[self._prompt_history_index])
        elif self._prompt_history_index == len(self._prompt_history) - 1:
            self._prompt_history_index = len(self._prompt_history)
            self._set_request_prompt(self._prompt_history_draft)
        return 'break'

    def _request_initial_dir(self) -> str:
        return self._get_cwd() or os.getcwd()

    def _insert_or_replace_request_text(self, text: str):
        text_widget = self._request_text.text
        try:
            selection_start = text_widget.index('sel.first')
            selection_end = text_widget.index('sel.last')
        except tk.TclError:
            selection_start = selection_end = None
        if selection_start and selection_end:
            text_widget.replace(selection_start, selection_end, text)
        else:
            text_widget.insert('insert', text)

    def _request_insert_directory_path(self):
        directory = _filedialog.open_directory_dialog(initialdir=self._request_initial_dir())
        if directory is None:
            return
        self._insert_or_replace_request_text(_textprocessing.shell_quote(directory))

    def _request_insert_file_path(self):
        path = _filedialog.open_file_dialog(initialdir=self._request_initial_dir())
        if path is None:
            return
        self._insert_or_replace_request_text(_textprocessing.shell_quote(path))

    def _request_insert_file_path_save(self):
        path = _filedialog.open_file_save_dialog(initialdir=self._request_initial_dir())
        if path is None:
            return
        self._insert_or_replace_request_text(_textprocessing.shell_quote(path))

    def _request_cut(self):
        self._request_text.text.event_generate('<<Cut>>')

    def _request_copy(self):
        self._request_text.text.event_generate('<<Copy>>')

    def _request_paste(self):
        self._request_text.text.event_generate('<<Paste>>')

    def _request_delete(self):
        try:
            self._request_text.text.delete('sel.first', 'sel.last')
        except tk.TclError:
            pass

    def _request_select_all(self):
        self._request_text.text.event_generate('<<SelectAll>>')

    def _update_request_context_menu(self):
        text_widget = self._request_text.text
        disabled = str(text_widget.cget('state')) == tk.DISABLED
        try:
            text_widget.index('sel.first')
            has_selection = not disabled
        except tk.TclError:
            has_selection = False
        has_clipboard = False
        if not disabled:
            try:
                self.clipboard_get()
                has_clipboard = True
            except tk.TclError:
                pass
        for label, enabled in (
            ('Cut', has_selection),
            ('Copy', has_selection),
            ('Paste', has_clipboard),
            ('Delete', has_selection),
        ):
            self._request_context_menu.entryconfigure(
                label, state=tk.NORMAL if enabled else tk.DISABLED)
        self._request_context_menu.entryconfigure(
            'Select All', state=tk.NORMAL if not disabled else tk.DISABLED)
        self._request_context_menu.entryconfigure(
            'Insert Path', state=tk.NORMAL if not disabled else tk.DISABLED)

    def _create_request_context_menu(self) -> tk.Menu:
        menu = tk.Menu(self._request_text, tearoff=0)
        menu.add_command(label='Cut', accelerator='Ctrl+X', command=self._request_cut)
        menu.add_command(label='Copy', accelerator='Ctrl+C', command=self._request_copy)
        menu.add_command(label='Paste', accelerator='Ctrl+V', command=self._request_paste)
        menu.add_command(label='Delete', accelerator='DEL', command=self._request_delete)
        menu.add_command(label='Select All', accelerator='Ctrl+A', command=self._request_select_all)
        menu.add_separator()
        path_submenu = tk.Menu(menu, tearoff=0)
        path_submenu.add_command(
            label='File Path (Open)', command=self._request_insert_file_path)
        path_submenu.add_command(
            label='File Path (Save)', command=self._request_insert_file_path_save)
        path_submenu.add_command(
            label='Directory Path', command=self._request_insert_directory_path)
        menu.add_cascade(label='Insert Path', menu=path_submenu)
        menu.configure(postcommand=self._update_request_context_menu)
        return menu

    def _show_request_context_menu(self, event):
        self._request_text.text.focus_set()
        try:
            self._request_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._request_context_menu.grab_release()

    def _theme_text_boxes(self):
        if not self.winfo_exists():
            return
        for text in (self._request_text.text, self._log_text.text):
            try:
                text.configure(insertbackground=text.cget('fg'))
            except tk.TclError:
                return

    def _add_embed_model(self, spec: str, label: str):
        if _catalog.is_downloaded(spec):
            label += ' (downloaded)'
        self._embed_specs.append(spec)
        self._embed_labels.append(label)

    def _selected_embed_model(self) -> str:
        label = self._embed_var.get()
        if label in self._embed_labels:
            return self._embed_specs[self._embed_labels.index(label)]
        return _catalog.DEFAULT_EMBED_MODEL

    def _add_model(self, spec: str, label: str):
        if _catalog.is_downloaded(spec):
            label += ' (downloaded)'
        self._model_specs.append(spec)
        self._model_labels.append(label)

    def _selected_model(self) -> str:
        label = self._model_var.get()
        if label in self._model_labels:
            return self._model_specs[self._model_labels.index(label)]
        return _catalog.DEFAULT_CHAT_MODEL

    def _log(self, line: str):
        if not line.endswith('\n'):
            line += '\n'
        self._write_status(line, stream='status')

    def _write_status(self, data: str, stream: str = 'stderr'):
        self._log_text.text.configure(state=tk.NORMAL)
        self._log_terminal.write(data, stream=stream)
        self._log_text.text.see(tk.END)
        self._log_text.text.configure(state=tk.DISABLED)

    def _clear_log(self):
        self._log_text.text.configure(state=tk.NORMAL)
        self._log_text.text.delete('1.0', tk.END)
        self._log_terminal.reset()
        self._log_text.text.configure(state=tk.DISABLED)

    def _set_running(self, running: bool):
        self._generate_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self._cancel_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self._request_text.text.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _sync_edit_mode(self, *_):
        editing = bool(self._edit_var.get())
        _remember(_SETTINGS_EDIT, editing)
        self._intro_task.configure(text=_request_intro_task(editing))
        if self._process is None:
            self._generate_button.configure(text='Edit' if editing else 'Generate')

    def _command(self, edit_path: str | None = None) -> list[str]:
        # Without --no-stdin, dgenerate runs piped stdin as a config instead of leaving it for the assistant.
        command = [self._dgenerate_exe, '--no-stdin']
        if self._get_offline():
            command.append('--offline-mode')
        command += ['--sub-command', 'assistant', '--model', self._selected_model(),
                    '--embed-model', self._selected_embed_model()]
        if edit_path:
            command += ['--edit', edit_path]
        if self._think_var.get():
            command += ['--think', '--reasoning-effort', self._selected_effort()]
        return command

    def _clear_edit_file(self):
        path = self._edit_file
        self._edit_file = None
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _remember_think(self, *_):
        _remember(_SETTINGS_THINK, bool(self._think_var.get()))
        self._set_effort_enabled()

    def _set_effort_enabled(self):
        self._effort_combo.configure(state='readonly' if self._think_var.get() else tk.DISABLED)

    def _selected_effort(self) -> str:
        label = self._effort_var.get()
        for name, text in _EFFORT_LABELS.items():
            if text == label:
                return name
        return _catalog.DEFAULT_REASONING_EFFORT

    def _generate(self):
        if self._process is not None:
            return
        request = self._request_text.text.get('1.0', 'end-1c').strip()
        if not request:
            self._request_text.text.focus_set()
            return

        self._append_prompt_history(request)

        edit_path = None
        if self._edit_var.get():
            editor = self._get_editor().replace('\r\n', '\n').strip()
            if not editor:
                self._log('Edit mode is on, but the editor is empty.')
                self._request_text.text.focus_set()
                return
            fd, edit_path = tempfile.mkstemp(suffix='.dgen', prefix='dgenerate-edit-')
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(editor)
                if not editor.endswith('\n'):
                    handle.write('\n')
            self._edit_file = edit_path

        cwd = self._get_cwd() or os.getcwd()

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        env['COLUMNS'] = '80'
        # huggingface_hub hides tqdm unless stderr is a terminal. Do not use
        # TQDM_POSITION=-1 (tqdm treats it as nested position). Force via
        # DGENERATE_FORCE_TQDM instead (see tqdm_huggingface_hub_patch).
        env['DGENERATE_FORCE_TQDM'] = '1'

        kwargs = {}
        if platform.system() == 'Windows':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        self._clear_log()
        self._stdout_chunks = []
        self._stderr_queue = queue.Queue()

        try:
            # The request goes through stdin so no part of it is parsed as an option.
            self._process = subprocess.Popen(
                self._command(edit_path), cwd=cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        except OSError as e:
            self._clear_edit_file()
            self._log(f'Could not start dgenerate: {e}')
            return

        self._set_running(True)

        process = self._process
        process.stdin.write(request.encode('utf-8'))
        process.stdin.close()

        def read_stdout():
            self._stdout_chunks.append(process.stdout.read())

        def read_stderr():
            # Break on carriage return so a tqdm bar is delivered on each redraw, not only when it finishes.
            reader = _files.TerminalLineReader(process.stderr)
            while True:
                line = reader.readline()
                if not line:
                    break
                self._stderr_queue.put(line.decode('utf-8', errors='replace'))

        self._readers = [threading.Thread(target=read_stdout, daemon=True),
                         threading.Thread(target=read_stderr, daemon=True)]
        for reader in self._readers:
            reader.start()

        self.after(_POLL_MS, self._poll)

    def _drain_stderr(self):
        while True:
            try:
                self._write_status(self._stderr_queue.get_nowait())
            except queue.Empty:
                break

    def _poll(self):
        if self._process is None:
            return
        self._drain_stderr()
        if self._process.poll() is None or any(r.is_alive() for r in self._readers):
            self.after(_POLL_MS, self._poll)
            return

        self._drain_stderr()
        return_code = self._process.returncode
        self._process = None
        self._clear_edit_file()
        self._set_running(False)
        self._sync_edit_mode()

        config = b''.join(self._stdout_chunks).decode('utf-8', errors='replace').replace('\r\n', '\n')
        if return_code == 0 and config.strip():
            self._populate(config)
            self._log('The config script is in the input pane.')
        else:
            self._log(f'The assistant failed (return code {return_code}).')

    def _kill(self):
        if self._process is None:
            return
        try:
            parent = psutil.Process(self._process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        self._process = None

    def _cancel(self):
        self._kill()
        self._clear_edit_file()
        self._set_running(False)
        self._sync_edit_mode()
        self._log('Cancelled.')

    def destroy(self):
        _themetext.unlisten(self._theme_text_boxes)
        self._kill()
        self._clear_edit_file()
        super().destroy()


def request_config(master,
                   populate: typing.Callable[[str], None],
                   dgenerate_exe: str,
                   get_cwd: typing.Callable[[], str | None],
                   get_offline: typing.Callable[[], bool],
                   get_editor: typing.Callable[[], str]):
    """
    Open the assistant dialog, or focus it if it is already open.

    :param master: The console window.
    :param populate: Receives the generated config.
    :param dgenerate_exe: The dgenerate executable to run the assistant sub-command with.
    :param get_cwd: Returns the directory the config will run from.
    :param get_offline: Returns whether the console is in offline mode.
    :param get_editor: Returns the config currently in the editor.
    """
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_AssistantForm,
        state=_dialog_state,
        dialog_kwargs={'populate': populate,
                       'dgenerate_exe': dgenerate_exe,
                       'get_cwd': get_cwd,
                       'get_offline': get_offline,
                       'get_editor': get_editor}
    )
