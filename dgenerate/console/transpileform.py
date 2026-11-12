# Copyright (c) 2026, Teriks
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

"""File menu dialog that exports the editor script as Python."""

import json
import pathlib
import subprocess
import threading
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.resources as _resources

_RELAXED = 'transpile_relaxed'
_FORMAT = 'transpile_format'
_COMMENTS = 'transpile_comments'


def transpile_settings_path():
    """The console settings file that also stores the transpile dialog choices."""
    return pathlib.Path.home() / '.dgenerate' / 'console_settings.json'


def load_transpile_preferences():
    """The last Strict, Format, and Comments choices, or the defaults."""
    preferences = {'relaxed': False, 'format': False, 'comments': False}
    try:
        with transpile_settings_path().open(encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return preferences
    if not isinstance(data, dict):
        return preferences
    for key, name in ((_RELAXED, 'relaxed'), (_FORMAT, 'format'), (_COMMENTS, 'comments')):
        value = data.get(key)
        if isinstance(value, bool):
            preferences[name] = value
    return preferences


def save_transpile_preferences(relaxed, format_output, comments):
    """Store the transpile dialog choices without dropping the other console settings."""
    path = transpile_settings_path()
    try:
        with path.open(encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[_RELAXED] = bool(relaxed)
    data[_FORMAT] = bool(format_output)
    data[_COMMENTS] = bool(comments)
    try:
        path.parent.mkdir(exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(data, handle)
    except OSError:
        pass


def transpile_command(executable, relaxed, format_output=False, comments=False):
    """Arguments for ``dgenerate --sub-command transpile`` reading stdin.

    A sub-command keeps a pipe for itself, so ``--stdin`` reads the editor text.
    Comments are off unless ``comments`` is true, matching the command default.
    """
    command = [executable, '--sub-command', 'transpile']
    if relaxed:
        command.append('--relaxed')
    if format_output:
        command.append('--format')
    if comments:
        command.append('--comments')
    command.append('--stdin')
    return command


class TranspileDialog:
    """Explain the transpile modes and write the current editor script to a file."""

    def __init__(self, parent, get_script, executable):
        self._parent = parent
        self._get_script = get_script
        self._executable = executable
        self._exporting = False

        self.top = tk.Toplevel(parent)
        self.top.title('Transpile')
        self.top.transient(parent)
        self.top.resizable(False, False)
        _resources.set_window_icon(self.top)

        frame = ttk.Frame(self.top, padding=12)
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(
            frame,
            wraplength=460,
            justify=tk.LEFT,
            text=(
                'Write the script in the editor as Python that runs the same '
                'actions through dgenerate. Translating does not load a diffusion model.'
            ),
        ).grid(row=0, column=0, sticky='w')

        preferences = load_transpile_preferences()
        self._relaxed = tk.BooleanVar(value=preferences['relaxed'])
        self._format = tk.BooleanVar(value=preferences['format'])
        self._comments = tk.BooleanVar(value=preferences['comments'])
        for variable in (self._relaxed, self._format, self._comments):
            variable.trace_add('write', lambda *_args: self._save_preferences())
        modes = ttk.LabelFrame(frame, text='Output', padding=8)
        modes.grid(row=1, column=0, sticky='ew', pady=(12, 0))

        ttk.Radiobutton(
            modes, text='Strict', variable=self._relaxed, value=False,
        ).grid(row=0, column=0, sticky='w')
        ttk.Label(
            modes,
            wraplength=420,
            justify=tk.LEFT,
            text=(
                'Keep the config interpreter\'s line numbers and error text. '
                'A failure is reported on the same line as the config.'
            ),
        ).grid(row=1, column=0, sticky='w', padx=(22, 0), pady=(0, 8))

        ttk.Radiobutton(
            modes, text='Relaxed', variable=self._relaxed, value=True,
        ).grid(row=2, column=0, sticky='w')
        ttk.Label(
            modes,
            wraplength=420,
            justify=tk.LEFT,
            text=(
                'Write the same actions without line numbers or the interpreter\'s '
                'error bookkeeping. The script is easier to read and edit.'
            ),
        ).grid(row=3, column=0, sticky='w', padx=(22, 0), pady=(0, 8))

        ttk.Checkbutton(
            modes, text='Format', variable=self._format,
        ).grid(row=4, column=0, sticky='w')
        ttk.Label(
            modes,
            wraplength=420,
            justify=tk.LEFT,
            text=(
                'Wrap long lists and calls, and leave a blank line between '
                'invocations. The script does the same work.'
            ),
        ).grid(row=5, column=0, sticky='w', padx=(22, 0), pady=(0, 8))

        ttk.Checkbutton(
            modes, text='Comments', variable=self._comments,
        ).grid(row=6, column=0, sticky='w')
        ttk.Label(
            modes,
            wraplength=420,
            justify=tk.LEFT,
            text=(
                'Keep # comments from the config, on their own line or at the '
                'end of a line.'
            ),
        ).grid(row=7, column=0, sticky='w', padx=(22, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky='e', pady=(12, 0))
        self._status = ttk.Label(buttons, text='')
        self._status.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(buttons, text='Cancel', command=self.close).pack(
            side=tk.RIGHT, padx=(6, 0))
        self._export = ttk.Button(buttons, text='Export', command=self._export_script)
        self._export.pack(side=tk.RIGHT)

        self.top.protocol('WM_DELETE_WINDOW', self.close)
        self.top.bind('<Escape>', lambda event: self.close())
        _appearance.install(self.top)
        self.top.update_idletasks()
        self.top.geometry(
            f'+{parent.winfo_rootx() + (parent.winfo_width() - self.top.winfo_width()) // 2}'
            f'+{parent.winfo_rooty() + (parent.winfo_height() - self.top.winfo_height()) // 2}')
        self.top.grab_set()

    def close(self):
        if self._exporting:
            return
        self._save_preferences()
        self.top.grab_release()
        self.top.destroy()

    def _save_preferences(self):
        save_transpile_preferences(
            self._relaxed.get(), self._format.get(), self._comments.get())

    def _export_script(self):
        script = self._get_script()
        if not script.strip():
            messagebox.showinfo(
                'Transpile', 'The editor is empty.', parent=self.top)
            return
        relaxed = self._relaxed.get()
        format_output = self._format.get()
        comments = self._comments.get()
        self._exporting = True
        self._export.state(['disabled'])
        self._status.configure(text='Transpiling...')
        threading.Thread(
            target=self._transpile,
            args=(script, relaxed, format_output, comments),
            daemon=True,
        ).start()

    def _transpile(self, script, relaxed, format_output, comments):
        try:
            completed = subprocess.run(
                transpile_command(self._executable, relaxed, format_output, comments),
                input=script,
                text=True,
                encoding='utf-8',
                capture_output=True,
            )
        except OSError as exc:
            self.top.after(0, lambda: self._show_error(str(exc)))
            return
        self.top.after(0, lambda: self._finish(completed))

    def _finish(self, completed):
        self._exporting = False
        self._export.state(['!disabled'])
        self._status.configure(text='')
        if not self.top.winfo_exists():
            return
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or '').strip()
            self._show_error(detail or f'Transpile exited with status {completed.returncode}.')
            return
        path = _filedialog.open_file_save_dialog(
            initialfile='script.py',
            defaultextension='.py',
            filetypes=[('Python Script', '*.py'), ('Text File', '*.txt')])
        if path is None:
            return
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(completed.stdout)

    def _show_error(self, detail):
        self._exporting = False
        if self.top.winfo_exists():
            self._export.state(['!disabled'])
            self._status.configure(text='')
            messagebox.showerror('Transpile', detail, parent=self.top)
