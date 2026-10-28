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
import pathlib
import platform
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import typing

import dgenerate.console.appearance as _appearance
import dgenerate.console.finddialog as _finddialog
import dgenerate.console.resources as _resources
from dgenerate.console.codeview import DgenerateCodeView

_editor_window: tk.Toplevel | None = None

_INIT_DGEN_PATH = pathlib.Path(pathlib.Path.home(), '.dgenerate', 'init.dgen')


def open_init_dgen_editor(
        parent: tk.Misc,
        *,
        theme_name: str,
        on_saved: typing.Callable[[], None] | None = None) -> None:
    """
    Open a window for editing ``~/.dgenerate/init.dgen``.

    :param parent: parent window (typically the console)
    :param theme_name: syntax highlighting theme name
    :param on_saved: called after a successful save (e.g. restart the shell)
    """
    global _editor_window

    if _editor_window is not None and _editor_window.winfo_exists():
        _editor_window.lift()
        _editor_window.focus_force()
        return

    init_path = _INIT_DGEN_PATH
    if not init_path.is_file():
        messagebox.showerror(
            'Startup Configuration',
            f'{init_path} does not exist yet.\n\n'
            'It is created when the shell process starts. '
            'Wait for the shell to finish starting, then try again.',
            parent=parent)
        return

    try:
        original_text = init_path.read_text(encoding='utf-8')
    except OSError as error:
        messagebox.showerror(
            'Startup Configuration',
            f'Could not read {init_path}:\n{error}',
            parent=parent)
        return

    dialog = tk.Toplevel(parent)
    _editor_window = dialog
    dialog.title('Startup Configuration (init.dgen)')
    dialog.transient(parent)
    dialog.minsize(520, 420)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(1, weight=1)
    _resources.set_window_icon(dialog)

    header = ttk.Frame(dialog, padding=(12, 12, 12, 0))
    header.grid(row=0, column=0, sticky='ew')

    ttk.Label(
        header,
        text=(
            'This file runs when dgenerate starts. Uncomment or add '
            '\\env lines to set HF_TOKEN, CIVIT_AI_TOKEN, cache paths, and more.'
        ),
        wraplength=560,
        justify=tk.LEFT,
    ).pack(anchor=tk.W)

    ttk.Label(
        header,
        text=str(init_path),
        font=('', 9),
    ).pack(anchor=tk.W, pady=(6, 0))

    editor_frame = ttk.Frame(dialog, padding=(12, 0, 12, 0))
    editor_frame.grid(row=1, column=0, sticky='nsew')
    editor_frame.columnconfigure(0, weight=1)
    editor_frame.rowconfigure(0, weight=1)

    code_view = DgenerateCodeView(editor_frame, undo=True)
    code_view.grid(row=0, column=0, sticky='nsew')
    code_view.enable_line_numbers()
    code_view.disable_word_wrap()
    code_view.text.insert('1.0', original_text)
    code_view.text.edit_reset()
    code_view.set_theme(theme_name)
    code_view.text.mark_set('insert', '1.0')
    code_view.text.focus_set()

    code_view.text.bind(
        '<Control-f>',
        lambda event: _finddialog.open_find_dialog(
            dialog,
            'Find In Startup Configuration',
            code_view.text))

    context_menu = tk.Menu(code_view.text, tearoff=0)
    context_menu.add_command(
        label='Cut',
        accelerator='Ctrl+X',
        command=code_view.gen_cut_event)
    context_menu.add_command(
        label='Copy',
        accelerator='Ctrl+C',
        command=code_view.gen_copy_event)
    context_menu.add_command(
        label='Paste',
        accelerator='Ctrl+V',
        command=code_view.gen_paste_event)
    context_menu.add_separator()
    context_menu.add_command(
        label='Select All',
        accelerator='Ctrl+A',
        command=code_view.gen_selectall_event)

    code_view.text.bind(
        '<Button-3>',
        lambda event: context_menu.tk_popup(event.x_root, event.y_root))

    if platform.system() == 'Darwin':
        code_view.text.bind('<Control-x>', lambda event: code_view.gen_cut_event())
        code_view.text.bind('<Control-c>', lambda event: code_view.gen_copy_event())
        code_view.text.bind('<Control-v>', lambda event: code_view.gen_paste_event())
        code_view.text.bind(
            '<Control-a>',
            lambda event: (code_view.gen_selectall_event(), 'break')[1])

    button_frame = ttk.Frame(dialog, padding=(12, 0, 12, 12))

    def current_text() -> str:
        return code_view.text.get('1.0', 'end-1c')

    def is_dirty() -> bool:
        return current_text() != original_text

    def close_dialog(force: bool = False) -> None:
        global _editor_window
        if not force and is_dirty():
            choice = messagebox.askyesnocancel(
                'Startup Configuration',
                'Save changes before closing?',
                parent=dialog)
            if choice is None:
                return
            if choice:
                if not persist_changes(restart_shell=True):
                    return
        dialog.grab_release()
        dialog.destroy()
        _editor_window = None

    def persist_changes(*, restart_shell: bool) -> bool:
        text = current_text()
        try:
            init_path.write_text(text, encoding='utf-8')
        except OSError as error:
            messagebox.showerror(
                'Startup Configuration',
                f'Could not save {init_path}:\n{error}',
                parent=dialog)
            return False

        nonlocal original_text
        original_text = text
        code_view.text.edit_reset()
        if restart_shell and on_saved is not None:
            on_saved()
        return True

    def save_and_apply() -> None:
        if persist_changes(restart_shell=True):
            close_dialog(force=True)

    ttk.Button(
        button_frame,
        text='Cancel',
        command=lambda: close_dialog(force=False)).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(
        button_frame,
        text='Save',
        command=save_and_apply).pack(side=tk.RIGHT)

    button_frame.grid(row=2, column=0, sticky='ew')

    dialog.protocol('WM_DELETE_WINDOW', lambda: close_dialog(force=False))
    dialog.bind('<Control-s>', lambda event: save_and_apply() or 'break')
    dialog.bind('<Escape>', lambda event: close_dialog(force=False))

    _appearance.install(dialog)
    dialog.update_idletasks()
    width = min(680, max(560, parent.winfo_width() * 3 // 5))
    height = min(480, max(420, parent.winfo_height() * 2 // 5))
    dialog.geometry(
        f'{width}x{height}+'
        f'{parent.winfo_rootx() + (parent.winfo_width() - width) // 2}+'
        f'{parent.winfo_rooty() + (parent.winfo_height() - height) // 2}')
    dialog.grab_set()
