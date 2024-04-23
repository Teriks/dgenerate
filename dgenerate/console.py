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
import pathlib
import queue
import subprocess
import time
import threading
from tkinter import ttk

import tkinter as tk
import tkinter.filedialog
import tkinter.scrolledtext
import os


class _ScrolledText(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.text = tk.Text(self, wrap='word')

        self.y_scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.text.yview)
        self.y_scrollbar.pack(side='right', fill='y')

        self.x_scrollbar = ttk.Scrollbar(self, orient='horizontal', command=self.text.xview)
        self.x_scrollbar.pack(side='bottom', fill='x')

        self.text['yscrollcommand'] = self.y_scrollbar.set
        self.text['xscrollcommand'] = self.x_scrollbar.set

        self.text.pack(side='left', fill='both', expand=True)

    def enable_word_wrap(self):
        self.text.config(wrap='word')

    def disable_word_wrap(self):
        self.text.config(wrap='none')


class _DgenerateConsole(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Dgenerate Console')
        self.geometry('800x800')

        # Create the top menu frame

        top_menu_frame = tk.Frame(self)
        top_menu_frame.pack(fill='x', padx=(0, 5), pady=(5, 5))

        self._kill_button = tk.Button(top_menu_frame,
                                      text='Kill (ctrl-c)',
                                      command=self._kill_sub_process)
        self._kill_button.pack(side='right', anchor='e')

        self._paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self._paned_window.pack(fill=tk.BOTH, expand=True)

        # Create the Multi-line Input Checkbutton

        self._multi_line_input_check_var = tk.BooleanVar(value=False)
        self._multi_line_input_checkbox = tk.Checkbutton(
            top_menu_frame,
            text='Multi line input (insert key)',
            variable=self._multi_line_input_check_var)
        self._multi_line_input_checkbox.pack(side='right', anchor='e')

        # Create the word wrap checkbox

        self._word_wrap_input_check_var = tk.BooleanVar(value=False)
        self._word_wrap_input_check_var.set(True)
        self._word_wrap_input_checkbox = tk.Checkbutton(
            top_menu_frame,
            text='Word Wrap',
            command=self._toggle_input_wrap, variable=self._word_wrap_input_check_var)
        self._word_wrap_input_checkbox.pack(side='right', anchor='e')

        # Create the top text input pane

        self._input_entry = _ScrolledText(self._paned_window)

        self._input_entry.text.bind('<Return>', self._handle_input)
        self._input_entry.text.bind('<Up>', self._show_previous_command)
        self._input_entry.text.bind('<Down>', self._show_next_command)
        self._input_entry.text.bind('<Button-3>',
                                    lambda e: self._input_text_context.tk_popup(
                                        self.winfo_pointerx(), self.winfo_pointery()))
        self._input_entry.text.bind('<Return>', self._handle_input)
        self._input_entry.text.bind('<Insert>',
                                    lambda e:
                                    self._multi_line_input_check_var.set(
                                        not self._multi_line_input_check_var.get()))
        self._input_entry.text.bind('<Control-c>', lambda e: self._kill_sub_process())

        self._input_entry.text.focus_set()

        self._input_text_context = tk.Menu(self._input_entry, tearoff=0)
        self._input_text_context.add_command(label='Copy', command=self._copy_input_entry_selection)
        self._input_text_context.add_command(label='Paste', command=self._paste_input_entry)
        self._input_text_context.add_command(label='Load', command=self._load_input_entry_text)

        self._paned_window.add(self._input_entry)

        # Create the bottom pane

        self._output_text = _ScrolledText(self._paned_window)
        self._output_text.disable_word_wrap()

        self._output_text.text.bind('<Button-3>',
                                    lambda e: self._output_text_context.tk_popup(
                                        self.winfo_pointerx(), self.winfo_pointery()))

        self._output_text_context = tk.Menu(self._output_text, tearoff=0)
        self._output_text_context.add_command(label='Clear', command=self._clear_output_text)
        self._output_text_context.add_command(label='Copy Selection', command=self._copy_output_text_selection)
        self._output_text_context.add_command(label='Save Selection', command=self._save_output_text_selection)
        self._output_text_context.add_command(label='Save All', command=self._save_output_text)

        self._paned_window.add(self._output_text)

        # config

        self._command_history = []
        self._current_command_index = -1

        self._load_command_history()

        self._termination_lock = threading.Lock()

        self._start_dgenerate_process()

        self._threads = [
            threading.Thread(target=self._write_stdout)
        ]

        self._text_queue = queue.Queue()

        for t in self._threads:
            t.daemon = True
            t.start()

        # max output scroll-back history
        self._max_output_lines = 10000

        # output (stdout) refresh rate (ms)
        self._output_refresh_rate = 100

        # output lines per refresh
        self._output_lines_per_refresh = 30

        # max terminal command history
        self._max_command_history = 100

        self._write_output(
            'This console supports sending dgenerate configuration into a dgenerate\n'
            'interpreter process running in the background, it functions similarly to a terminal.\n\n'
            'Enter configuration above and hit enter to submit, use the insert key to enter\n'
            'and exit multi-line input mode, you must exit multi-line input mode to submit configuration.\n\n'
            'Command history is supported via the up and down arrow keys.\n'
            '============================================================\n\n')

        self._text_update()

    def _toggle_input_wrap(self):
        if self._word_wrap_input_check_var.get():
            self._input_entry.enable_word_wrap()

        else:
            self._input_entry.disable_word_wrap()

    def _restart_dgenerate_process(self):
        self._write_output('Restarting Shell Process...\n')
        self._start_dgenerate_process()
        self._write_output('Shell Process Started.\n')

    def _kill_sub_process(self):
        with self._termination_lock:
            self._sub_process.terminate()
            self._sub_process.communicate()
            self._write_output(
                f'\nShell Process Terminated, Exit Code: {self._sub_process.poll()}\n')
            self._restart_dgenerate_process()

    def _start_dgenerate_process(self):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'

        self._sub_process = subprocess.Popen(
            ['dgenerate', '--server'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=env)

    def _load_input_entry_text(self):
        f = tkinter.filedialog.askopenfile(
            mode='r',
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        self._input_entry.text.delete('1.0', tk.END)
        self._input_entry.text.insert('1.0', f.read())

    def _copy_input_entry_selection(self):
        text = self._input_entry.selection_get()
        self.clipboard_clear()
        self.clipboard_append(text)

    def _paste_input_entry(self):
        try:
            start = self._input_entry.text.index(tk.SEL_FIRST)
            end = self._input_entry.text.index(tk.SEL_LAST)
            self._input_entry.text.delete(start, end)
        except tk.TclError:
            start = self._input_entry.text.index(tk.INSERT)

        clipboard_content = self.clipboard_get()
        self._input_entry.text.insert(start, clipboard_content)

    def _copy_output_text_selection(self):
        text = self._output_text.selection_get()
        self.clipboard_clear()
        self.clipboard_append(text)

    def _save_output_text(self):
        f = tkinter.filedialog.asksaveasfile(
            mode='w',
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return
        self._output_text.text.config(state=tk.NORMAL)
        f.write(self._output_text.get('1.0', tk.END))
        self._output_text.text.config(state=tk.DISABLED)
        f.close()

    def _save_output_text_selection(self):
        f = tkinter.filedialog.asksaveasfile(
            mode='w',
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return
        f.write(self._output_text.selection_get())
        f.close()

    def _clear_output_text(self):
        self._output_text.text.config(state=tk.NORMAL)
        self._output_text.text.delete('1.0', tk.END)
        self._output_text.text.config(state=tk.DISABLED)

    def _text_update(self):
        lines = 0

        self._output_text.text.config(state=tk.NORMAL)

        while True:
            if lines > self._output_lines_per_refresh:
                break
            try:
                text = self._text_queue.get_nowait()

                output_lines = int(self._output_text.text.index('end-1c').split('.')[0])

                if output_lines > self._max_output_lines:
                    self._output_text.text.delete('1.0', f'{output_lines - self._max_output_lines}.0')

                self._output_text.text.insert(tk.END, text)

                self._output_text.text.see(tk.END)

            except queue.Empty:
                break
            lines += 1

        self._output_text.text.config(state=tk.DISABLED)

        self.after(self._output_refresh_rate, self._text_update)

    def _write_output(self, text):
        self._text_queue.put(text)

    def _write_stdout(self):
        exit_message = True
        while True:
            with self._termination_lock:
                return_code = self._sub_process.poll()

            if return_code is None:
                exit_message = True
                self._write_output(self._sub_process.stdout.readline())
            elif exit_message:
                exit_message = False
                with self._termination_lock:
                    self._write_output(
                        f'\nShell Process Terminated, Exit Code: {return_code}\n')
                    self._restart_dgenerate_process()
            else:
                time.sleep(1)

    def _show_previous_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index > 0:
            self._current_command_index -= 1
            self._input_entry.text.delete('1.0', tk.END)
            self._input_entry.text.insert(tk.END, self._command_history[self._current_command_index])
            self._input_entry.text.see(tk.END)
            self._input_entry.text.mark_set(tk.INSERT, tk.END)
            return "break"

    def _show_next_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index < len(self._command_history) - 1:
            self._current_command_index += 1
            self._input_entry.text.delete('1.0', tk.END)
            self._input_entry.text.insert(tk.END, self._command_history[self._current_command_index])
            self._input_entry.text.see(tk.END)
            self._input_entry.text.mark_set(tk.INSERT, tk.END)
            return "break"

    def _load_command_history(self):
        history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        if history_path.exists():
            with history_path.open('r') as file:
                self._command_history = json.load(file)
                self._current_command_index = len(self._command_history)

    def _save_command_history(self):
        history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        with history_path.open('w') as file:
            json.dump(self._command_history[-self._max_command_history:], file)

    def _handle_input(self, event):
        if self._multi_line_input_check_var.get():
            return

        with self._termination_lock:
            if self._sub_process.poll() is not None:
                self._restart_dgenerate_process()

        user_input = self._input_entry.text.get('1.0', 'end-1c')

        self._input_entry.text.delete(1.0, tk.END)
        self._sub_process.stdin.write(user_input + '\n\n')

        self._command_history.append(user_input)

        self._current_command_index = len(self._command_history)
        self._save_command_history()

        self._sub_process.stdin.flush()

        return 'break'

    def destroy(self) -> None:
        self._sub_process.terminate()
        super().destroy()


def main():
    app = _DgenerateConsole()
    app.mainloop()
