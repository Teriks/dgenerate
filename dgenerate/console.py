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
import collections.abc
import json
import pathlib
import queue
import subprocess
import sys
import time
import threading

import tkinter as tk
import tkinter.font
import tkinter.filedialog
import tkinter.scrolledtext
import os
import typing

import psutil


class _ScrolledText(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.text = tk.Text(self, wrap='word')

        font = tkinter.font.Font(font=self.text['font'])
        self.text.config(tabs=font.measure(' ' * 4))

        self.y_scrollbar = tk.Scrollbar(self, orient='vertical', command=self.text.yview)
        self.y_scrollbar.pack(side='right', fill='y')

        self.x_scrollbar = tk.Scrollbar(self, orient='horizontal', command=self.text.xview)
        self.x_scrollbar.pack(side='bottom', fill='x')

        self.text['yscrollcommand'] = self.y_scrollbar.set
        self.text['xscrollcommand'] = self.x_scrollbar.set

        self.text.pack(side='left', fill='both', expand=True)

    def enable_word_wrap(self):
        self.text.config(wrap='word')

    def disable_word_wrap(self):
        self.text.config(wrap='none')


class _FindDialog(tk.Toplevel):
    def __init__(self, master, name, text_widget):
        super().__init__(master=master)
        self.master = master
        self.title(name)
        self.text_widget = text_widget
        self.transient(self.master)

        self.find_entry = tk.Entry(self)
        self.find_entry.pack(side='top', fill='x', padx=2, pady=2)

        find_frame = tk.Frame(self)
        find_frame.pack(side='top', fill='x')

        self.case_var = tk.IntVar()
        self.case_check = tk.Checkbutton(find_frame, text="Case Sensitive", variable=self.case_var)
        self.case_check.pack(side='left', padx=2, pady=2)

        self.previous_button = tk.Button(find_frame, text='Previous', command=self.find_previous)
        self.previous_button.pack(side='left', padx=2, pady=2)

        self.next_button = tk.Button(find_frame, text='Next', command=self.find_next)
        self.next_button.pack(side='left', padx=2, pady=2)

        self.last_find = None

        # self.geometry(f'{self.winfo_reqwidth()}x{self.winfo_reqheight()}')
        self.resizable(False, False)

        self.master.update_idletasks()
        self.update_idletasks()

        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        top_level_width = self.winfo_width()
        top_level_height = self.winfo_height()

        # Calculate position coordinates
        position_top = self.master.winfo_y() + window_height // 2 - top_level_height // 2
        position_left = self.master.winfo_x() + window_width // 2 - top_level_width // 2

        # Position the window
        self.geometry(f"+{position_left}+{position_top}")

        self.find_entry.focus_set()

    def find_next(self):
        start_idx = '1.0' if not self.last_find else self.last_find + '+1c'
        s = self.find_entry.get()
        idx = self.text_widget.search(s, start_idx, nocase=1 - self.case_var.get(), stopindex=tk.END)
        if idx:
            end_idx = f'{idx}+{len(s)}c'
            self.text_widget.tag_remove('found', '1.0', tk.END)
            self.text_widget.tag_add('found', idx, end_idx)
            self.text_widget.mark_set(tk.INSERT, end_idx)
            self.text_widget.see(idx)
            self.text_widget.tag_config('found', foreground='white', background='blue')
            self.last_find = idx

    def find_previous(self):
        if not self.last_find:
            return
        s = self.find_entry.get()
        idx = self.text_widget.search(s, '1.0', nocase=1 - self.case_var.get(), stopindex=self.last_find)
        if idx:
            end_idx = f'{idx}+{len(s)}c'
            self.text_widget.tag_remove('found', '1.0', tk.END)
            self.text_widget.tag_add('found', idx, end_idx)
            self.text_widget.mark_set(tk.INSERT, idx)
            self.text_widget.see(idx)
            self.text_widget.tag_config('found', foreground='white', background='blue')
            self.last_find = idx

    def destroy(self) -> None:
        try:
            self.text_widget.tag_remove('found', '1.0', tk.END)
        except tk.TclError:
            # main window already destroyed
            pass
        super().destroy()


class _DgenerateConsole(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Dgenerate Console')
        self.geometry('800x800')

        # Create top menu

        menu_bar = tk.Menu(self)

        options_menu = tk.Menu(menu_bar, tearoff=0)
        run_menu = tk.Menu(menu_bar, tearoff=0)

        run_menu.add_command(label='Run', command=self._run_input_text)
        run_menu.add_command(label='Kill (ctrl-c)', command=self._kill_sub_process)

        # Create the Multi-line Input Checkbutton

        self._multi_line_input_check_var = tk.BooleanVar(value=False)

        self._multi_line_input_check_var.trace_add('write',
                                                   lambda *args: self._input_text.text.configure(
                                                       insertbackground='red') if self._multi_line_input_check_var.get()
                                                   else self._input_text.text.configure(insertbackground='black'))

        options_menu.add_checkbutton(label='Multiline input (insert key)',
                                     variable=self._multi_line_input_check_var)

        # Create the word wrap output checkbox

        self._word_wrap_output_check_var = tk.BooleanVar(value=True)
        self._word_wrap_output_check_var.trace_add('write', self._toggle_output_wrap)

        options_menu.add_checkbutton(label='Word Wrap Output',
                                     variable=self._word_wrap_output_check_var)

        # Create the word wrap input checkbox

        self._word_wrap_input_check_var = tk.BooleanVar(value=True)
        self._word_wrap_input_check_var.trace_add('write', self._toggle_input_wrap)

        options_menu.add_checkbutton(label='Word Wrap Input',
                                     variable=self._word_wrap_input_check_var)

        # Create the auto scroll checkbox (scroll on input)
        self._auto_scroll_on_run_check_var = tk.BooleanVar(value=True)
        options_menu.add_checkbutton(label='Auto Scroll Output On Run',
                                     variable=self._auto_scroll_on_run_check_var)

        menu_bar.add_cascade(label="Run", menu=run_menu)
        menu_bar.add_cascade(label="Options", menu=options_menu)

        self._paned_window = tk.PanedWindow(self, orient=tk.VERTICAL, sashwidth=4, bg="#808080")
        self._paned_window.pack(fill=tk.BOTH, expand=True)

        self.config(menu=menu_bar)

        # Create the top text input pane

        self._input_text = _ScrolledText(self._paned_window)

        self._input_text.text.bind('<Return>',
                                   lambda e:
                                   None if self._multi_line_input_check_var.get()
                                   else self._run_input_text())

        self._input_text.text.bind('<Up>', self._show_previous_command)
        self._input_text.text.bind('<Down>', self._show_next_command)
        self._input_text.text.bind('<Button-3>',
                                   lambda e: self._input_text_context.tk_popup(
                                       self.winfo_pointerx(), self.winfo_pointery()))
        self._input_text.text.bind('<Insert>',
                                   lambda e:
                                   self._multi_line_input_check_var.set(
                                       not self._multi_line_input_check_var.get()))
        self._input_text.text.bind('<Control-f>',
                                   lambda e: self._open_find_dialog(
                                       'Find In Input',
                                       self._input_text.text))

        def bind_input_text_ctrl_c(event):
            if event.widget == self._input_text.text:
                self._input_text.text.bind('<Control-c>',
                                           lambda e: self._kill_sub_process())

        def unbind_input_text_ctrl_c(event):
            if event.widget == self._input_text.text:
                self._input_text.unbind('<Control-c>')

        self._input_text.text.bind('<FocusIn>', bind_input_text_ctrl_c)
        self._input_text.text.bind('<FocusOut>', unbind_input_text_ctrl_c)

        self._input_text.text.focus_set()

        self._input_text_context = tk.Menu(self._input_text, tearoff=0)

        self._input_text_context.add_command(label='Load', command=self._load_input_entry_text)
        self._input_text_context.add_command(label='Save', command=self._save_input_entry_text)
        self._input_text_context.add_separator()
        self._input_text_context.add_command(label='Find',
                                             command=lambda:
                                             self._open_find_dialog(
                                                 'Find In Input',
                                                 self._input_text.text))
        self._input_text_context.add_separator()
        self._input_text_context.add_command(label='Copy', command=self._copy_input_entry_selection)
        self._input_text_context.add_command(label='Paste', command=self._paste_input_entry)

        self._paned_window.add(self._input_text)

        # Create the bottom pane

        self._output_text = _ScrolledText(self._paned_window)

        self._output_text.text.bind('<Button-3>',
                                    lambda e: self._output_text_context.tk_popup(
                                        self.winfo_pointerx(), self.winfo_pointery()))
        self._output_text.text.bind('<Control-f>',
                                    lambda e: self._open_find_dialog(
                                        'Find In Output',
                                        self._output_text.text))

        self._output_text_context = tk.Menu(self._output_text, tearoff=0)

        self._output_text_context.add_command(label='Clear', command=self._clear_output_text)
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Find',
                                              command=lambda:
                                              self._open_find_dialog(
                                                  'Find In Output',
                                                  self._output_text.text))
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Copy', command=self._copy_output_text_selection)
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Save Selection', command=self._save_output_text_selection)
        self._output_text_context.add_command(label='Save All', command=self._save_output_text)

        self._paned_window.add(self._output_text)

        # config

        self._termination_lock = threading.Lock()

        self._cwd = os.getcwd()
        self._start_dgenerate_process()

        self._threads = [
            threading.Thread(target=self._read_sub_process_stream_thread,
                             args=(lambda p: p.stdout, self._write_stdout_output)),
            threading.Thread(target=self._read_sub_process_stream_thread,
                             args=(lambda p: p.stderr, self._write_stderr_output))
        ]

        self._output_text_queue = queue.Queue()

        self._find_dialog = None

        for t in self._threads:
            t.daemon = True
            t.start()

        self._update_cwd()

        # output lines per refresh
        self._output_lines_per_refresh = 30

        # output (stdout) refresh rate (ms)
        self._output_refresh_rate = 100

        # max output scroll-back history
        max_output_lines = os.environ.get('DGENERATE_CONSOLE_MAX_SCROLLBACK', 10000)
        try:
            self._max_output_lines = int(max_output_lines)
        except ValueError:
            self._max_output_lines = 10000
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_SCROLLBACK '
                f'set to invalid value: "{max_output_lines}", defaulting to 10000',
                file=sys.stderr)

        if self._max_output_lines < self._output_lines_per_refresh:
            self._max_output_lines = 10000
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_SCROLLBACK '
                f'set to invalid value: "{max_output_lines}", defaulting to 10000. '
                f'Value must be greater than or equal to "{self._output_lines_per_refresh}"',
                file=sys.stderr)

        # max terminal command history
        max_command_history = os.environ.get('DGENERATE_CONSOLE_MAX_HISTORY', 500)
        try:
            self._max_command_history = int(max_command_history)
        except ValueError:
            self._max_command_history = 500
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_HISTORY '
                f'set to invalid value: "{max_command_history}", defaulting to 500',
                file=sys.stderr)

        if self._max_command_history < 0:
            self._max_command_history = 500
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_HISTORY '
                f'set to invalid value: "{max_command_history}", defaulting to 500. '
                f'Value must be greater than or equal to "0"',
                file=sys.stderr)

        self._command_history = []
        self._current_command_index = -1

        self._load_command_history()
        self._load_settings()

        self._write_stdout_output(
            'This console provides a REPL for dgenerates configuration language.\n\n'
            'Enter configuration above and hit enter to submit, use the insert key to enter\n'
            'and exit multiline input mode, you must exit multiline input mode to submit configuration.\n\n'
            'Command history is supported via the up and down arrow keys.\nRight clicking the input or output '
            'pane will reveal further menu options.\n\n'
            'Enter --help or the alias \help to print dgenerates help text, all lines which\n'
            'are not directives or top level templates are processed as arguments to dgenerate.\n\n'
            'See: \\directives_help or \\directives_help (directive) for help with config directives.\n'
            'See: \\functions_help or \\functions_help (function) for help with template functions.\n'
            'See: \\templates_help or \\templates_help (variable name) for help with template variables.\n'
            '============================================================\n\n')

        self._next_text_update_line_return = False

        self._text_update()

    def _open_find_dialog(self, name, text_box):
        if self._find_dialog is not None:
            if self._find_dialog.text_widget is text_box:
                try:
                    self._find_dialog.find_entry.focus_set()
                    return
                except tk.TclError:
                    pass
            else:
                self._find_dialog.destroy()
        self._find_dialog = _FindDialog(self, name, text_box)

    def _toggle_input_wrap(self, *args):
        if self._word_wrap_input_check_var.get():
            self._input_text.enable_word_wrap()

        else:
            self._input_text.disable_word_wrap()

    def _toggle_output_wrap(self, *args):
        if self._word_wrap_output_check_var.get():
            self._output_text.enable_word_wrap()

        else:
            self._output_text.disable_word_wrap()

    def _restart_dgenerate_process(self):
        self._write_stdout_output('Restarting Shell Process...\n')
        self._start_dgenerate_process()
        self._write_stdout_output('Shell Process Started.\n'
                                  '======================\n')

    def _kill_sub_process(self):
        with self._termination_lock:
            self._sub_process.terminate()
            self._sub_process.wait()

    def _start_dgenerate_process(self):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        env['DGENERATE_LONG_TEXT_WRAP_WIDTH'] = '80'

        self._sub_process = subprocess.Popen(
            ['dgenerate', '--server'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=env,
            cwd=self._cwd)

    def _load_input_entry_text(self):
        f = tkinter.filedialog.askopenfile(
            mode='r',
            initialfile='config.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        self._input_text.text.delete('1.0', tk.END)
        self._input_text.text.insert('1.0', f.read())

    def _save_input_entry_text(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='config.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        with open(f, mode='w', encoding='utf-8') as f:
            f.write(self._input_text.text.get('1.0', tk.END))
            f.close()

    def _copy_input_entry_selection(self):
        text = self._input_text.text.selection_get()
        self.clipboard_clear()
        self.clipboard_append(text)

    def _paste_input_entry(self):
        try:
            start = self._input_text.text.index(tk.SEL_FIRST)
            end = self._input_text.text.index(tk.SEL_LAST)
            self._input_text.text.delete(start, end)
        except tk.TclError:
            start = self._input_text.text.index(tk.INSERT)

        clipboard_content = self.clipboard_get()
        self._input_text.text.insert(start, clipboard_content)

    def _copy_output_text_selection(self):
        text = self._output_text.text.selection_get()
        self.clipboard_clear()
        self.clipboard_append(text)

    def _save_output_text(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        with open(f, mode='w', encoding='utf-8') as f:
            self._output_text.text.config(state=tk.NORMAL)
            f.write(self._output_text.text.get('1.0', tk.END))
            self._output_text.text.config(state=tk.DISABLED)
            f.close()

    def _save_output_text_selection(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        with open(f, mode='w', encoding='utf-8') as f:
            f.write(self._output_text.text.selection_get())
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
                text = self._output_text_queue.get_nowait()

                scroll = self._output_text.text.yview()[1] == 1.0

                output_lines = int(self._output_text.text.index('end-1c').split('.')[0])

                if output_lines + 1 > self._max_output_lines:
                    self._output_text.text.delete('1.0',
                                                  f'{output_lines - self._max_output_lines}.0')

                if self._next_text_update_line_return:
                    self._output_text.text.delete("end-2c linestart", "end-1c")
                    self._output_text.text.insert(tk.END, text)
                else:
                    self._output_text.text.insert(tk.END, text)

                self._next_text_update_line_return = text.endswith('\r')

                if scroll:
                    self._output_text.text.see(tk.END)

            except queue.Empty:
                break
            lines += 1

        self._output_text.text.config(state=tk.DISABLED)
        self.after(self._output_refresh_rate, self._text_update)

    def _update_cwd(self):
        try:
            with self._termination_lock:
                p = psutil.Process(self._sub_process.pid)
                while p.children():
                    p = p.children()[0]
                self._cwd = p.cwd()
            self.title(f'Dgenerate Console: {self._cwd}')
        except KeyboardInterrupt:
            pass
        except IndexError:
            pass
        except psutil.NoSuchProcess:
            pass
        finally:
            self.after(100, self._update_cwd)

    def _write_stdout_output(self, text: typing.Union[bytes, str]):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stdout.buffer.write(text)
        sys.stdout.flush()
        self._output_text_queue.put(text.decode('utf-8'))

    def _write_stderr_output(self, text: typing.Union[bytes, str]):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stderr.buffer.write(text)
        sys.stderr.flush()
        self._output_text_queue.put(text.decode('utf-8'))

    @staticmethod
    def readline(file):
        line = []
        while True:
            byte = file.read(1)
            if not byte:
                break
            line.append(byte)
            if byte == b'\n':
                return b''.join(line)
            elif byte == b'\r':
                next_byte = file.read(1)
                if next_byte == b'\n':
                    line.append(next_byte)
                    return b''.join(line)
                else:
                    return b''.join(line)
        if line:
            return b''.join(line)

    def _read_sub_process_stream_thread(self, get_read_stream, write_out_handler):
        exit_message = True
        while True:
            with self._termination_lock:
                return_code = self._sub_process.poll()

            if return_code is None:
                exit_message = True
                line = self.readline(get_read_stream(self._sub_process))
                if line is not None:
                    write_out_handler(line)
            elif exit_message:
                with self._termination_lock:
                    exit_message = False
                    # immediately run all pending window events including text updates
                    self.update()
                    self._write_stdout_output(
                        f'\nShell Process Terminated, Exit Code: {return_code}\n')
                    self._restart_dgenerate_process()
            else:
                time.sleep(1)

    def _show_previous_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index > 0:
            self._current_command_index -= 1
            self._input_text.text.delete('1.0', tk.END)
            self._input_text.text.insert(tk.END, self._command_history[self._current_command_index])
            self._input_text.text.see(tk.END)
            self._input_text.text.mark_set(tk.INSERT, tk.END)
            return "break"

    def _show_next_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index < len(self._command_history) - 1:
            self._current_command_index += 1
            self._input_text.text.delete('1.0', tk.END)
            self._input_text.text.insert(tk.END, self._command_history[self._current_command_index])
            self._input_text.text.see(tk.END)
            self._input_text.text.mark_set(tk.INSERT, tk.END)
            return "break"

    def _load_settings(self):
        settings_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_settings')
        if settings_path.exists():
            with settings_path.open('r') as file:
                config = json.load(file)
                self._auto_scroll_on_run_check_var.set(config.get('auto_scroll_on_run', True))
                self._word_wrap_input_check_var.set(config.get('word_wrap_input', True))
                self._word_wrap_output_check_var.set(config.get('word_wrap_output', True))

    def save_settings(self):
        settings_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_settings')
        with settings_path.open('w') as file:
            config = {
                'auto_scroll_on_run': self._auto_scroll_on_run_check_var.get(),
                'word_wrap_input': self._word_wrap_input_check_var.get(),
                'word_wrap_output': self._word_wrap_output_check_var.get()
            }
            json.dump(config, file)

    def _load_command_history(self):
        if self._max_command_history == 0:
            return

        history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        if history_path.exists():
            with history_path.open('r') as file:
                self._command_history = json.load(file)[-self._max_command_history:]
                self._current_command_index = len(self._command_history)

    def _save_command_history(self):
        if self._max_command_history == 0:
            return

        history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        with history_path.open('w') as file:
            json.dump(self._command_history[-self._max_command_history:], file)

    def _run_input_text(self):
        user_input = self._input_text.text.get('1.0', 'end-1c')

        self._input_text.text.delete(1.0, tk.END)

        if self._auto_scroll_on_run_check_var.get():
            self._output_text.text.see(tk.END)

        self._sub_process.stdin.write((user_input + '\n\n').encode('utf-8'))

        if self._command_history:
            if self._command_history[-1] != user_input:
                # ignoredups
                self._command_history.append(user_input)
        else:
            self._command_history.append(user_input)

        self._current_command_index = len(self._command_history)
        self._save_command_history()

        self._sub_process.stdin.flush()

        return 'break'

    def destroy(self) -> None:
        self._sub_process.terminate()
        self.save_settings()
        super().destroy()


def main(args: collections.abc.Sequence[str]):
    app = None
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        app = _DgenerateConsole()
        app.mainloop()
    except KeyboardInterrupt:
        if app is not None:
            app.save_settings()
        print('Exiting dgenerate console UI due to keyboard interrupt!',
              file=sys.stderr)
        sys.exit(1)
