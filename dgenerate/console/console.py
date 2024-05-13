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
import ctypes
import json
import os
import pathlib
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.scrolledtext
import typing
import webbrowser

import PIL.Image
import PIL.ImageTk
import psutil

import dgenerate.console.finddialog as _finddialog
import dgenerate.console.karrasschedulerselect as _karrasschedulerselect
import dgenerate.console.recipesform as _recipesform
from dgenerate.console.resources import set_window_icon
from dgenerate.console.scrolledtext import ScrolledText


class DgenerateConsole(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Dgenerate Console')
        self.geometry('1000x800')

        set_window_icon(self)

        # Create main menu

        menu_bar = tk.Menu(self)

        # File menu

        self._file_menu = tk.Menu(menu_bar, tearoff=0)
        self._file_menu.add_command(label='Load',
                                    command=self._load_input_entry_text)
        self._file_menu.add_command(label='Save',
                                    command=self._load_input_entry_text)
        self._file_menu.add_separator()
        self._file_menu.add_command(label='New Window',
                                    command=lambda:
                                    subprocess.Popen('dgenerate --dgenerate-console',
                                                     stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL,
                                                     start_new_session=True))

        # Edit menu

        self._edit_menu = tk.Menu(menu_bar, tearoff=0)
        self._edit_menu.add_command(
            label='Undo', accelerator='Ctrl+Z',
            command=self._undo_input_entry)
        self._edit_menu.add_command(
            label='Redo', accelerator='Ctrl+Shift+Z',
            command=self._redo_input_entry)

        self._edit_menu.add_separator()

        self._edit_menu.add_command(label='Cut', accelerator='Ctrl+X',
                                    command=self._cut_input_entry_selection)
        self._edit_menu.add_command(label='Copy', accelerator='Ctrl+C',
                                    command=self._copy_input_entry_selection)
        self._edit_menu.add_command(label='Paste', accelerator='Ctrl+V',
                                    command=self._paste_input_entry)
        self._edit_menu.add_command(label='Delete', accelerator='DEL',
                                    command=self._delete_input_entry_selection)
        self._edit_menu.add_command(label='Select All', accelerator='Ctrl+A',
                                    command=self._select_all_input_entry)

        self._edit_menu.add_separator()

        self._edit_menu.add_command(label='Find',
                                    accelerator='Ctrl+F',
                                    command=lambda:
                                    _finddialog.open_find_dialog(
                                        self,
                                        'Find In Input',
                                        self._input_text.text))

        self._edit_menu.add_command(label='Replace',
                                    accelerator='Ctrl+R',
                                    command=lambda:
                                    _finddialog.open_find_replace_dialog(
                                        self,
                                        'Replace In Input',
                                        self._input_text.text))

        self._edit_menu.add_separator()

        self._edit_menu.add_command(label='Insert File Path (Open)',
                                    command=self._input_text_insert_file_path)
        self._edit_menu.add_command(label='Insert File Path (Save)',
                                    command=self._input_text_insert_file_path_save)
        self._edit_menu.add_command(label='Insert Directory Path',
                                    command=self._input_text_insert_directory_path)

        self._edit_menu.add_separator()

        self._edit_menu.add_command(label='Insert Recipe',
                                    command=self._input_text_insert_recipe)
        self._edit_menu.add_command(label='Insert Karras Scheduler',
                                    command=self._input_text_insert_karras_scheduler)

        # Run menu

        self._run_menu = tk.Menu(menu_bar, tearoff=0)
        self._run_menu.add_command(label='Run', accelerator='Ctrl+Space',
                                   command=self._run_input_text)
        self._run_menu.add_command(label='Kill', accelerator='Ctrl+Q',
                                   command=lambda: self.kill_shell_process(restart=True))

        # Options menu

        # Create the Multi-line Input Checkbutton

        self._multi_line_input_check_var = tk.BooleanVar(value=False)

        self._multi_line_input_check_var.trace_add(
            'write',
            lambda *args: self._input_text.text.configure(
                insertbackground='red') if self._multi_line_input_check_var.get()
            else self._input_text.text.configure(insertbackground='black'))

        self._options_menu = tk.Menu(menu_bar, tearoff=0)
        self._options_menu.add_checkbutton(label='Multiline input',
                                           accelerator='Insert Key',
                                           variable=self._multi_line_input_check_var)

        # Create the word wrap input checkbox

        self._word_wrap_input_check_var = tk.BooleanVar(value=True)
        self._word_wrap_input_check_var.trace_add('write', self._update_input_wrap)

        self._options_menu.add_checkbutton(label='Word Wrap Input',
                                           variable=self._word_wrap_input_check_var)

        # Create the word wrap output checkbox

        self._word_wrap_output_check_var = tk.BooleanVar(value=True)
        self._word_wrap_output_check_var.trace_add('write', self._update_output_wrap)

        self._options_menu.add_checkbutton(label='Word Wrap Output',
                                           variable=self._word_wrap_output_check_var)

        # Create the auto scroll checkbox (scroll on input)
        self._auto_scroll_on_run_check_var = tk.BooleanVar(value=True)
        self._options_menu.add_checkbutton(label='Auto Scroll Output On Run',
                                           variable=self._auto_scroll_on_run_check_var)

        # View Menu

        self._image_pane_visible_var = tk.BooleanVar(value=False)
        self._image_pane_window_visible_var = tk.BooleanVar(value=False)

        self._view_menu = tk.Menu(menu_bar, tearoff=0)
        self._view_menu.add_checkbutton(label='Latest Image Pane',
                                        variable=self._image_pane_visible_var)
        self._view_menu.add_checkbutton(label='Latest Image Window',
                                        variable=self._image_pane_window_visible_var)

        self._image_pane_visible_var.trace_add('write',
                                               lambda *args: self._update_image_pane_visibility())

        self._image_pane_window_visible_var.trace_add('write',
                                                      lambda *args: self._update_image_pane_window_visibility())

        # Help menu

        self._help_menu = tk.Menu(menu_bar, tearoff=0)

        self._help_menu.add_command(
            label='Documentation',
            command=lambda:
            webbrowser.open(
                'https://dgenerate.readthedocs.io/en/v3.4.5/readme.html#writing-and-running-configs'))

        self._help_menu.add_command(
            label='Examples',
            command=lambda:
            webbrowser.open(
                'https://github.com/Teriks/dgenerate/tree/v3.4.5/examples'))

        self._help_menu.add_command(
            label='Project Homepage',
            command=lambda:
            webbrowser.open(
                'https://github.com/Teriks/dgenerate/tree/v3.4.5')
        )

        # Add sub menus to main menu

        menu_bar.add_cascade(label="File", menu=self._file_menu)
        menu_bar.add_cascade(label="Edit", menu=self._edit_menu)
        menu_bar.add_cascade(label="Run", menu=self._run_menu)
        menu_bar.add_cascade(label="Options", menu=self._options_menu)
        menu_bar.add_cascade(label="View", menu=self._view_menu)
        menu_bar.add_cascade(label="Help", menu=self._help_menu)

        self.config(menu=menu_bar)

        # Split window layout

        self._paned_window_horizontal = tk.PanedWindow(self,
                                                       orient=tk.HORIZONTAL,
                                                       sashwidth=4,
                                                       bg="#808080", bd=0)

        self._paned_window_horizontal.pack(fill=tk.BOTH, expand=True)

        self._paned_window_vertical = tk.PanedWindow(self._paned_window_horizontal,
                                                     orient=tk.VERTICAL, sashwidth=4,
                                                     bg="#808080")

        self._paned_window_vertical.pack(fill=tk.BOTH, expand=True)
        self._paned_window_horizontal.add(self._paned_window_vertical)

        # Create the input text pane

        self._input_text = ScrolledText(self._paned_window_vertical, undo=True)

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
                                   lambda e: _finddialog.open_find_dialog(self,
                                                                          'Find In Input',
                                                                          self._input_text.text))

        self._input_text.text.bind('<Control-r>',
                                   lambda e: _finddialog.open_find_replace_dialog(self,
                                                                                  'Replace In Input',
                                                                                  self._input_text.text))

        self._input_text.text.bind('<Control-Z>', lambda e: self._redo_input_entry())

        self._input_text.text.bind('<Control-q>',
                                   lambda e: self.kill_shell_process(restart=True))

        self._input_text.text.bind('<Control-space>',
                                   lambda e: self._run_input_text())

        self._input_text.text.focus_set()

        self._input_text_context = tk.Menu(self._input_text, tearoff=0)

        self._input_text_context.add_command(label='Cut', accelerator='Ctrl+X',
                                             command=self._cut_input_entry_selection)
        self._input_text_context.add_command(label='Copy', accelerator='Ctrl+C',
                                             command=self._copy_input_entry_selection)
        self._input_text_context.add_command(label='Paste', accelerator='Ctrl+V',
                                             command=self._paste_input_entry)
        self._input_text_context.add_command(label='Delete', accelerator='DEL',
                                             command=self._delete_input_entry_selection)
        self._input_text_context.add_command(label='Select All', accelerator='Ctrl+A',
                                             command=self._select_all_input_entry)
        self._input_text_context.add_separator()

        self._input_text_context.add_command(label='Find',
                                             accelerator='Ctrl+F',
                                             command=lambda:
                                             _finddialog.open_find_dialog(
                                                 self,
                                                 'Find In Input',
                                                 self._input_text.text))

        self._input_text_context.add_command(label='Replace',
                                             accelerator='Ctrl+R',
                                             command=lambda:
                                             _finddialog.open_find_replace_dialog(
                                                 self,
                                                 'Replace In Input',
                                                 self._input_text.text))
        self._input_text_context.add_separator()
        self._input_text_context.add_command(label='Load', command=self._load_input_entry_text)
        self._input_text_context.add_command(label='Save', command=self._save_input_entry_text)
        self._input_text_context.add_separator()
        self._input_text_context.add_command(label='Insert File Path (Open)',
                                             command=self._input_text_insert_file_path)
        self._input_text_context.add_command(label='Insert File Path (Save)',
                                             command=self._input_text_insert_file_path_save)
        self._input_text_context.add_command(label='Insert Directory Path',
                                             command=self._input_text_insert_directory_path)
        self._input_text_context.add_separator()
        self._input_text_context.add_command(label='Insert Recipe',
                                             command=self._input_text_insert_recipe)
        self._input_text_context.add_command(label='Insert Karras Scheduler',
                                             command=self._input_text_insert_karras_scheduler)

        self._paned_window_vertical.add(self._input_text)

        # Create the output text pane

        self._output_text = ScrolledText(self._paned_window_vertical)

        self._output_text.text.bind('<Button-3>',
                                    lambda e: self._output_text_context.tk_popup(
                                        self.winfo_pointerx(), self.winfo_pointery()))
        self._output_text.text.bind('<Control-f>',
                                    lambda e: _finddialog.open_find_dialog(
                                        self, 'Find In Output',
                                        self._output_text.text))

        self._output_text.text.bind('<Down>',
                                    lambda e: self._output_text.text.see(tk.END))

        self._output_text_context = tk.Menu(self._output_text, tearoff=0)

        self._output_text_context.add_command(label='Clear', command=self._clear_output_text)
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Find',
                                              accelerator='Ctrl+F',
                                              command=lambda:
                                              _finddialog.open_find_dialog(
                                                  self, 'Find In Output',
                                                  self._output_text.text))
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Copy',
                                              accelerator='Ctrl+C',
                                              command=self._copy_output_text_selection)
        self._output_text_context.add_command(label='Select All',
                                              accelerator='Ctrl+A',
                                              command=self._select_all_output_text)
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Save Selection', command=self._save_output_text_selection)
        self._output_text_context.add_command(label='Save All', command=self._save_output_text)
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='To Bottom',
                                              accelerator='Down Arrow',
                                              command=
                                              lambda: self._output_text.text.see(tk.END))

        self._paned_window_vertical.add(self._output_text)

        # Optional latest image pane

        self._displayed_image = None
        self._displayed_image_path = None
        self._image_pane_window = None
        self._image_pane_window_last_pos = None
        self._image_pane = tk.Frame(self._paned_window_horizontal, bg='black')
        image_label = tk.Label(self._image_pane, bg='black')
        image_label.pack(fill=tk.BOTH, expand=True)
        image_label.bind('<Configure>',
                         lambda e: self._resize_image_pane_image(image_label))

        self._image_pane_context = tk.Menu(self._image_pane, tearoff=0)

        if self._install_show_in_directory_entry(
                self._image_pane_context, lambda: self._displayed_image_path):
            self._image_pane_context.add_separator()

        self._image_pane_context.add_command(
            label='Hide Image Pane',
            command=lambda: self._image_pane_visible_var.set(False))

        self._image_pane_context.add_command(
            label='Make Window',
            command=lambda: self._image_pane_window_visible_var.set(True))

        image_label.bind('<Button-3>',
                         lambda e: self._image_pane_context.tk_popup(
                             self.winfo_pointerx(), self.winfo_pointery()))

        # Misc Config

        self._termination_lock = threading.Lock()

        self._cwd = os.getcwd()
        self._start_shell_process()

        self._output_text_queue = queue.Queue()

        self._shell_reader_threads = []
        self._start_shell_reader_threads()

        self._find_dialog = None

        self._update_cwd()

        # output lines per refresh
        self._output_lines_per_refresh = 30

        # output (stdout) refresh rate (ms)
        self._output_refresh_rate = 100

        max_scrollback_default = 10000

        # max output scroll-back history
        max_output_lines = os.environ.get('DGENERATE_CONSOLE_MAX_SCROLLBACK', max_scrollback_default)
        try:
            self._max_output_lines = int(max_output_lines)
        except ValueError:
            self._max_output_lines = max_scrollback_default
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_SCROLLBACK '
                f'set to invalid value: "{max_output_lines}", defaulting to 10000',
                file=sys.stderr)

        if self._max_output_lines < self._output_lines_per_refresh:
            self._max_output_lines = max_scrollback_default
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_SCROLLBACK '
                f'set to invalid value: "{max_output_lines}", defaulting to 10000. '
                f'Value must be greater than or equal to "{self._output_lines_per_refresh}"',
                file=sys.stderr)

        max_history_default = 500

        # max terminal command history
        max_command_history = os.environ.get('DGENERATE_CONSOLE_MAX_HISTORY', max_history_default)
        try:
            self._max_command_history = int(max_command_history)
        except ValueError:
            self._max_command_history = max_history_default
            print(
                f'WARNING: environmental variable DGENERATE_CONSOLE_MAX_HISTORY '
                f'set to invalid value: "{max_command_history}", defaulting to 500',
                file=sys.stderr)

        if self._max_command_history < 0:
            self._max_command_history = max_history_default
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
            'and exit multiline input mode, you must exit multiline input mode to submit\n'
            'configuration via the enter key, or instead use Ctrl+Space / the run menu.\n\n'
            'Command history is supported via the up and down arrow keys when not in multiline\n'
            'input mode. Right clicking the input or output pane will reveal further menu options.\n\n'
            'Enter --help or the alias \\help to print dgenerates help text. All lines which\n'
            'are not directives (starting with \\ ) or top level templates (starting with { )\n'
            'are processed as arguments to dgenerate.\n\n'
            'See: \\directives_help or \\directives_help (directive) for help with config directives.\n'
            'See: \\functions_help or \\functions_help (function) for help with template functions.\n'
            'See: \\templates_help or \\templates_help (variable name) for help with template variables.\n'
            '============================================================\n\n')

        self._next_text_update_line_return = False

        self._text_update()

        self.bind("<<UpdateEvent>>", lambda *a: self.update())

    def _start_shell_reader_threads(self):
        self._shell_reader_threads = [
            threading.Thread(target=self._read_shell_output_stream_thread,
                             args=(lambda p: p.stdout, self._write_stdout_output)),
            threading.Thread(target=self._read_shell_output_stream_thread,
                             args=(lambda p: p.stderr, self._write_stderr_output))
        ]

        for t in self._shell_reader_threads:
            t.start()

    @staticmethod
    def _install_show_in_directory_entry(menu: tk.Menu, get_path: typing.Callable[[], str]):
        open_file_explorer_support = platform.system() == 'Windows' or shutil.which('nautilus')

        if open_file_explorer_support:

            if platform.system() == 'Windows':
                def open_image_pane_directory_in_explorer():
                    subprocess.Popen(
                        rf'explorer /select,"{get_path()}"',
                        start_new_session=True)
            else:
                def open_image_pane_directory_in_explorer():
                    subprocess.Popen(
                        rf'nautilus "{get_path()}"',
                        start_new_session=True)

            menu.add_command(label='Open In Directory',
                             command=open_image_pane_directory_in_explorer)

            og_popup = menu.tk_popup

            def patch_tk_popup(*args, **kwargs):
                path = get_path()

                if path is not None and \
                        os.path.exists(path):
                    menu.entryconfigure(
                        'Open In Directory', state=tk.NORMAL)
                else:
                    menu.entryconfigure(
                        'Open In Directory', state=tk.DISABLED)
                og_popup(*args, **kwargs)

            menu.tk_popup = patch_tk_popup

            return True
        return False

    def _insert_or_replace_input_text(self, text):
        try:
            selection_start = self._input_text.text.index("sel.first")
            selection_end = self._input_text.text.index("sel.last")
        except tk.TclError:
            selection_start = selection_end = None

        self._input_text.text.edit_separator()
        if selection_start and selection_end:
            self._input_text.text.replace(selection_start, selection_end, text)
        else:
            self._input_text.text.insert("insert", text)

    def _input_text_insert_recipe(self):
        s = _recipesform.request_recipe(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_karras_scheduler(self):
        s = _karrasschedulerselect.request_scheduler(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_directory_path(self):
        d = tkinter.filedialog.askdirectory(
            initialdir=self._cwd)

        if d is None or not d.strip():
            return

        self._insert_or_replace_input_text(shlex.quote(d))

    def _input_text_insert_file_path(self):
        f = tkinter.filedialog.askopenfilename(
            initialdir=self._cwd)

        if f is None or not f.strip():
            return

        self._insert_or_replace_input_text(shlex.quote(f))

    def _input_text_insert_file_path_save(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialdir=self._cwd)

        if f is None or not f.strip():
            return

        self._insert_or_replace_input_text(shlex.quote(f))

    def _update_image_pane_visibility(self):
        if not self._image_pane_visible_var.get():
            self._paned_window_horizontal.remove(self._image_pane)
        else:
            if self._image_pane_window_visible_var.get():
                self._image_pane_window_visible_var.set(False)
            self._paned_window_horizontal.add(self._image_pane)

    def _create_image_pane_window(self):
        self._image_pane_window = tk.Toplevel(self)

        self._image_pane_window.geometry('512x512' + (
            '+{}+{}'.format(*self._image_pane_window_last_pos) if
            self._image_pane_window_last_pos is not None else ''))

        self._image_pane_window.title('Latest Image')
        image_pane = tk.Frame(self._image_pane_window, bg='black')
        image_pane.pack(fill=tk.BOTH, expand=True)
        image_label = tk.Label(image_pane, bg='black')
        image_label.pack(fill=tk.BOTH, expand=True)
        image_label.bind(
            '<Configure>', lambda e: self._resize_image_pane_image(image_label))

        image_window_context = tk.Menu(self._image_pane_window, tearoff=0)

        if self._install_show_in_directory_entry(image_window_context,
                                                 lambda: self._displayed_image_path):
            image_window_context.add_separator()

        image_window_context.add_command(label='Make Pane',
                                         command=lambda: self._image_pane_visible_var.set(True))

        image_label.bind(
            '<Button-3>', lambda e:
            image_window_context.tk_popup(self.winfo_pointerx(), self.winfo_pointery()))

        def on_closing():
            self._image_pane_window_visible_var.set(False)
            self._image_pane_window.withdraw()

        self._image_pane_window.protocol("WM_DELETE_WINDOW", on_closing)

    def _update_image_pane_window_visibility(self):
        if not self._image_pane_window_visible_var.get():
            if self._image_pane_window is not None:
                self._image_pane_window.withdraw()
        else:
            if self._image_pane_visible_var.get():
                self._image_pane_visible_var.set(False)

            if self._image_pane_window is not None:
                self._image_pane_window.deiconify()
                return

            self._create_image_pane_window()

    def _image_pane_load_image(self, image_path):
        if self._displayed_image is not None:
            self._displayed_image.close()

        self._displayed_image = PIL.Image.open(image_path)
        self._displayed_image_path = image_path

        self._image_pane.winfo_children()[0].event_generate('<Configure>')

        if self._image_pane_window is not None:
            self._image_pane_window.winfo_children()[0].winfo_children()[0].event_generate('<Configure>')

    def _resize_image_pane_image(self, label: tk.Label):
        if self._displayed_image is None:
            return

        label_width = label.winfo_width()
        label_height = label.winfo_height()

        image_width = self._displayed_image.width
        image_height = self._displayed_image.height

        image_aspect = image_width / image_height
        label_aspect = label_width / label_height

        if image_aspect > label_aspect:
            width = label_width
            height = int(width / image_aspect)
        else:
            height = label_height
            width = int(height * image_aspect)

        img = self._displayed_image.resize((width, height), PIL.Image.Resampling.LANCZOS)

        photo_img = PIL.ImageTk.PhotoImage(img)

        label.config(image=photo_img)

        label.image = photo_img

    def _update_input_wrap(self, *args):
        if self._word_wrap_input_check_var.get():
            self._input_text.enable_word_wrap()

        else:
            self._input_text.disable_word_wrap()

    def _update_output_wrap(self, *args):
        if self._word_wrap_output_check_var.get():
            self._output_text.enable_word_wrap()

        else:
            self._output_text.disable_word_wrap()

    def kill_shell_process(self, restart=False):
        with self._termination_lock:
            if self._shell_process is None:
                return

            try:
                self._shell_process.terminate()
            except psutil.NoSuchProcess:
                return

            return_code = -1

            try:
                return_code = self._shell_process.wait(timeout=5)
            except psutil.TimeoutExpired:
                self._shell_process.kill()
                try:
                    return_code = self._shell_process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    self._write_stderr_output(
                        'WARNING: Could not kill interpreter process, possible zombie process.')

            self._shell_return_code_message(return_code)

            del self._shell_process
            self._shell_process = None

            if restart:
                self._shell_restarting_commence_message()
                self._start_shell_process()

            for thread in self._shell_reader_threads:
                # there is no better way than to raise an exception on the thread,
                # the threads will live forever if they are blocking on read even if they are daemon threads,
                # and there is no platform independent non-blocking IO with a subprocess
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.native_id, ctypes.py_object(SystemExit))
                if res > 1:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.native_id, 0)
                    raise SystemError("PyThreadState_SetAsyncExc failed")

            if restart:
                self._start_shell_reader_threads()
                self._shell_restarting_finish_message()

    def _shell_return_code_message(self, return_code):
        self._write_stdout_output(
            f'\nShell Process Terminated, Exit Code: {return_code}\n')

    def _shell_restarting_commence_message(self):
        self._write_stdout_output('Restarting Shell Process...\n')

    def _shell_restarting_finish_message(self):
        self._write_stdout_output('Shell Process Started.\n'
                                  '======================\n')

    def _start_shell_process(self):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        env['DGENERATE_LONG_TEXT_WRAP_WIDTH'] = '100'

        self._shell_process = psutil.Popen(
            ['dgenerate', '--shell'],
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

        self._input_text.text.replace('1.0', tk.END, f.read())

    def _save_input_entry_text(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='config.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None or not f.strip():
            return

        with open(f, mode='w', encoding='utf-8') as f:
            f.write(self._input_text.text.get('1.0', 'end-1c'))
            f.close()

    def _undo_input_entry(self):
        self._input_text.text.event_generate("<<Undo>>")

    def _redo_input_entry(self):
        self._input_text.text.event_generate("<<Redo>>")

    def _cut_input_entry_selection(self):
        self._input_text.text.event_generate("<<Cut>>")

    def _copy_input_entry_selection(self):
        self._input_text.text.event_generate("<<Copy>>")

    def _paste_input_entry(self):
        self._input_text.text.event_generate("<<Paste>>")

    def _delete_input_entry_selection(self):
        self._input_text.text.event_generate("<Delete>")

    def _select_all_input_entry(self):
        self._input_text.text.event_generate("<<SelectAll>>")

    def _copy_output_text_selection(self):
        self._output_text.text.event_generate("<<Copy>>")

    def _select_all_output_text(self):
        self._output_text.text.focus_set()
        self._output_text.text.event_generate("<<SelectAll>>")

    def _save_output_text(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None or not f.strip():
            return

        with open(f, mode='w', encoding='utf-8') as f:
            self._output_text.text.config(state=tk.NORMAL)
            f.write(self._output_text.text.get('1.0', 'end-1c'))
            self._output_text.text.config(state=tk.DISABLED)
            f.close()

    def _save_output_text_selection(self):
        f = tkinter.filedialog.asksaveasfilename(
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None or not f.strip():
            return

        with open(f, mode='w', encoding='utf-8') as f:
            f.write(self._output_text.text.selection_get())
            f.close()

    def _clear_output_text(self):
        self._output_text.text.config(state=tk.NORMAL)
        self._output_text.text.delete('1.0', tk.END)
        self._output_text.text.config(state=tk.DISABLED)

    def _check_text_for_latest_image(self, text):
        match = re.match(
            r'Wrote Image File: "(.*?)"|\\image_process: Wrote Image "(.*?)"|\\image_process: Wrote Frame "(.*?)"',
            text)
        if match is not None:
            path = os.path.join(self._cwd, ''.join(filter(None, match.groups())))
            if os.path.exists(path):
                self._image_pane_load_image(path)

    def _text_update(self):
        lines = 0

        self._output_text.text.config(state=tk.NORMAL)

        while True:
            if lines > self._output_lines_per_refresh:
                break
            try:
                with self._termination_lock:
                    # wait for possible submission of an exit message
                    text = self._output_text_queue.get_nowait()

                self._check_text_for_latest_image(text)

                scroll = self._output_text.text.yview()[1] == 1.0

                output_lines = int(self._output_text.text.index('end-1c').split('.')[0])

                if output_lines + 1 > self._max_output_lines:
                    self._output_text.text.delete('1.0',
                                                  f'{output_lines - self._max_output_lines}.0')

                if self._next_text_update_line_return:
                    self._output_text.text.replace("end-2c linestart", "end-1c", text)
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
                p = self._shell_process
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

    def _read_shell_output_stream_thread(self, get_read_stream, write_out_handler):
        exit_message = True
        while True:
            with self._termination_lock:
                return_code = self._shell_process.poll()

            if return_code is None:
                exit_message = True
                line = self.readline(get_read_stream(self._shell_process))
                if line is not None:
                    write_out_handler(line)
            elif exit_message:
                with self._termination_lock:
                    exit_message = False
                    self._shell_return_code_message(return_code)

                    del self._shell_process
                    self._shell_process = None

                    self._shell_restarting_commence_message()
                    self._start_shell_process()
                    self._shell_restarting_finish_message()
            else:
                time.sleep(1)

    def _show_previous_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index > 0:
            self._current_command_index -= 1
            self._input_text.text.edit_separator()
            self._input_text.text.replace('1.0', tk.END, self._command_history[self._current_command_index])
            self._input_text.text.see(tk.END)
            self._input_text.text.mark_set(tk.INSERT, tk.END)
            return "break"

    def _show_next_command(self, event):
        if self._multi_line_input_check_var.get():
            return
        if self._current_command_index < len(self._command_history) - 1:
            self._current_command_index += 1
            self._input_text.text.edit_separator()
            self._input_text.text.replace('1.0', tk.END, self._command_history[self._current_command_index])
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

        if not self._multi_line_input_check_var.get():
            self._input_text.text.delete(1.0, tk.END)

        if self._auto_scroll_on_run_check_var.get():
            self._output_text.text.see(tk.END)

        with self._termination_lock:
            self._shell_process.stdin.write((user_input + '\n\n').encode('utf-8'))
            self._shell_process.stdin.flush()

        if self._command_history:
            if self._command_history[-1] != user_input:
                # ignoredups
                self._command_history.append(user_input)
        else:
            self._command_history.append(user_input)

        self._current_command_index = len(self._command_history)
        self._save_command_history()

        return 'break'

    def destroy(self) -> None:
        self.kill_shell_process()
        self.save_settings()
        super().destroy()


def main(args: collections.abc.Sequence[str]):
    if platform.system() == 'Windows':
        # secret hack for Windows desktop shortcut
        if os.path.abspath(os.getcwd()) == os.path.abspath(r'C:\Program Files\dgenerate'):
            os.chdir(os.path.expanduser('~'))

    app = None
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        app = DgenerateConsole()
        app.mainloop()
    except KeyboardInterrupt:
        if app is not None:
            app.save_settings()
        app.kill_shell_process()
        print('Exiting dgenerate console UI due to keyboard interrupt!',
              file=sys.stderr)
        sys.exit(1)
