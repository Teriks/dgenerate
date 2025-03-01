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
import os
import pathlib
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tkinter as tk
import typing

import PIL.Image
import PIL.ImageTk

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.finddialog as _finddialog
import dgenerate.console.imageprocessorselect as _imageprocessorselect
import dgenerate.console.imageseedselect as _imageseedselect
import dgenerate.console.karrasschedulerselect as _karrasschedulerselect
import dgenerate.console.promptupscalerselect as _promptupscalerselect
import dgenerate.console.promptweighterselect as _promptweighterselect
import dgenerate.console.recipesform as _recipesform
import dgenerate.console.argumentselect as _argumentselect
import dgenerate.console.resources as _resources
import dgenerate.textprocessing as _textprocessing
from dgenerate.console.codeview import DgenerateCodeView
from dgenerate.console.scrolledtext import ScrolledText
from dgenerate.console.stdinpipe import StdinPipeFullError
from dgenerate.console.procmon import ProcessMonitor

DGENERATE_EXE = \
    os.path.splitext(
        os.path.basename(os.path.realpath(sys.argv[0])))[0]


class DgenerateConsole(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title('Dgenerate Console')

        width = 1000
        height = 800

        if platform.system() == 'Windows':
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            position_left = (screen_width // 2) - (width // 2)
            position_top = (screen_height // 2) - (height // 2)

            self.geometry(f"{width}x{height}+{position_left}+{position_top}")
        else:
            self.geometry(f"{width}x{height}")

        _resources.set_window_icon(self)

        self._shell_process = None

        # Create main menu

        # File menu
        self._menu_bar = tk.Menu(self)
        self._file_menu = tk.Menu(self._menu_bar, tearoff=0)
        self._last_known_working_dir = os.getcwd()

        def handle_new_window():
            subprocess.Popen(
                [DGENERATE_EXE, '--console'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        self._file_menu.add_command(label='New',
                                    command=handle_new_window)

        self._file_menu.add_separator()

        self._file_menu.add_command(label='Load',
                                    command=self._load_input_entry_text)
        self._file_menu.add_command(label='Save',
                                    command=self._save_input_entry_text)

        # Edit menu

        self._edit_menu = tk.Menu(self._menu_bar, tearoff=0)
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

        self._edit_menu.add_command(label='Format Code',
                                    accelerator='Ctrl+Shift+F',
                                    command=self._format_code)

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
        self._edit_menu.add_command(label='Insert Argument',
                                    command=self._input_text_insert_argument)

        self._edit_menu.add_separator()

        self._edit_menu.add_command(label='Insert Image Seed URI',
                                    command=self._input_text_insert_image_seed)
        self._edit_menu.add_command(label='Insert Karras Scheduler URI',
                                    command=self._input_text_insert_karras_scheduler)
        self._edit_menu.add_command(label='Insert Image Processor URI',
                                    command=self._input_text_insert_image_processor)
        self._edit_menu.add_command(label='Insert Prompt Upscaler URI',
                                    command=self._input_text_insert_prompt_upscaler)
        self._edit_menu.add_command(label='Insert Prompt Weighter URI',
                                    command=self._input_text_insert_prompt_weighter)

        # Run menu

        self._run_menu = tk.Menu(self._menu_bar, tearoff=0)

        self._run_menu.add_command(
            label='Run',
            accelerator=
            'Command+r' if platform.system() == 'Darwin' else 'Ctrl+Space',
            command=self._run_input_text)

        self._run_menu.add_command(label='Kill', accelerator='Ctrl+Q',
                                   command=lambda: self.kill_shell_process(restart=True))

        self._run_menu.add_separator()

        self._debug_mode_var = tk.BooleanVar(value=False)

        self._debug_mode_var.trace_add('write',
                                       lambda *a: self._update_debug_mode_state())

        self._run_menu.add_checkbutton(label='Debug Mode',
                                       variable=self._debug_mode_var)

        # Options menu

        # Create the Multi-line Input Checkbutton

        self._multi_line_input_check_var = tk.BooleanVar(value=False)

        def multiline_check_trace(*args):
            if self._multi_line_input_check_var.get():
                self._input_text.enable_line_numbers()
            else:
                self._input_text.disable_line_numbers()

        self._multi_line_input_check_var.trace_add(
            'write',
            multiline_check_trace)

        self._options_menu = tk.Menu(self._menu_bar, tearoff=0)
        self._theme_menu = tk.Menu(self._menu_bar, tearoff=0)
        self._theme_menu_var = tk.StringVar(value='dgenerate')

        self._theme_menu_var.trace_add('write',
                                       lambda *a:
                                       self._set_theme(self._theme_menu_var.get()))

        self._theme_menu.add_radiobutton(label='none', value='none',
                                         variable=self._theme_menu_var)

        for theme_name in sorted(DgenerateCodeView.THEMES.keys()):
            self._theme_menu.add_radiobutton(label=theme_name, value=theme_name,
                                             variable=self._theme_menu_var)

        self._options_menu.add_cascade(label='Theme', menu=self._theme_menu)

        self._options_menu.add_checkbutton(
            label='Multiline input',
            accelerator=
            'Command+i' if platform.system() == 'Darwin' else 'Insert Key',
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
        self._options_menu.add_checkbutton(label='Scroll Output On Run',
                                           variable=self._auto_scroll_on_run_check_var)

        # Create the auto scroll checkbox (scroll on output)
        self._auto_scroll_on_output_check_var = tk.BooleanVar(value=False)
        self._options_menu.add_checkbutton(label='Scroll Output On New Output',
                                           variable=self._auto_scroll_on_output_check_var)

        # View Menu

        self._image_pane_visible_var = tk.BooleanVar(value=False)
        self._image_pane_window_visible_var = tk.BooleanVar(value=False)

        self._view_menu = tk.Menu(self._menu_bar, tearoff=0)
        self._view_menu.add_checkbutton(label='Latest Image Pane',
                                        variable=self._image_pane_visible_var)
        self._view_menu.add_checkbutton(label='Latest Image Window',
                                        variable=self._image_pane_window_visible_var)

        self._image_pane_visible_var.trace_add('write',
                                               lambda *args: self._update_image_pane_visibility())

        self._image_pane_window_visible_var.trace_add('write',
                                                      lambda *args: self._update_image_pane_window_visibility())

        # Help menu

        self._help_menu = tk.Menu(self._menu_bar, tearoff=0)

        _resources.add_help_menu_links(self._help_menu)

        # Add sub menus to main menu

        self._menu_bar.add_cascade(label='File', menu=self._file_menu)
        self._menu_bar.add_cascade(label='Edit', menu=self._edit_menu)
        self._menu_bar.add_cascade(label='Run', menu=self._run_menu)
        self._menu_bar.add_cascade(label='Options', menu=self._options_menu)
        self._menu_bar.add_cascade(label='View', menu=self._view_menu)
        self._menu_bar.add_cascade(label='Help', menu=self._help_menu)

        self.config(menu=self._menu_bar)

        # Split window layout

        self._paned_window_horizontal = tk.PanedWindow(self,
                                                       orient=tk.HORIZONTAL,
                                                       sashwidth=4,
                                                       bg='#808080', bd=0)

        self._paned_window_horizontal.pack(fill=tk.BOTH, expand=True)

        self._paned_window_vertical = tk.PanedWindow(self._paned_window_horizontal,
                                                     orient=tk.VERTICAL, sashwidth=4,
                                                     bg='#808080')

        self._paned_window_vertical.pack(fill=tk.BOTH, expand=True)
        self._paned_window_horizontal.add(self._paned_window_vertical)

        # Create the input text pane

        self._input_text = DgenerateCodeView(self._paned_window_vertical, undo=True)

        self._input_text.text.bind('<Return>',
                                   lambda e:
                                   None if self._multi_line_input_check_var.get()
                                   else self._run_input_text())
        self._input_text.text.bind('<Up>', self._show_previous_command)
        self._input_text.text.bind('<Down>', self._show_next_command)
        self._input_text.text.bind('<Button-3>',
                                   lambda e: self._input_text_context.tk_popup(
                                       self.winfo_pointerx(), self.winfo_pointery()))

        def handle_insert(e):
            self._multi_line_input_check_var.set(
                not self._multi_line_input_check_var.get())
            # not safe to assume tkinter does not do something
            # else with the insert key
            return 'break'

        def handle_ctrl_a(event):
            event.widget.event_generate('<<SelectAll>>')
            return 'break'

        if platform.system() == 'Darwin':
            self._input_text.text.bind('<Command-i>', handle_insert)
        else:
            self._input_text.text.bind('<Insert>', handle_insert)

        self._input_text.text.bind('<Control-f>',
                                   lambda e: _finddialog.open_find_dialog(
                                       self,
                                       'Find In Input',
                                       self._input_text.text))

        self._input_text.text.bind('<Control-r>',
                                   lambda e: _finddialog.open_find_replace_dialog(
                                       self,
                                       'Replace In Input',
                                       self._input_text.text))

        if platform.system() == 'Darwin':
            # not native
            self._input_text.text.bind('<Control-z>', lambda e: self._undo_input_entry())

        self._input_text.text.bind('<Control-Z>', lambda e: self._redo_input_entry())

        self._input_text.text.bind('<Control-q>',
                                   lambda e: self.kill_shell_process(restart=True))

        if platform.system() == 'Darwin':
            self._input_text.text.bind('<Command-r>',
                                       lambda e: self._run_input_text())
        else:
            self._input_text.text.bind('<Control-space>',
                                       lambda e: self._run_input_text())

        self._input_text.text.bind('<Control-F>', lambda e: self._format_code())

        self._input_text.text.bind('<Control-a>', handle_ctrl_a)

        self._input_text.text.focus_set()

        self._input_text_context = tk.Menu(self._input_text, tearoff=0)

        if platform.system() == 'Darwin':
            # these are not native
            self._input_text.text.bind('<Control-x>',
                                       lambda e: self._cut_input_entry_selection())
            self._input_text.text.bind('<Control-c>',
                                       lambda e: self._copy_input_entry_selection())
            self._input_text.text.bind('<Control-v>',
                                       lambda e: self._paste_input_entry())

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
        self._input_text_context.add_command(label='Format Code',
                                             accelerator='Ctrl+Shift+F',
                                             command=self._format_code)
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
        self._input_text_context.add_command(label='Insert Argument',
                                             command=self._input_text_insert_argument)

        self._input_text_context.add_separator()

        self._input_text_context.add_command(label='Insert Image Seed URI',
                                             command=self._input_text_insert_image_seed)
        self._input_text_context.add_command(label='Insert Karras Scheduler URI',
                                             command=self._input_text_insert_karras_scheduler)
        self._input_text_context.add_command(label='Insert Image Processor URI',
                                             command=self._input_text_insert_image_processor)
        self._input_text_context.add_command(label='Insert Prompt Upscaler URI',
                                             command=self._input_text_insert_prompt_upscaler)
        self._input_text_context.add_command(label='Insert Prompt Weighter URI',
                                             command=self._input_text_insert_prompt_weighter)

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

        self.bind('<Control-Up>',
                  lambda e: self._output_text.text.see('1.0'))

        self.bind('<Control-Down>',
                  lambda e: self._output_text.text.see(tk.END))

        self.bind('<Control-a>', handle_ctrl_a)

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
        self._output_text_context.add_command(label='To Top',
                                              accelerator='Ctrl+Up Arrow',
                                              command=
                                              lambda: self._output_text.text.see('1.0'))
        self._output_text_context.add_command(label='To Bottom',
                                              accelerator='Ctrl+Down Arrow',
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

        self._find_dialog = None

        # output lines per refresh
        self._output_lines_per_refresh = 30

        if platform.system() == 'Windows':
            # the windows tcl backend cannot handle
            # rapid updates, it will block-up the event loop
            # causing the window to be unresponsive

            # output (stdout) refresh rate (ms)
            self._output_refresh_rate = 100
        else:
            # linux / macos tcl backend can handle the
            # shell process output in near real time

            # output (stdout) refresh rate (ms)
            self._output_refresh_rate = 5

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

        multiline_enter_mode = 'Command+i' if platform.system() == 'Darwin' else 'the insert key'
        run_key = 'Command+r' if platform.system() == 'Darwin' else 'Ctrl+Space'

        self._output_text.text.insert(
            '1.0',
            "This console provides a REPL for dgenerate's configuration language.\n\n"
            f"Enter configuration above and hit enter to submit, or use {multiline_enter_mode}\n"
            "to enter multiline input mode (indicated by the presence of line numbers). You must\n"
            "exit multiline input mode to submit configuration with the enter key, or instead\n"
            f"use {run_key} / the run menu.\n\n"
            "Command history is supported via the up and down arrow keys when not in multiline\n"
            "input mode. Right clicking the input or output pane will reveal further menu options.\n\n"
            "To get started quickly with image generation, see the menu option: Edit -> Insert Recipe.\n\n"
            "Enter --help or the alias \\help to print dgenerate's help text. All lines which\n"
            "are not directives (starting with \\ ) or top level templates (starting with { )\n"
            "are processed as arguments to dgenerate.\n\n"
            "See: \\directives_help or \\directives_help (directive) for help with config directives.\n"
            "See: \\functions_help or \\functions_help (function) for help with template functions.\n"
            "See: \\templates_help or \\templates_help (variable name) for help with template variables.\n"
            "============================================================\n\n")

        self._output_text.text.config(state=tk.DISABLED)

        self._next_text_update_line_return = False
        self._next_text_update_line_escape = False

        self._shell_procmon = ProcessMonitor(events_per_tick=self._output_lines_per_refresh)
        self._shell_procmon.stderr_callback = self._write_stderr_output
        self._shell_procmon.stdout_callback = self._write_stdout_output
        self._shell_procmon.process_exit_callback = self._shell_return_code_message
        self._shell_procmon.process_restarting_callback = self._shell_restarting_commence_message
        self._shell_procmon.process_restarted_callback = self._shell_restarting_finish_message

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        env['COLUMNS'] = '100'

        self._shell_procmon.popen([DGENERATE_EXE, '--shell'], env=env)

        def process_monitor_events():
            self._shell_procmon.process_events()

            self.after(self._output_refresh_rate, process_monitor_events)

        process_monitor_events()

        self._update_cwd_title(os.getcwd())

    def _update_debug_mode_state(self):
        self._shell_procmon.popen_args = \
            [[DGENERATE_EXE, '--shell'] + (['-v'] if self._debug_mode_var.get() else [])]
        self.kill_shell_process(restart=True)

    def kill_shell_process(self, restart=False):
        if restart:
            self._shell_procmon.restart()
        else:
            self._shell_procmon.close()

    def _format_code(self):
        self._input_text.format_code()

    def _set_theme(self, name):
        self._input_text.set_theme(name)

        bg = self._input_text.text.cget('bg')
        fg = self._input_text.text.cget('fg')

        self._output_text.text.configure(
            bg=bg,
            fg=fg
        )

        _resources.set_textbox_theme(bg=bg, fg=fg)

        error_color = fg

        if name in DgenerateCodeView.THEMES:
            theme = DgenerateCodeView.THEMES[name]
            error_color = theme['general']['error']

        self._output_text.text.tag_configure(
            'error', foreground=error_color)

        self.save_settings()

    @staticmethod
    def _install_show_in_directory_entry(menu: tk.Menu, get_path: typing.Callable[[], str]):
        open_file_explorer_support = platform.system() in {'Windows', 'Darwin'} or shutil.which('nautilus')

        if open_file_explorer_support:

            if platform.system() == 'Windows':
                def open_image_pane_directory_in_explorer():
                    subprocess.Popen(
                        ['explorer', '/select', ',', get_path()],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
            elif platform.system() == 'Darwin':  # macOS
                def open_image_pane_directory_in_explorer():
                    file_path = get_path()
                    # Use AppleScript to open Finder and highlight the file
                    subprocess.Popen(
                        ['osascript', '-e', f'tell application "Finder" to reveal POSIX file "{file_path}"'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    subprocess.Popen(
                        ['osascript', '-e', 'tell application "Finder" to activate'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
            else:  # Linux (assuming nautilus is available)
                def open_image_pane_directory_in_explorer():
                    subprocess.Popen(
                        ['nautilus', get_path()],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

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
            selection_start = self._input_text.text.index('sel.first')
            selection_end = self._input_text.text.index('sel.last')
        except tk.TclError:
            selection_start = selection_end = None

        self._input_text.text.edit_separator()
        if selection_start and selection_end:
            self._input_text.text.replace(selection_start, selection_end, text)
        else:
            self._input_text.text.insert('insert', text)

    def _input_text_insert_recipe(self):
        s = _recipesform.request_recipe(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_argument(self):
        s = _argumentselect.request_argument(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_image_seed(self):
        s = _imageseedselect.request_uri(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_karras_scheduler(self):
        s = _karrasschedulerselect.request_uri(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_image_processor(self):
        s = _imageprocessorselect.request_uri(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_prompt_upscaler(self):
        s = _promptupscalerselect.request_uri(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_prompt_weighter(self):
        s = _promptweighterselect.request_uri(
            master=self)

        if s is None or not s.strip():
            return

        self._insert_or_replace_input_text(s)

    def _input_text_insert_directory_path(self):
        d = _filedialog.open_directory_dialog(
            initialdir=self._shell_procmon.cwd())

        if d is None:
            return

        self._insert_or_replace_input_text(shlex.quote(d))

    def _input_text_insert_file_path(self):
        f = _filedialog.open_file_dialog(
            initialdir=self._shell_procmon.cwd())

        if f is None:
            return

        self._insert_or_replace_input_text(shlex.quote(f))

    def _input_text_insert_file_path_save(self):
        f = _filedialog.open_file_save_dialog(
            initialdir=self._shell_procmon.cwd())

        if f is None:
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

        self._image_pane_window.protocol('WM_DELETE_WINDOW', on_closing)

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

        if width < 0:
            return

        if height < 0:
            return

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

    def _shell_return_code_message(self, return_code):
        self._write_stdout_output(
            f'\nShell Process Terminated, Exit Code: {return_code}\n')

    def _shell_restarting_commence_message(self):
        self._write_stdout_output('Restarting Shell Process...\n')
        return self._last_known_working_dir

    def _shell_restarting_finish_message(self):
        self._update_cwd_title(self._last_known_working_dir)

        self._write_stdout_output('Shell Process Started.\n'
                                  '======================\n')

    def _load_input_entry_text(self):
        fn = _filedialog.open_file_dialog(
            initialfile='config.dgen',
            defaultextension='.dgen',
            filetypes=[('Config Script', '*.dgen'),
                       ('Text File', '*.txt')])

        if fn is None:
            return

        self.load_file(fn)

    def load_file(self, file_name):
        with open(file_name, 'rt') as f:
            self._input_text.text.replace('1.0', tk.END, f.read())
        self._input_text.text.mark_set('insert', '1.0')
        self.multiline_mode(True)

    def _save_input_entry_text(self):
        fn = _filedialog.open_file_save_dialog(
            initialfile='config.dgen',
            defaultextension='.dgen',
            filetypes=[('Config Script', '*.dgen'),
                       ('Text File', '*.txt')])

        if fn is None:
            return

        with open(fn, mode='w', encoding='utf-8') as f:
            f.write(self._input_text.text.get('1.0', 'end-1c'))
            f.close()

    def _undo_input_entry(self):
        self._input_text.gen_undo_event()

    def _redo_input_entry(self):
        self._input_text.gen_redo_event()

    def _cut_input_entry_selection(self):
        self._input_text.gen_cut_event()

    def _copy_input_entry_selection(self):
        self._input_text.gen_copy_event()

    def _paste_input_entry(self):
        self._input_text.gen_paste_event()

    def _delete_input_entry_selection(self):
        self._input_text.gen_delete_event()

    def _select_all_input_entry(self):
        self._input_text.gen_selectall_event()

    def _copy_output_text_selection(self):
        self._output_text.text.event_generate('<<Copy>>')

    def _select_all_output_text(self):
        self._output_text.text.focus_set()
        self._output_text.text.event_generate('<<SelectAll>>')

    def _save_output_text(self):
        f = _filedialog.open_file_save_dialog(
            initialfile='log.txt',
            defaultextension='.txt',
            filetypes=[('Text Documents', '*.txt')])

        if f is None:
            return

        with open(f, mode='w', encoding='utf-8') as f:
            f.write(self._output_text.text.get('1.0', 'end-1c'))
            f.close()

    def _save_output_text_selection(self):
        f = _filedialog.open_file_save_dialog(
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

        self._output_text.text.tag_remove(
            'error', '1.0', 'end')

        self._output_text.text.delete('1.0', tk.END)
        self._output_text.text.config(state=tk.DISABLED)

    def _check_text_for_latest_image(self, text):
        match = re.match(
            r'Wrote Image File: "(.*?)"|\\image_process: Wrote Image "(.*?)"|\\image_process: Wrote Frame "(.*?)"',
            text)
        if match is not None:
            mentioned_path = ''.join(filter(None, match.groups()))
            if os.path.isabs(mentioned_path):
                self._image_pane_load_image(mentioned_path)
            else:
                path = os.path.join(
                    self._shell_procmon.cwd(deep=True), mentioned_path)
                self._image_pane_load_image(path)

    def _update_cwd_title(self, directory=None):
        if directory is None:
            directory = self._shell_procmon.cwd()
        self.title(f'Dgenerate Console: {directory}')
        self._last_known_working_dir = directory

    def _add_output_line(self, text, tag=None):
        if text.startswith('Working Directory Changed To: '):
            # update using shallow psutil query
            self._update_cwd_title()

        self._check_text_for_latest_image(text)

        scroll = self._output_text.text.yview()[1] == 1.0 or \
                 self._auto_scroll_on_output_check_var.get()

        output_lines = int(self._output_text.text.index('end-1c').split('.')[0])

        self._output_text.text.config(state=tk.NORMAL)

        if output_lines + 1 > self._max_output_lines:
            remove_range = ('1.0', f'{output_lines - self._max_output_lines}.0')
            self._output_text.text.tag_remove('error', *remove_range)
            self._output_text.text.delete(*remove_range)

        clean_text = _textprocessing.remove_terminal_escape_sequences(text).rstrip() + '\n'
        replace_range = ('end-2c linestart', 'end-1c')

        if self._next_text_update_line_return:
            self._output_text.text.tag_remove('error', *replace_range)
            self._output_text.text.replace(*replace_range, clean_text)
        elif self._next_text_update_line_escape:
            if text.strip():
                self._output_text.text.tag_remove('error', *replace_range)
                self._output_text.text.replace(*replace_range, clean_text)
        else:
            self._output_text.text.insert(tk.END, clean_text)

        if tag is not None:
            self._output_text.text.tag_add(tag, f'end - {len(clean_text) + 1} chars', 'end-1c')

        self._output_text.text.config(state=tk.DISABLED)

        self._next_text_update_line_escape = text.endswith('\u001B[A\r')
        self._next_text_update_line_return = text.endswith('\r')

        if scroll:
            self._output_text.text.see(tk.END)

    def _write_stdout_output(self, text: bytes | str):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stdout.buffer.write(text)
        sys.stdout.flush()
        self._add_output_line(text.decode('utf-8'))

    def _write_stderr_output(self, text: bytes | str):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stderr.buffer.write(text)
        sys.stderr.flush()
        self._add_output_line(text.decode('utf-8'), tag='error')

    def _show_previous_command(self, event):
        if event.state & 0x0004:
            # ignore Ctrl modifier
            return

        if self._multi_line_input_check_var.get():
            return

        if self._current_command_index > 0:
            self._current_command_index -= 1
            self._input_text.text.edit_separator()
            self._input_text.text.replace('1.0', tk.END, self._command_history[self._current_command_index])
            self._input_text.text.see(tk.END)
            self._input_text.text.mark_set(tk.INSERT, tk.END)
        return 'break'

    def _show_next_command(self, event):
        if event.state & 0x0004:
            # ignore Ctrl modifier
            return

        if self._multi_line_input_check_var.get():
            return

        if self._current_command_index < len(self._command_history) - 1:
            self._current_command_index += 1
            self._input_text.text.edit_separator()
            self._input_text.text.replace('1.0', tk.END, self._command_history[self._current_command_index])
            self._input_text.text.see(tk.END)
            self._input_text.text.mark_set(tk.INSERT, tk.END)
        return 'break'

    def _load_settings(self):
        settings_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_settings')
        if settings_path.exists():
            with settings_path.open('r') as file:
                config = json.load(file)
                self._theme_menu_var.set(config.get('theme', 'dgenerate'))
                self._auto_scroll_on_run_check_var.set(config.get('auto_scroll_on_run', True))
                self._auto_scroll_on_output_check_var.set(config.get('auto_scroll_on_output', False))
                self._word_wrap_input_check_var.set(config.get('word_wrap_input', True))
                self._word_wrap_output_check_var.set(config.get('word_wrap_output', True))
        else:
            self._theme_menu_var.set('dgenerate')

    def save_settings(self):
        settings_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_settings')
        with settings_path.open('w') as file:
            config = {
                'theme': self._theme_menu_var.get(),
                'auto_scroll_on_run': self._auto_scroll_on_run_check_var.get(),
                'auto_scroll_on_output': self._auto_scroll_on_output_check_var.get(),
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

        try:
            self._shell_procmon.stdin_pipe.write(
                (user_input + '\n\n' + '\\reset_lineno' + '\n\n').encode('utf-8'))
        except StdinPipeFullError:
            self._write_stderr_output(
                'WARNING: The command queue is full, '
                'please wait before submitting more commands, '
                'or kill the interpreter.')

        if self._command_history:
            if self._command_history[-1] != user_input:
                # ignoredups
                self._command_history.append(user_input)
        else:
            self._command_history.append(user_input)

        self._current_command_index = len(self._command_history)
        self._save_command_history()

        return 'break'

    def multiline_mode(self, state: bool):
        self._multi_line_input_check_var.set(state)

    def destroy(self) -> None:
        self.save_settings()
        self.kill_shell_process()
        super().destroy()


def main(args: collections.abc.Sequence[str]):
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            # The application is running as a bundled pyinstaller executable
            application_path = os.path.dirname(sys.executable)

            # hack for Windows desktop shortcut launch, would rather
            # the working directory be the users home directory
            # than in the install directory of the program
            if os.path.abspath(os.getcwd()) == os.path.abspath(application_path):
                os.chdir(os.path.expanduser('~'))

    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    if args:
        if not os.path.exists(args[0]):
            print(f'File not found: {args[0]}',
                  file=sys.stderr)
            sys.exit(1)

    app = DgenerateConsole()

    def signal_handler(sig, frame):
        print('Exiting dgenerate console UI due to keyboard interrupt!',
              file=sys.stderr)
        app.destroy()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    if args:
        try:
            app.load_file(args[0])
        except (FileNotFoundError, IsADirectoryError):
            print(f'File not found: {args[0]}',
                  file=sys.stderr)
            app.destroy()
            sys.exit(1)

    app.mainloop()
