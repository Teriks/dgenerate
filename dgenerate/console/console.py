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
import shutil
import signal
import subprocess
import sys
import tkinter as tk
import typing

import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
import charset_normalizer

import dgenerate.console.argumentselect as _argumentselect
import dgenerate.console.directiveselect as _directiveselect
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.finddialog as _finddialog
import dgenerate.console.functionselect as _functionselect
import dgenerate.console.imageprocessorselect as _imageprocessorselect
import dgenerate.console.latentsprocessorselect as _latentsprocessorselect
import dgenerate.console.imageseedselect as _imageseedselect
import dgenerate.console.karrasschedulerselect as _karrasschedulerselect
import dgenerate.console.promptupscalerselect as _promptupscalerselect
import dgenerate.console.promptweighterselect as _promptweighterselect
import dgenerate.console.quantizerselect as _quantizerselect
import dgenerate.console.submodelselect as _submodelselect
import dgenerate.console.recipesform as _recipesform
import dgenerate.console.resources as _resources
import dgenerate.textprocessing as _textprocessing
from dgenerate.console.codeview import DgenerateCodeView
from dgenerate.console.procmon import ProcessMonitor
from dgenerate.console.scrolledtext import ScrolledText
from dgenerate.console.stdinpipe import StdinPipeFullError

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
        self._create_edit_menu(self._edit_menu)

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

        self._offline_mode_var = tk.BooleanVar(value=False)

        self._offline_mode_var.trace_add('write',
                                       lambda *a: self._update_shell_options_state())

        self._run_menu.add_checkbutton(label='Offline Mode',
                                       variable=self._offline_mode_var)

        self._debug_mode_var = tk.BooleanVar(value=False)

        self._debug_mode_var.trace_add('write',
                                       lambda *a: self._update_shell_options_state())

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

        # Create the input text context menu
        self._input_text_context = tk.Menu(self._input_text, tearoff=0)
        self._create_edit_menu(self._input_text_context)

        if platform.system() == 'Darwin':
            # these are not native
            self._input_text.text.bind('<Control-x>',
                                       lambda e: self._cut_input_entry_selection())
            self._input_text.text.bind('<Control-c>',
                                       lambda e: self._copy_input_entry_selection())
            self._input_text.text.bind('<Control-v>',
                                       lambda e: self._paste_input_entry())

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
        self._output_text_context.add_command(label='To Top',
                                              accelerator='Ctrl+Up Arrow',
                                              command=
                                              lambda: self._output_text.text.see('1.0'))
        self._output_text_context.add_command(label='To Bottom',
                                              accelerator='Ctrl+Down Arrow',
                                              command=
                                              lambda: self._output_text.text.see(tk.END))
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Preview Selected Image',
                                              command=lambda: self._preview_selected_image(self._output_text.text))

        # Patch the output text context menu to enable/disable Preview Selected Image
        og_output_popup = self._output_text_context.tk_popup
        def patch_output_tk_popup(*args, **kwargs):
            selected_text = self._get_selected_text(self._output_text.text)
            if selected_text and self._is_valid_image_path(selected_text):
                self._output_text_context.entryconfigure('Preview Selected Image', state=tk.NORMAL)
            else:
                self._output_text_context.entryconfigure('Preview Selected Image', state=tk.DISABLED)
            og_output_popup(*args, **kwargs)
        self._output_text_context.tk_popup = patch_output_tk_popup

        self._paned_window_vertical.add(self._output_text)

        # Optional image pane

        self._displayed_image = None
        self._displayed_image_path = None

        self._image_pane_window = None
        self._image_pane_window_last_pos = None
        self._image_pane_last_right_clicked_coords = None
        
        # Bounding box selection state
        self._bbox_selection_thickness = 3
        self._bbox_selection_dash_length = 8
        self._bbox_selection_gap_length = 4
        self._bbox_selection_mode = False
        self._bbox_selection_format = None  # 'csv' or 'x'
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._original_displayed_image = None  # Keep original image for restoration
        
        self._image_pane = tk.Frame(self._paned_window_horizontal, bg='black')

        self._image_pane_image_label = tk.Label(self._image_pane, bg='black')
        self._image_pane_image_label.pack(fill=tk.BOTH, expand=True)
        self._image_pane_image_label.bind('<Configure>',
                         lambda e: self._resize_image_pane_image(self._image_pane_image_label))
        
        # Bounding box selection mouse events
        self._image_pane_image_label.bind('<Button-1>', 
                         lambda e: self._handle_image_left_click(e, self._image_pane_image_label))
        self._image_pane_image_label.bind('<B1-Motion>', 
                         lambda e: self._handle_image_drag(e, self._image_pane_image_label))
        self._image_pane_image_label.bind('<ButtonRelease-1>', 
                         lambda e: self._handle_image_left_release(e, self._image_pane_image_label))
        
        # Key binding for canceling selection (bind to main window)
        self.bind('<Escape>', lambda e: self._cancel_bbox_selection())
        self.bind('<KeyPress-Escape>', lambda e: self._cancel_bbox_selection())

        self._image_pane_context = tk.Menu(self._image_pane, tearoff=0)

        self._install_common_image_pane_context_options(self._image_pane_context)
        
        self._image_pane_context.add_separator()

        if self._install_show_in_directory_entry(
                self._image_pane_context, lambda: self._displayed_image_path):
            self._image_pane_context.add_separator()

        self._image_pane_context.add_command(
            label='Hide Image Pane',
            command=lambda: self._image_pane_visible_var.set(False))

        self._image_pane_context.add_command(
            label='Make Window',
            command=lambda: self._image_pane_window_visible_var.set(True))

        self._image_pane_image_label.bind(
            '<Button-3>',
            lambda e: self._show_image_pane_context_menu(
                e,
                self._image_pane_image_label,
                self._image_pane_context
            )
        )

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

    def _update_shell_options_state(self):
        self._shell_procmon.popen_args = \
            [[DGENERATE_EXE, '--shell'] +
             (['-v'] if self._debug_mode_var.get() else []) +
             (['-ofm'] if self._offline_mode_var.get() else [])]

        self.kill_shell_process(restart=True)

    def kill_shell_process(self, restart=False):
        if restart:
            self._shell_procmon.restart()
        else:
            self._shell_procmon.close()

    def _format_code(self):
        try:
            text = self._input_text.text.get('1.0', 'end-1c')
            if text.strip():  # check that it's not just whitespace
                formatted_text = '\n'.join(_textprocessing.format_dgenerate_config(iter(text.split('\n'))))
                self._input_text.clear()
                self._input_text.text.insert('1.0', formatted_text)
        except Exception as e:
            self._write_stderr_output(str(e))

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
        _recipesform.request_recipe(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_argument(self):
        _argumentselect.request_argument(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_directive(self):
        _directiveselect.request_directive(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_function(self):
        _functionselect.request_function(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_image_seed(self):
        _imageseedselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_karras_scheduler(self):
        _karrasschedulerselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_image_processor(self):
        _imageprocessorselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_latents_processor(self):
        _latentsprocessorselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_prompt_upscaler(self):
        _promptupscalerselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_prompt_weighter(self):
        _promptweighterselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_quantizer(self):
        _quantizerselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_sub_model(self):
        _submodelselect.request_uri(
            master=self, insert=self._insert_or_replace_input_text
        )

    def _input_text_insert_directory_path(self):
        d = _filedialog.open_directory_dialog(
            initialdir=self._shell_procmon.cwd())

        if d is None:
            return

        self._insert_or_replace_input_text(_textprocessing.shell_quote(d))

    def _input_text_insert_file_path(self):
        f = _filedialog.open_file_dialog(
            initialdir=self._shell_procmon.cwd())

        if f is None:
            return

        self._insert_or_replace_input_text(_textprocessing.shell_quote(f))

    def _input_text_insert_file_path_save(self):
        f = _filedialog.open_file_save_dialog(
            initialdir=self._shell_procmon.cwd())

        if f is None:
            return

        self._insert_or_replace_input_text(_textprocessing.shell_quote(f))

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
        self._image_pane_window_image_label = tk.Label(image_pane, bg='black')
        self._image_pane_window_image_label.pack(fill=tk.BOTH, expand=True)
        self._image_pane_window_image_label.bind(
            '<Configure>', lambda e: self._resize_image_pane_image(
                self._image_pane_window_image_label)
        )
        
        # Bounding box selection mouse events for window
        self._image_pane_window_image_label.bind('<Button-1>', 
                         lambda e: self._handle_image_left_click(e, self._image_pane_window_image_label))
        self._image_pane_window_image_label.bind('<B1-Motion>', 
                         lambda e: self._handle_image_drag(e, self._image_pane_window_image_label))
        self._image_pane_window_image_label.bind('<ButtonRelease-1>', 
                         lambda e: self._handle_image_left_release(e, self._image_pane_window_image_label))

        image_window_context = tk.Menu(self._image_pane_window, tearoff=0)

        self._install_common_image_pane_context_options(image_window_context)

        image_window_context.add_separator()

        if self._install_show_in_directory_entry(image_window_context,
                                                 lambda: self._displayed_image_path):
            image_window_context.add_separator()

        image_window_context.add_command(label='Make Pane',
                                         command=lambda: self._image_pane_visible_var.set(True))

        self._image_pane_window_image_label.bind(
            '<Button-3>', lambda e:
            self._show_image_pane_context_menu(
                e,
                self._image_pane_window_image_label,
                image_window_context
            )
        )

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

        # Clear any existing bounding box selection
        if self._bbox_selection_mode:
            self._end_bbox_selection()

        try:
            self._displayed_image = PIL.Image.open(image_path)
            
            # Ensure image is in RGB mode for compatibility
            if self._displayed_image.mode not in ['RGB', 'RGBA']:
                self._displayed_image = self._displayed_image.convert('RGB')
                
            self._original_displayed_image = self._displayed_image.copy()  # Keep original
            self._displayed_image_path = image_path

            self._image_pane_image_label.event_generate('<Configure>')

            if self._image_pane_window is not None:
                self._image_pane_window_image_label.event_generate('<Configure>')
        except Exception as e:
            self._write_stderr_output(f"Error loading image: {e}\n")

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

    def _try_load_utf8(self, file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return True, f.read()
        except UnicodeDecodeError:
            return False, None

    def _load_to_utf8(self, file_name):
        is_utf8, content = self._try_load_utf8(file_name)
        if is_utf8:
            return 'utf-8', content

        result = charset_normalizer.from_path(file_name).best()

        return result.encoding, result.output().decode('utf-8')

    def load_file(self, file_name):
        encoding, content = self._load_to_utf8(file_name)

        self._input_text.text.replace('1.0', tk.END, content)

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
            with settings_path.open('r', encoding='utf-8') as file:
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
        with settings_path.open('w', encoding='utf-8') as file:
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
            with history_path.open('r', encoding='utf-8') as file:
                self._command_history = json.load(file)[-self._max_command_history:]
                self._current_command_index = len(self._command_history)

    def _save_command_history(self):
        if self._max_command_history == 0:
            return

        history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        with history_path.open('w', encoding='utf-8') as file:
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
        
        # Clean up bounding box selection
        if self._bbox_selection_mode:
            self._end_bbox_selection()
        
        super().destroy()

    def _create_edit_menu(self, menu):
        # Standard editing operations
        menu.add_command(label='Cut', accelerator='Ctrl+X',
                         command=self._cut_input_entry_selection)
        menu.add_command(label='Copy', accelerator='Ctrl+C',
                         command=self._copy_input_entry_selection)
        menu.add_command(label='Paste', accelerator='Ctrl+V',
                         command=self._paste_input_entry)
        menu.add_command(label='Delete', accelerator='DEL',
                         command=self._delete_input_entry_selection)
        menu.add_command(label='Select All', accelerator='Ctrl+A',
                         command=self._select_all_input_entry)
        menu.add_separator()

        # Undo/Redo operations
        menu.add_command(label='Undo', accelerator='Ctrl+Z',
                         command=self._undo_input_entry)
        menu.add_command(label='Redo', accelerator='Ctrl+Shift+Z',
                         command=self._redo_input_entry)
        menu.add_separator()

        # Search submenu
        search_submenu = tk.Menu(menu, tearoff=0)
        search_submenu.add_command(label='Find',
                                   accelerator='Ctrl+F',
                                   command=lambda:
                                   _finddialog.open_find_dialog(
                                       self,
                                       'Find In Input',
                                       self._input_text.text))
        search_submenu.add_command(label='Replace',
                                   accelerator='Ctrl+R',
                                   command=lambda:
                                   _finddialog.open_find_replace_dialog(
                                       self,
                                       'Replace In Input',
                                       self._input_text.text))
        menu.add_cascade(label='Find / Replace', menu=search_submenu)

        # Format option
        menu.add_command(label='Format Code',
                         accelerator='Ctrl+Shift+F',
                         command=self._format_code)
        menu.add_separator()

        # Code submenu
        code_submenu = tk.Menu(menu, tearoff=0)
        code_submenu.add_command(label='Recipe',
                                 command=self._input_text_insert_recipe)
        code_submenu.add_command(label='Argument',
                                 command=self._input_text_insert_argument)
        code_submenu.add_command(label='Directive',
                                 command=self._input_text_insert_directive)
        code_submenu.add_command(label='Function',
                                 command=self._input_text_insert_function)
        menu.add_cascade(label='Insert Code', menu=code_submenu)

        # URI submenu
        uri_submenu = tk.Menu(menu, tearoff=0)
        uri_submenu.add_command(label='Image Seed URI',
                                command=self._input_text_insert_image_seed)
        uri_submenu.add_command(label='Karras Scheduler URI',
                                command=self._input_text_insert_karras_scheduler)
        uri_submenu.add_command(label='Image Processor URI',
                                command=self._input_text_insert_image_processor)
        uri_submenu.add_command(label='Latents Processor URI',
                                command=self._input_text_insert_latents_processor)
        uri_submenu.add_command(label='Prompt Upscaler URI',
                                command=self._input_text_insert_prompt_upscaler)
        uri_submenu.add_command(label='Prompt Weighter URI',
                                command=self._input_text_insert_prompt_weighter)
        uri_submenu.add_command(label='Sub Model URI',
                                command=self._input_text_insert_sub_model)
        uri_submenu.add_command(label='Quantizer URI',
                                command=self._input_text_insert_quantizer)
        menu.add_cascade(label='Insert URI', menu=uri_submenu)

        # Path submenu
        path_submenu = tk.Menu(menu, tearoff=0)
        path_submenu.add_command(label='File Path (Open)',
                                 command=self._input_text_insert_file_path)
        path_submenu.add_command(label='File Path (Save)',
                                 command=self._input_text_insert_file_path_save)
        path_submenu.add_command(label='Directory Path',
                                 command=self._input_text_insert_directory_path)
        menu.add_cascade(label='Insert Path', menu=path_submenu)

        # Add separator and Preview Selected Image option
        menu.add_separator()
        menu.add_command(label='Preview Selected Image',
                         command=lambda: self._preview_selected_image(self._input_text.text))

        def menu_post(*args, **kwargs):
            selected_text = self._get_selected_text(self._input_text.text)
            if selected_text and self._is_valid_image_path(selected_text):
                menu.entryconfigure(
                    'Preview Selected Image', state=tk.NORMAL)
            else:
               menu.entryconfigure(
                    'Preview Selected Image', state=tk.DISABLED)

        menu.configure(postcommand=menu_post)

        return menu

    def _install_common_image_pane_context_options(self, context_menu: tk.Menu):

        context_menu.add_command(
            label='Copy Coordinates "x"',
            command=lambda: self._copy_image_coordinates('x'))

        context_menu.add_command(
            label='Copy Coordinates CSV',
            command=lambda: self._copy_image_coordinates(','))

        context_menu.add_separator()

        context_menu.add_command(
            label='Copy Bounding Box "x"',
            command=lambda: self._start_bbox_selection('x')
        )

        context_menu.add_command(
            label='Copy Bounding Box CSV',
            command=lambda: self._start_bbox_selection('csv')
        )

        context_menu.add_separator()

        context_menu.add_command(
            label='Copy Path',
            command=self._copy_image_path)

        context_menu.add_separator()

        context_menu.add_command(
            label='Load Image',
            command=self._load_image_manually)


    def _show_image_pane_context_menu(self, event, image_label: tk.Label, context_menu: tk.Menu):
        # Capture coordinates from the right-click event
        if self._displayed_image is not None:
            # Get the label widget (first child of the first child - Frame -> Label)
            image_x, image_y = self._widget_to_image_coordinates(event.x, event.y, image_label)
            if image_x is not None and image_y is not None:
                self._image_pane_last_right_clicked_coords = (image_x, image_y)

        # Enable/disable Load Image (always enabled)
        context_menu.entryconfigure('Load Image', state=tk.NORMAL)
        
        # Enable/disable Copy Coordinates based on whether coordinates are available
        if self._image_pane_last_right_clicked_coords is not None:
            context_menu.entryconfigure('Copy Coordinates "x"', state=tk.NORMAL)
            context_menu.entryconfigure('Copy Coordinates CSV', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Coordinates "x"', state=tk.DISABLED)
            context_menu.entryconfigure('Copy Coordinates CSV', state=tk.DISABLED)

        # Enable/disable Copy Bounding Box based on whether image is loaded
        if self._displayed_image is not None:
            context_menu.entryconfigure('Copy Bounding Box "x"', state=tk.NORMAL)
            context_menu.entryconfigure('Copy Bounding Box CSV', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Bounding Box "x"', state=tk.DISABLED)
            context_menu.entryconfigure('Copy Bounding Box CSV', state=tk.DISABLED)

        # Enable/disable Copy Path based on whether path is available
        if self._displayed_image_path is not None:
            context_menu.entryconfigure('Copy Path', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Path', state=tk.DISABLED)
        
        context_menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def _widget_to_image_coordinates(self, widget_x, widget_y, label):
        if self._displayed_image is None:
            return None, None

        label_width = label.winfo_width()
        label_height = label.winfo_height()
        
        image_width = self._displayed_image.width
        image_height = self._displayed_image.height
        
        # Calculate the displayed image size (same logic as _resize_image_pane_image)
        image_aspect = image_width / image_height
        label_aspect = label_width / label_height
        
        if image_aspect > label_aspect:
            # Image is wider than label - fit to width
            display_width = label_width
            display_height = int(display_width / image_aspect)
        else:
            # Image is taller than label - fit to height
            display_height = label_height
            display_width = int(display_height * image_aspect)
        
        # Calculate offsets for centering
        x_offset = (label_width - display_width) // 2
        y_offset = (label_height - display_height) // 2
        
        # Check if click is within the displayed image bounds
        if (widget_x < x_offset or widget_x >= x_offset + display_width or
            widget_y < y_offset or widget_y >= y_offset + display_height):
            return None, None
        
        # Convert to image coordinates
        relative_x = widget_x - x_offset
        relative_y = widget_y - y_offset
        
        image_x = int((relative_x / display_width) * image_width)
        image_y = int((relative_y / display_height) * image_height)
        
        # Ensure coordinates are within bounds
        image_x = max(0, min(image_x, image_width - 1))
        image_y = max(0, min(image_y, image_height - 1))
        
        return image_x, image_y

    def _copy_image_coordinates(self, sep):
        if self._image_pane_last_right_clicked_coords is None:
            return
        
        x, y = self._image_pane_last_right_clicked_coords
        coordinate_text = f"{x}{sep}{y}"
        
        try:
            self.clipboard_clear()
            self.clipboard_append(coordinate_text)
        except Exception as e:
            self._write_stderr_output(f"Failed to copy coordinates to clipboard: {e}\n")

    def _copy_image_path(self):
        if self._displayed_image_path is None:
            return

        try:
            self.clipboard_clear()
            self.clipboard_append(
                pathlib.Path(self._displayed_image_path).as_posix()
            )
        except Exception as e:
            self._write_stderr_output(f"Failed to copy image path to clipboard: {e}\n")

    def _load_image_manually(self):
        f = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']),
            initialdir=self._shell_procmon.cwd())
        
        if f is None:
            return
        
        try:
            self._image_pane_load_image(f)
            self._write_stdout_output(f"Manually loaded image: {f}\n")
        except Exception as e:
            self._write_stderr_output(f"Failed to load image: {e}\n")

    @staticmethod
    def _shell_unquote_path(path: str) -> str:
        try:
            parsed = _textprocessing.shell_parse(
                path.strip(),
                expand_home=False,
                expand_vars=False,
                expand_glob=False
            )
            return parsed[0] if parsed else path.strip()
        except _textprocessing.ShellParseSyntaxError:
            return path.strip()

    def _is_valid_image_path(self, path: str) -> bool:
        if not path or not path.strip():
            return False
        
        path = self._shell_unquote_path(path)
        
        if not os.path.exists(path):
            abs_path = os.path.join(self._shell_procmon.cwd(deep=True), path)
            if not os.path.exists(abs_path):
                return False
            path = abs_path
        
        if not os.path.isfile(path):
            return False
        
        try:
            schema = _resources.get_schema('mediaformats')
            supported_extensions = schema['images-in']
            file_ext = os.path.splitext(path)[1].lower().lstrip('.')
            return file_ext in supported_extensions
        except Exception:
            return False

    def _preview_selected_image(self, text_widget):
        try:
            selected_text = text_widget.selection_get()
        except tk.TclError:
            return
        
        if not self._is_valid_image_path(selected_text):
            return
        
        path = self._shell_unquote_path(selected_text)
        
        if not os.path.isabs(path):
            path = os.path.join(self._shell_procmon.cwd(deep=True), path)
        
        if not self._image_pane_visible_var.get() and not self._image_pane_window_visible_var.get():
            self._image_pane_visible_var.set(True)
        
        try:
            self._image_pane_load_image(path)
            self._write_stdout_output(f"Manually loaded image: {path}\n")
        except Exception as e:
            self._write_stderr_output(f"Failed to load image: {e}\n")

    @staticmethod
    def _get_selected_text(text_widget: tk.Text):
        try:
            return text_widget.selection_get()
        except tk.TclError:
            return None

    def _start_bbox_selection(self, format_type):
        if self._displayed_image is None:
            return
            
        self._bbox_selection_mode = True
        self._bbox_selection_format = format_type
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        
        # Ensure we have the original image
        if self._original_displayed_image is None:
            self._original_displayed_image = self._displayed_image.copy()
        
        self._write_stdout_output(
            "Bounding box selection mode started. Left-click and drag to select area. Press Escape to cancel.\n")


    def _handle_image_left_click(self, event, label):
        if not self._bbox_selection_mode:
            return
            
        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y, label)
        if image_x is None or image_y is None:
            return
            
        self._bbox_start_coords = (image_x, image_y)
        self._bbox_end_coords = (image_x, image_y)
        
        # Start drawing the selection rectangle on the image
        self._draw_selection_on_image()
        
        return "break"

    def _handle_image_drag(self, event, label):
        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return
            
        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y, label)
        if image_x is None or image_y is None:
            return
            
        self._bbox_end_coords = (image_x, image_y)
        # Update the selection rectangle on the image
        self._draw_selection_on_image()
        
        return "break"

    def _handle_image_left_release(self, event, label):
        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return
            
        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y, label)
        if image_x is None or image_y is None:
            return
            
        self._bbox_end_coords = (image_x, image_y)
        
        # Complete the selection
        self._complete_bbox_selection()
        
        return "break"

    def _draw_selection_on_image(self):
        if (self._bbox_start_coords is None or 
            self._bbox_end_coords is None or
            self._original_displayed_image is None):
            return
            
        try:
            # Create a copy of the original image to draw on
            working_image = self._original_displayed_image.copy()
            
            # Ensure image is in RGB mode for drawing
            if working_image.mode not in ['RGB', 'RGBA']:
                working_image = working_image.convert('RGB')
                
            draw = PIL.ImageDraw.Draw(working_image)
        except Exception as e:
            self._write_stderr_output(f"Error creating drawing context: {e}\n")
            return
        
        x1, y1 = self._bbox_start_coords
        x2, y2 = self._bbox_end_coords
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        try:
            # Fixed visual thickness values (in screen pixels)
            visual_thickness = self._bbox_selection_thickness
            visual_dash_length = self._bbox_selection_dash_length
            visual_gap_length = self._bbox_selection_gap_length
            
            # Get the current display size to calculate how the image is scaled
            display_width, display_height = self._get_current_image_display_size()
            
            if display_width is None or display_height is None:
                # Fallback to fixed values if we can't get display size
                original_thickness = self._bbox_selection_thickness
                original_dash_length = self._bbox_selection_dash_length
                original_gap_length = self._bbox_selection_gap_length
            else:
                original_width = working_image.width
                original_height = working_image.height

                width_scale = display_width / original_width
                height_scale = display_height / original_height

                display_scale = min(width_scale, height_scale)

                # Calculate scaled values maintaining float precision
                original_thickness = visual_thickness / display_scale
                original_dash_length = visual_dash_length / display_scale
                original_gap_length = visual_gap_length / display_scale
                
                # Ensure minimum values that maintain visual distinction
                # The gap should be at least 1 pixel to ensure dashes don't merge
                min_gap = 1.0
                if original_gap_length < min_gap:
                    # Scale all values proportionally to maintain ratio
                    scale_factor = min_gap / original_gap_length
                    original_thickness = max(1.0, original_thickness * scale_factor)
                    original_dash_length = max(2.0, original_dash_length * scale_factor)
                    original_gap_length = min_gap

            # Top line
            self._draw_dotted_line(draw, x1, y1, x2, y1, original_dash_length, original_gap_length, original_thickness)
            # Right line
            self._draw_dotted_line(draw, x2, y1, x2, y2, original_dash_length, original_gap_length, original_thickness)
            # Bottom line
            self._draw_dotted_line(draw, x2, y2, x1, y2, original_dash_length, original_gap_length, original_thickness)
            # Left line
            self._draw_dotted_line(draw, x1, y2, x1, y1, original_dash_length, original_gap_length, original_thickness)

            self._displayed_image = working_image

            # Trigger redisplay
            self._image_pane_image_label.event_generate('<Configure>')
            if self._image_pane_window is not None:
                self._image_pane_window_image_label.event_generate('<Configure>')

        except Exception as e:
            self._write_stderr_output(f"Error drawing selection rectangle: {e}\n")
            return

    def _get_current_image_label_size(self):
        # Determine which label to use
        if hasattr(self, '_image_pane_image_label') and self._image_pane_image_label.winfo_exists():
            label = self._image_pane_image_label
        elif hasattr(self, '_image_pane_window_image_label') and self._image_pane_window_image_label.winfo_exists():
            label = self._image_pane_window_image_label
        else:
            return None, None
            
        label_width = label.winfo_width()
        label_height = label.winfo_height()
        
        # Return None if label hasn't been sized yet
        if label_width <= 0 or label_height <= 0:
            return None, None
            
        return label_width, label_height

    def _get_current_image_display_size(self):
        if self._displayed_image is None:
            return None, None
            
        # Get label size
        label_width, label_height = self._get_current_image_label_size()
        
        if label_width is None or label_height is None:
            return None, None
            
        image_width = self._displayed_image.width
        image_height = self._displayed_image.height
        
        image_aspect = image_width / image_height
        label_aspect = label_width / label_height
        
        if image_aspect > label_aspect:
            display_width = label_width
            display_height = int(display_width / image_aspect)
        else:
            display_height = label_height
            display_width = int(display_height * image_aspect)
            
        return display_width, display_height

    @staticmethod
    def _draw_dotted_line(
             draw: PIL.ImageDraw.Draw,
             x1: int,
             y1: int,
             x2: int,
             y2: int,
             dash_length: float,
             gap_length: float,
             line_thickness: float
     ):
        try:
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate true line length using Pythagorean theorem
            length = (dx * dx + dy * dy) ** 0.5
            
            if length == 0:
                return
                
            # Normalize direction vectors
            dx_norm = dx / length
            dy_norm = dy / length
                
            # Draw dotted line
            current_pos = 0
            dash_count = 0
            
            while current_pos < length:
                # Calculate dash start and end
                dash_start = current_pos
                dash_end = min(length, current_pos + dash_length)
                
                if dash_end > dash_start:
                    # Calculate actual coordinates for this dash
                    start_x = x1 + int(dash_start * dx_norm)
                    start_y = y1 + int(dash_start * dy_norm)
                    end_x = x1 + int(dash_end * dx_norm)
                    end_y = y1 + int(dash_end * dy_norm)
                    
                    # Draw alternating black and white dashes for contrast
                    # Use scaled thickness - convert to int for draw.line
                    int_thickness = max(1, int(round(line_thickness)))
                    
                    # Determine color based on dash count (alternate between black and white)
                    color = 'black' if dash_count % 2 == 0 else 'white'
                    
                    draw.line([start_x, start_y, end_x, end_y], fill=color, width=int_thickness)
                
                # Move to next dash position
                current_pos = dash_end + gap_length
                dash_count += 1
        except Exception:
            pass

    def _image_to_widget_coordinates(self, image_x, image_y):
        if self._displayed_image is None:
            return None

        label_width, label_height = self._get_current_image_label_size()
        
        image_width = self._displayed_image.width
        image_height = self._displayed_image.height
        
        # Calculate the displayed image size (same logic as _resize_image_pane_image)
        image_aspect = image_width / image_height
        label_aspect = label_width / label_height
        
        if image_aspect > label_aspect:
            display_width = label_width
            display_height = int(display_width / image_aspect)
        else:
            display_height = label_height
            display_width = int(display_height * image_aspect)
        
        # Calculate offsets for centering
        x_offset = (label_width - display_width) // 2
        y_offset = (label_height - display_height) // 2
        
        # Convert image coordinates to widget coordinates
        widget_x = int((image_x / image_width) * display_width) + x_offset
        widget_y = int((image_y / image_height) * display_height) + y_offset
        
        return widget_x, widget_y


    def _complete_bbox_selection(self):
        if (self._bbox_start_coords is None or 
            self._bbox_end_coords is None or 
            self._bbox_selection_format is None):
            return
            
        x1, y1 = self._bbox_start_coords
        x2, y2 = self._bbox_end_coords
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Format the bounding box coordinates
        if self._bbox_selection_format == 'csv':
            bbox_text = f"{x1},{y1},{x2},{y2}"
        else:  # 'x' format
            bbox_text = f"{x1}x{y1}x{x2}x{y2}"
            
        # Copy to clipboard
        try:
            self.clipboard_clear()
            self.clipboard_append(bbox_text)
            self._write_stdout_output(f"Bounding box copied to clipboard: {bbox_text}\n")
        except Exception as e:
            self._write_stderr_output(f"Failed to copy bounding box to clipboard: {e}\n")
            
        # Clean up selection
        self._end_bbox_selection()

    def _end_bbox_selection(self):
        self._bbox_selection_mode = False
        self._bbox_selection_format = None
        self._bbox_start_coords = None
        self._bbox_end_coords = None

        # Restore original image
        if self._original_displayed_image is not None:
            self._displayed_image = self._original_displayed_image.copy()
            self._image_pane_image_label.event_generate('<Configure>')
            if self._image_pane_window is not None:
                self._image_pane_window_image_label.event_generate('<Configure>')

    def _cancel_bbox_selection(self):
        if self._bbox_selection_mode:
            self._write_stdout_output("Bounding box selection cancelled.\n")
            self._end_bbox_selection()


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
