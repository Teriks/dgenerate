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
import importlib.util
import json
import os
import pathlib
import platform
import re
import select
import signal
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance

import charset_normalizer

import dgenerate.console.argumentselect as _argumentselect
import dgenerate.console.directiveselect as _directiveselect
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.finddialog as _finddialog
import dgenerate.console.fonts as _fonts
import dgenerate.console.functionselect as _functionselect
import dgenerate.console.imageprocessorselect as _imageprocessorselect
import dgenerate.console.assistantform as _assistantform
import dgenerate.console.imageseedselect as _imageseedselect
import dgenerate.console.karrasschedulerselect as _karrasschedulerselect
import dgenerate.console.latentsprocessorselect as _latentsprocessorselect
import dgenerate.console.promptupscalerselect as _promptupscalerselect
import dgenerate.console.promptweighterselect as _promptweighterselect
import dgenerate.console.quantizerselect as _quantizerselect
import dgenerate.console.recipesform as _recipesform
import dgenerate.console.resources as _resources
import dgenerate.console.showindirectory as _showindirectory
import dgenerate.console.submodelselect as _submodelselect
import dgenerate.console.themechrome as _themechrome
import dgenerate.console.themedialog as _themedialog
import dgenerate.console.thememenu as _thememenu
import dgenerate.console.themetext as _themetext
import dgenerate.console.themescrollbar as _themescrollbar
import dgenerate.console.transpileform as _transpileform
import dgenerate.console.initdgeneditor as _initdgeneditor
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing
from dgenerate.console.codeview import DgenerateCodeView
from dgenerate.console.procmon import ProcessMonitor
from dgenerate.console.scrolledtext import ScrolledText
from dgenerate.console.stdinpipe import StdinPipeFullError
from dgenerate.console.tearoffimagepane import TearOffImagePane
from dgenerate.console.terminaltext import TerminalText

DGENERATE_EXE = \
    os.path.splitext(
        os.path.basename(os.path.realpath(sys.argv[0])))[0]


_PREVIEW_OUTPUT_FILE = re.compile(
    r'(?:^|\n)'
    r'(?:Wrote Image File: "(.*?)"'
    r'|Wrote Animation File: "(.*?)"'
    r'|Wrote Frame: "(.*?)"'
    r'|image-process: Wrote Image "(.*?)"'
    r'|image-process: Wrote Frame "(.*?)"'
    r'|image-process: Wrote File "(.*?)"'
    r'|\\image_process: Wrote Image "(.*?)"'
    r'|\\image_process: Wrote Frame "(.*?)"'
    r'|\\image_process: Wrote File "(.*?)")'
)


def _preview_path_from_output(text: str) -> str | None:
    """
    Last media path mentioned in a console output chunk.

    ``\\image_process`` logs the finished animation as ``Wrote File``.
    Frame lines in the same chunk are earlier, so the animation wins.
    """
    found = None
    for match in _PREVIEW_OUTPUT_FILE.finditer(text):
        path = ''.join(filter(None, match.groups()))
        if path:
            found = path
    return found


class DgenerateConsole(tk.Tk):

    def __init__(self):
        super().__init__()
        # Tk shows the window at its default size before geometry is set.
        self.withdraw()

        # Set up proper font defaults to prevent bitmap font fallbacks
        _fonts.set_tkinter_font_defaults()
        _themechrome.install(self)

        # Fit ttk buttons to their text (classic Button behavior) on all platforms.
        _appearance.install_button_fitting(self)

        # Drop the ttk checkbutton focus ring so it matches classic Tk, on all platforms.
        _appearance.install_checkbutton_fit(self)

        self.title('Dgenerate Console')

        width = 1000
        height = 800

        if platform.system() == 'Windows':
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            position_left = (screen_width // 2) - (width // 2)
            position_top = (screen_height // 2) - (height // 2)

            self.geometry(f"{width}x{height}+{position_left}+{position_top}")
            chrome = getattr(self, '_chrome', None)
            if chrome is not None:
                chrome['placed'] = True
        else:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            fit_width = min(width, max(screen_width - 48, 320))
            fit_height = min(height, max(screen_height - 80, 240))
            position_left = max((screen_width - fit_width) // 2, 0)
            position_top = max((screen_height - fit_height) // 2, 0)
            self.geometry(f"{fit_width}x{fit_height}+{position_left}+{position_top}")
            chrome = getattr(self, '_chrome', None)
            if chrome is not None:
                chrome['placed'] = True

        _resources.set_window_icon(self)

        self._shell_process = None
        self._sash_width = 4
        self._settings_dialog = None

        # Create main menu

        # File menu
        self._menu_bar = _thememenu.MenuBar(self)
        self._file_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
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
        self._file_menu.add_separator()
        self._file_menu.add_command(label='Transpile',
                                    command=self._open_transpile_dialog)
        self._transpile_dialog = None

        # Edit menu
        self._edit_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._create_edit_menu(self._edit_menu)

        # Run menu

        self._run_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)

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

        # After Run: Assistant writes a config, then Options, View, and Help.
        self._show_assistant = importlib.util.find_spec('xllamacpp') is not None

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

        self._options_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._appearance_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._console_modes_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._startup_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._theme_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
        self._theme_menu_var = tk.StringVar(value='dgenerate')

        self._theme_menu_var.trace_add('write',
                                       lambda *a:
                                       self._set_theme(self._theme_menu_var.get()))

        self._theme_menu.add_radiobutton(label='none', value='none',
                                         variable=self._theme_menu_var)

        for theme_name in sorted(DgenerateCodeView.THEMES.keys()):
            self._theme_menu.add_radiobutton(label=theme_name, value=theme_name,
                                             variable=self._theme_menu_var)

        self._appearance_menu.add_cascade(label='Theme', menu=self._theme_menu)
        self._appearance_menu.add_command(
            label='Window Sash Size',
            command=self._open_sash_width_dialog)

        self._console_modes_menu.add_checkbutton(
            label='Multiline Input',
            accelerator=
            'Command+i' if platform.system() == 'Darwin' else 'Insert Key',
            variable=self._multi_line_input_check_var)

        # Create the word wrap input checkbox

        self._word_wrap_input_check_var = tk.BooleanVar(value=True)
        self._word_wrap_input_check_var.trace_add('write', self._update_input_wrap)

        self._console_modes_menu.add_checkbutton(label='Word Wrap Input',
                                                 variable=self._word_wrap_input_check_var)

        # Create the word wrap output checkbox

        self._word_wrap_output_check_var = tk.BooleanVar(value=True)
        self._word_wrap_output_check_var.trace_add('write', self._update_output_wrap)

        self._console_modes_menu.add_checkbutton(label='Word Wrap Output',
                                                 variable=self._word_wrap_output_check_var)

        # Create the auto scroll checkbox (scroll on input)
        self._auto_scroll_on_run_check_var = tk.BooleanVar(value=True)
        self._console_modes_menu.add_checkbutton(label='Scroll Output On Run',
                                                 variable=self._auto_scroll_on_run_check_var)

        # Create the auto scroll checkbox (scroll on output)
        self._auto_scroll_on_output_check_var = tk.BooleanVar(value=False)
        self._console_modes_menu.add_checkbutton(label='Scroll Output On New Output',
                                                 variable=self._auto_scroll_on_output_check_var)

        self._startup_menu.add_command(
            label='Edit init.dgen',
            command=self._open_init_dgen_editor)

        self._options_menu.add_cascade(label='Appearance', menu=self._appearance_menu)
        self._options_menu.add_cascade(label='Console Modes', menu=self._console_modes_menu)
        self._options_menu.add_cascade(label='Startup', menu=self._startup_menu)

        # View Menu

        self._view_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)

        # Help menu

        self._help_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)

        _resources.add_help_menu_links(self._help_menu, self)

        # Add sub menus to main menu

        self._menu_bar.add_cascade(label='File', menu=self._file_menu)
        self._menu_bar.add_cascade(label='Edit', menu=self._edit_menu)
        self._menu_bar.add_cascade(label='Run', menu=self._run_menu)
        if self._show_assistant:
            if platform.system() == 'Darwin':
                # The Mac menu bar only displays menus. A command added
                # directly to it is not shown.
                assistant_menu = _thememenu.ThemeMenu(self._menu_bar, tearoff=0)
                assistant_menu.add_command(
                    label='Assistant',
                    command=self._input_text_generate_with_assistant)
                self._menu_bar.add_cascade(label='Assistant', menu=assistant_menu)
            else:
                self._menu_bar.add_command(
                    label='Assistant',
                    command=self._input_text_generate_with_assistant)
        self._menu_bar.add_cascade(label='Options', menu=self._options_menu)
        self._menu_bar.add_cascade(label='View', menu=self._view_menu)
        self._menu_bar.add_cascade(label='Help', menu=self._help_menu)

        self._menu_bar.install()

        # Split window layout

        self._paned_window_horizontal = tk.PanedWindow(self,
                                                       orient=tk.HORIZONTAL,
                                                       sashwidth=self._sash_width,
                                                       bg='#808080', bd=0)

        self._paned_window_horizontal.pack(fill=tk.BOTH, expand=True)

        self._paned_window_vertical = tk.PanedWindow(self._paned_window_horizontal,
                                                     orient=tk.VERTICAL,
                                                     sashwidth=self._sash_width,
                                                     bg='#808080')

        self._paned_window_vertical.pack(fill=tk.BOTH, expand=True)
        self._paned_window_horizontal.add(self._paned_window_vertical)

        # Create the input text pane

        self._input_text = DgenerateCodeView(self._paned_window_vertical, undo=True)

        self._input_text.text.bind('<Return>',
                                   lambda e:
                                   None if self._multi_line_input_check_var.get()
                                   else self._run_input_text())
        self._input_text.text.bind('<Double-Button-1>', _util.select_whitespace_delimited_token)
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
        self._input_text_context = _thememenu.ThemeMenu(self._input_text, tearoff=0)
        self._create_edit_menu(self._input_text_context)
        self._input_text_context.add_separator()
        self._input_text_context.add_checkbutton(
            label='Multiline',
            variable=self._multi_line_input_check_var)
        self._input_text_context.add_checkbutton(
            label='Wrap Text',
            variable=self._word_wrap_input_check_var)

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

        # Parent the log to the console for wm_manage float/dock (see _float_output).
        # Tk on Windows rejects wm_manage on ttk.Frame; ScrolledText stays ttk inside
        # a classic Frame shell that only exists so the output can become its own window.
        self._output_pane = tk.Frame(self)
        self._output_text = ScrolledText(self._output_pane)
        self._output_text.pack(fill=tk.BOTH, expand=True)

        self._output_text.text.bind('<Double-Button-1>', _util.select_whitespace_delimited_token)
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

        self._output_text_context = _thememenu.ThemeMenu(self._output_text, tearoff=0)

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
        if _showindirectory.is_supported():
            self._output_text_context.add_command(
                label='Show File In Directory',
                command=lambda: self._show_selected_image_in_directory(self._output_text.text))
        self._output_text_context.add_separator()
        self._output_text_context.add_command(label='Make Window', command=self._toggle_output_float)
        self._output_float_menu_index = self._output_text_context.index('end')
        self._output_text_context.add_separator()
        self._output_text_context.add_checkbutton(
            label='Wrap Text',
            variable=self._word_wrap_output_check_var)

        # Patch the output text context menu to enable/disable Preview Selected Image
        og_output_popup = self._output_text_context.tk_popup

        def patch_output_tk_popup(*args, **kwargs):
            selected_text = self._get_selected_text(self._output_text.text)
            if selected_text and self._is_valid_image_path(selected_text):
                self._output_text_context.entryconfigure('Preview Selected Image', state=tk.NORMAL)
                if _showindirectory.is_supported():
                    self._output_text_context.entryconfigure('Show File In Directory', state=tk.NORMAL)
            else:
                self._output_text_context.entryconfigure('Preview Selected Image', state=tk.DISABLED)
                if _showindirectory.is_supported():
                    self._output_text_context.entryconfigure('Show File In Directory', state=tk.DISABLED)
            self._sync_output_window_labels()
            og_output_popup(*args, **kwargs)

        self._output_text_context.tk_popup = patch_output_tk_popup

        self._paned_window_vertical.add(self._output_pane)
        self._output_floated = tk.BooleanVar(value=False)
        self._output_is_floating = False
        self._output_float_geometry = '700x420'
        self._output_sash = None
        self._dock_output_callback = self.register(self._dock_output_from_close)

        # Initialize tearoff image pane
        self._tearoff_image_pane = TearOffImagePane(
            master=self._paned_window_horizontal,
            on_error=lambda s: self._write_console_message(s, error=True),
            on_info=lambda s: self._write_console_message(s),
            get_cwd=lambda: self._shell_procmon.cwd(),
            initial_window_geometry='512x512',
            window_title='Image Preview'
        )

        # Add menu items for the tearoff image pane
        self._view_menu.add_checkbutton(label='Image Pane',
                                        variable=self._tearoff_image_pane.get_pane_visible_var())
        self._view_menu.add_checkbutton(label='Image Window',
                                        variable=self._tearoff_image_pane.get_window_visible_var())
        self._view_menu.add_separator()
        self._view_menu.add_command(label='Console Output Window', command=self._toggle_output_float)
        self._output_view_menu_index = self._view_menu.index('end')

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
        self._update_input_wrap()
        self._update_output_wrap()

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
            "To see generated media, see the menu option: View -> Image Pane or Image Window.\n"
            "Right click the image pane or window to see more options.\n\n"
            "Enter --help or the alias \\help to print dgenerate's help text. All lines which\n"
            "are not directives (starting with \\ ) or top level templates (starting with { )\n"
            "are processed as arguments to dgenerate.\n\n"
            "See: \\directives_help or \\directives_help (directive) for help with config directives.\n"
            "See: \\functions_help or \\functions_help (function) for help with template functions.\n"
            "See: \\templates_help or \\templates_help (variable name) for help with template variables.\n"
            "============================================================\n\n")

        self._output_terminal = TerminalText(self._output_text.text)

        self._output_text.text.config(state=tk.DISABLED)

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
        _appearance.install(self)
        self.deiconify()

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
            self._write_console_message(str(e), error=True)

    def _set_theme(self, name):
        self._input_text.set_theme(name)

        bg = self._input_text.text.cget('bg')
        fg = self._input_text.text.cget('fg')
        sel_bg = self._input_text.text.cget('selectbackground')
        sel_fg = self._input_text.text.cget('selectforeground')

        window_bg = bg
        window_chrome = None
        if name in DgenerateCodeView.THEMES:
            editor = DgenerateCodeView.THEMES[name].get('editor', {})
            window_bg = editor.get('window_bg', bg)
            window_chrome = editor.get('window_chrome', None)

        _themetext.configure(
            bg=bg,
            fg=fg,
            selectbackground=sel_bg,
            selectforeground=sel_fg,
            borderwidth=self._input_text.text.cget('borderwidth'),
            highlightthickness=self._input_text.text.cget('highlightthickness'),
            relief=self._input_text.text.cget('relief'),
        )
        _themedialog.apply(self, bg, fg, sel_bg, sel_fg,
                           window_background=window_bg, window_chrome=window_chrome)
        _themechrome.apply(bg, fg, window_background=window_bg, window_chrome=window_chrome)
        _themescrollbar.apply(self, bg, fg)
        _thememenu.apply(self, window_bg, fg, sel_bg, sel_fg)

        error_color = fg

        if name in DgenerateCodeView.THEMES:
            theme = DgenerateCodeView.THEMES[name]
            error_color = theme['general']['error']

        self._output_text.text.tag_configure(
            'error', foreground=error_color)

        self.save_settings()

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

    def _replace_input_text(self, text):
        self._input_text.text.edit_separator()
        self._input_text.text.replace('1.0', tk.END, text)
        self._input_text.text.edit_separator()
        self._input_text.text.mark_set('insert', '1.0')
        self.multiline_mode(True)

    def _input_text_generate_with_assistant(self):
        _assistantform.request_config(
            master=self,
            populate=self._replace_input_text,
            dgenerate_exe=DGENERATE_EXE,
            get_cwd=lambda: self._shell_procmon.cwd(deep=True),
            get_offline=self._offline_mode_var.get,
            get_editor=lambda: self._input_text.text.get('1.0', 'end-1c'),
            release_shell_memory=lambda done: _assistantform.release_shell_device_cache(
                self._shell_procmon, self.after, done),
        )

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

    def _image_preview_load_image(self, image_path):
        self._tearoff_image_pane.load_image(image_path, fit=True)

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
        self._write_console_message(
            f'\nShell Process Terminated, Exit Code: {return_code}\n')

    def _shell_restarting_commence_message(self):
        self._write_console_message('Restarting Shell Process...\n')
        return self._last_known_working_dir

    def _shell_restarting_finish_message(self):
        self._update_cwd_title(self._last_known_working_dir)

        self._write_console_message('Shell Process Started.\n'
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

    def _open_transpile_dialog(self):
        dialog = self._transpile_dialog
        if dialog is not None and dialog.top.winfo_exists():
            dialog.top.lift()
            return
        self._transpile_dialog = _transpileform.TranspileDialog(
            self,
            lambda: self._input_text.text.get('1.0', 'end-1c'),
            DGENERATE_EXE,
        )

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

    def _sync_output_window_labels(self):
        floated = self._output_is_floating
        labels = (
            (self._output_text_context, getattr(self, '_output_float_menu_index', None),
             'Make Pane' if floated else 'Make Window'),
            (self._view_menu, getattr(self, '_output_view_menu_index', None),
             'Console Output Pane' if floated else 'Console Output Window'),
        )
        for menu, index, label in labels:
            if index is None:
                continue
            try:
                menu.entryconfigure(index, label=label)
            except tk.TclError:
                pass

    def _apply_output_float(self):
        if self._output_floated.get():
            self._float_output()
        else:
            self._dock_output()

    def _toggle_output_float(self):
        self._output_floated.set(not self._output_is_floating)
        self._apply_output_float()

    def _dock_output_from_close(self):
        self._output_floated.set(False)
        self._dock_output()

    _OUTPUT_WINDOW_TITLE = 'Console Output'

    def _float_output(self):
        if self._output_is_floating:
            return
        output = self._output_pane
        try:
            self._output_sash = self._paned_window_vertical.sash_coord(0)
        except tk.TclError:
            self._output_sash = None
        try:
            self._paned_window_vertical.remove(output)
            self.update_idletasks()
            self.wm_manage(output)
            output.tk.call('wm', 'title', output._w, self._OUTPUT_WINDOW_TITLE)
            output.tk.call('wm', 'minsize', output._w, 320, 160)
            output.tk.call('wm', 'geometry', output._w, self._output_float_geometry)
            output.tk.call('wm', 'protocol', output._w, 'WM_DELETE_WINDOW',
                           self._dock_output_callback)
            if platform.system() == 'Darwin':
                # Docking withdraws the frame so Aqua can forget it. The
                # next manage leaves that withdrawn state in place.
                output.tk.call('wm', 'deiconify', output._w)
                output.tk.call('raise', output._w)
            _themechrome.adopt(output)
        except tk.TclError:
            self._output_is_floating = False
            self._output_floated.set(False)
            try:
                self.wm_forget(output)
            except tk.TclError:
                pass
            try:
                self._paned_window_vertical.add(output)
            except tk.TclError:
                pass
            self._sync_output_window_labels()
            return
        self._output_is_floating = True
        self._sync_output_window_labels()
        if platform.system() == 'Darwin':
            self._settle_output_windows()
        self._output_text.text.focus_set()

    def _settle_output_windows(self):
        title = self._OUTPUT_WINDOW_TITLE
        output = self._output_pane
        if self._output_is_floating:
            _themechrome.claim_output_window(output, title)
        else:
            _themechrome.close_leftover_windows(title)
        self.update_idletasks()
        if self._output_is_floating:
            _themechrome.claim_output_window(output, title)
        else:
            _themechrome.close_leftover_windows(title)

        def again(title=title, output=output):
            try:
                if self._output_is_floating:
                    _themechrome.claim_output_window(output, title)
                else:
                    _themechrome.close_leftover_windows(title)
            except tk.TclError:
                pass

        self.after_idle(again)

    def _restore_output_sash(self):
        sash = self._output_sash
        if sash is None:
            return

        def restore_sash(sash=sash):
            try:
                self._paned_window_vertical.sash_place(0, sash[0], sash[1])
            except tk.TclError:
                pass

        self.after_idle(restore_sash)

    def _dock_output(self):
        if not self._output_is_floating:
            return
        output = self._output_pane
        try:
            self._output_float_geometry = output.tk.call('wm', 'geometry', output._w)
        except tk.TclError:
            pass
        if platform.system() == 'Darwin':
            # Forget while a menu is still attached, or while this frame is
            # the key window, leaves a geometry idle that crashes Aqua.
            for child in list(output.winfo_children()):
                if isinstance(child, tk.Toplevel):
                    try:
                        child.destroy()
                    except tk.TclError:
                        pass
            try:
                self.focus_set()
                output.tk.call('wm', 'withdraw', output._w)
                self.update_idletasks()
            except tk.TclError:
                pass
        # Drop the themed frame before forget, or its title bar is packed
        # back into the pane with the log.
        _themechrome.release(output)
        self.wm_forget(output)
        self._paned_window_vertical.add(output)
        self._output_is_floating = False
        if platform.system() == 'Darwin':
            # wm forget leaves the Aqua window on screen, and the next
            # manage creates another one beside it. Tk orders the old
            # window back in on the following update, so hide it again.
            self._settle_output_windows()
        self._sync_output_window_labels()
        self._restore_output_sash()

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
        self._output_terminal.reset()
        self._output_text.text.config(state=tk.DISABLED)

    def _check_text_for_latest_image(self, text):
        mentioned_path = _preview_path_from_output(text)
        if mentioned_path:
            if os.path.isabs(mentioned_path):
                self._image_preview_load_image(mentioned_path)
            else:
                path = os.path.join(
                    self._shell_procmon.cwd(deep=True), mentioned_path)
                self._image_preview_load_image(path)

    def _update_cwd_title(self, directory=None):
        if directory is None:
            directory = self._shell_procmon.cwd()
        self.title(f'Dgenerate Console: {directory}')
        self._last_known_working_dir = directory

    def _add_output_line(self, text, tag=None, stream=None):
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

        self._output_terminal.write(text, tag=tag, stream=stream)

        self._output_text.text.config(state=tk.DISABLED)

        if scroll:
            self._output_text.text.see(tk.END)

    def _write_stdout_output(self, text: bytes | str):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stdout.buffer.write(text)
        sys.stdout.flush()
        self._add_output_line(text.decode('utf-8'), stream='stdout')

    def _write_stderr_output(self, text: bytes | str):
        if isinstance(text, str):
            text = text.encode('utf-8')

        sys.stderr.buffer.write(text)
        sys.stderr.flush()
        self._add_output_line(text.decode('utf-8'), tag='error', stream='stderr')

    def _write_console_message(self, text: str, error: bool = False):
        # Separate stream from the shell process, so a message never lands on a live progress bar line.
        if not text.endswith('\n'):
            text += '\n'
        file = sys.stderr if error else sys.stdout
        file.buffer.write(text.encode('utf-8'))
        file.flush()
        self._add_output_line(text, tag='error' if error else None, stream='console')

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
        # Create ~/.dgenerate directory if it doesn't exist
        dgenerate_dir = pathlib.Path(pathlib.Path.home(), '.dgenerate')
        dgenerate_dir.mkdir(exist_ok=True)
        
        settings_path = dgenerate_dir / 'console_settings.json'
        old_settings_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_settings')
        
        config = None
        
        # Check if new settings file exists
        if settings_path.exists():
            with settings_path.open('r', encoding='utf-8') as file:
                config = json.load(file)
        # Check if old settings file exists and migrate it
        elif old_settings_path.exists():
            try:
                with old_settings_path.open('r', encoding='utf-8') as file:
                    config = json.load(file)
                # Save to new location
                with settings_path.open('w', encoding='utf-8') as file:
                    json.dump(config, file)
                # Remove old file after successful migration
                old_settings_path.unlink()
                print(f'Migrated console settings from {old_settings_path} to {settings_path}', file=sys.stderr)
            except Exception as e:
                print(f'Warning: Failed to migrate old console settings: {e}', file=sys.stderr)
                config = None
        
        if config:
            sash_width = config.get('sash_width', self._sash_width)
            if type(sash_width) is int and 1 <= sash_width <= 30:
                self._set_sash_width(sash_width)
            self._theme_menu_var.set(config.get('theme', 'dgenerate'))
            self._auto_scroll_on_run_check_var.set(config.get('auto_scroll_on_run', True))
            self._auto_scroll_on_output_check_var.set(config.get('auto_scroll_on_output', False))
            self._word_wrap_input_check_var.set(config.get('word_wrap_input', True))
            self._word_wrap_output_check_var.set(config.get('word_wrap_output', True))
        else:
            self._theme_menu_var.set('dgenerate')

    def save_settings(self):
        # Create ~/.dgenerate directory if it doesn't exist
        dgenerate_dir = pathlib.Path(pathlib.Path.home(), '.dgenerate')
        dgenerate_dir.mkdir(exist_ok=True)
        
        settings_path = dgenerate_dir / 'console_settings.json'
        try:
            with settings_path.open('r', encoding='utf-8') as existing:
                config = json.load(existing)
        except (OSError, json.JSONDecodeError):
            config = {}
        if not isinstance(config, dict):
            config = {}
        config.update({
            'theme': self._theme_menu_var.get(),
            'auto_scroll_on_run': self._auto_scroll_on_run_check_var.get(),
            'auto_scroll_on_output': self._auto_scroll_on_output_check_var.get(),
            'word_wrap_input': self._word_wrap_input_check_var.get(),
            'word_wrap_output': self._word_wrap_output_check_var.get(),
            'sash_width': self._sash_width
        })
        with settings_path.open('w', encoding='utf-8') as file:
            json.dump(config, file)

    def _set_sash_width(self, width):
        self._sash_width = width
        self._paned_window_horizontal.configure(sashwidth=width)
        self._paned_window_vertical.configure(sashwidth=width)

    def _open_init_dgen_editor(self):
        _initdgeneditor.open_init_dgen_editor(
            self,
            theme_name=self._theme_menu_var.get(),
            on_saved=lambda: self.kill_shell_process(restart=True))

    def _open_sash_width_dialog(self):
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            return

        dialog = tk.Toplevel(self)
        # Measuring the window paints it. Hide it until it is placed.
        dialog.withdraw()
        self._settings_dialog = dialog
        dialog.title('Window Sash Size')
        dialog.transient(self)
        dialog.resizable(False, False)
        _resources.set_window_icon(dialog)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frame, text='Window sash width (pixels):').grid(
            row=0, column=0, sticky='w', padx=(0, 8))
        sash_width_var = tk.StringVar(value=str(self._sash_width))
        sash_width_entry = ttk.Spinbox(
            frame, from_=1, to=30, textvariable=sash_width_var, width=5)
        sash_width_entry.grid(row=0, column=1, sticky='ew')

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky='e', pady=(12, 0))

        def close_dialog():
            dialog.grab_release()
            dialog.destroy()
            self._settings_dialog = None

        def apply_settings():
            try:
                width = int(sash_width_var.get())
            except ValueError:
                dialog.bell()
                sash_width_entry.focus_set()
                return

            if not 1 <= width <= 30:
                dialog.bell()
                sash_width_entry.focus_set()
                return

            self._set_sash_width(width)
            self.save_settings()
            close_dialog()

        ttk.Button(button_frame, text='Cancel', command=close_dialog).pack(
            side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_frame, text='OK', command=apply_settings).pack(
            side=tk.RIGHT)

        dialog.protocol('WM_DELETE_WINDOW', close_dialog)
        dialog.bind('<Return>', lambda event: apply_settings())
        dialog.bind('<Escape>', lambda event: close_dialog())
        _appearance.install(dialog)
        dialog.update_idletasks()
        dialog.geometry(
            f'+{self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2}'
            f'+{self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2}')
        dialog.deiconify()
        dialog.grab_set()
        sash_width_entry.focus_set()

    def _load_command_history(self):
        if self._max_command_history == 0:
            return

        # Create ~/.dgenerate directory if it doesn't exist
        dgenerate_dir = pathlib.Path(pathlib.Path.home(), '.dgenerate')
        dgenerate_dir.mkdir(exist_ok=True)
        
        history_path = dgenerate_dir / 'console_history.json'
        old_history_path = pathlib.Path(pathlib.Path.home(), '.dgenerate_console_history')
        
        # Check if new history file exists
        if history_path.exists():
            with history_path.open('r', encoding='utf-8') as file:
                self._command_history = json.load(file)[-self._max_command_history:]
                self._current_command_index = len(self._command_history)
        # Check if old history file exists and migrate it
        elif old_history_path.exists():
            try:
                with old_history_path.open('r', encoding='utf-8') as file:
                    self._command_history = json.load(file)[-self._max_command_history:]
                    self._current_command_index = len(self._command_history)
                # Save to new location
                with history_path.open('w', encoding='utf-8') as file:
                    json.dump(self._command_history, file)
                # Remove old file after successful migration
                old_history_path.unlink()
                print(f'Migrated console history from {old_history_path} to {history_path}', file=sys.stderr)
            except Exception as e:
                print(f'Warning: Failed to migrate old console history: {e}', file=sys.stderr)
                self._command_history = []
                self._current_command_index = 0

    def _save_command_history(self):
        if self._max_command_history == 0:
            return

        # Create ~/.dgenerate directory if it doesn't exist
        dgenerate_dir = pathlib.Path(pathlib.Path.home(), '.dgenerate')
        dgenerate_dir.mkdir(exist_ok=True)
        
        history_path = dgenerate_dir / 'console_history.json'
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
            self._write_console_message(
                'WARNING: The command queue is full, '
                'please wait before submitting more commands, '
                'or kill the interpreter.',
                error=True)

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

        # Clean up tearoff image pane
        if self._output_is_floating:
            try:
                self._dock_output()
            except tk.TclError:
                pass
        self._tearoff_image_pane.cleanup()

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
        search_submenu = _thememenu.ThemeMenu(menu, tearoff=0)
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
        code_submenu = _thememenu.ThemeMenu(menu, tearoff=0)
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
        uri_submenu = _thememenu.ThemeMenu(menu, tearoff=0)
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
        path_submenu = _thememenu.ThemeMenu(menu, tearoff=0)
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
        if _showindirectory.is_supported():
            menu.add_command(
                label='Show File In Directory',
                command=lambda: self._show_selected_image_in_directory(self._input_text.text))

        def menu_post(*args, **kwargs):
            selected_text = self._get_selected_text(self._input_text.text)
            if selected_text and self._is_valid_image_path(selected_text):
                menu.entryconfigure(
                    'Preview Selected Image', state=tk.NORMAL)
                if _showindirectory.is_supported():
                    menu.entryconfigure('Show File In Directory', state=tk.NORMAL)
            else:
                menu.entryconfigure(
                    'Preview Selected Image', state=tk.DISABLED)
                if _showindirectory.is_supported():
                    menu.entryconfigure('Show File In Directory', state=tk.DISABLED)

        menu.configure(postcommand=menu_post)

        return menu

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
            supported_extensions = set(schema.get('images-in', []))
            supported_extensions.update(schema.get('videos-in', []))
            supported_extensions.update({
                'gif', 'webp', 'apng', 'mp4', 'm4v', 'webm', 'mkv', 'mov', 'avi'})
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

        if not self._tearoff_image_pane.is_pane_visible() and not self._tearoff_image_pane.is_window_visible():
            self._tearoff_image_pane.set_pane_visible(True)

        try:
            self._image_preview_load_image(path)
            self._write_console_message(f"Manually loaded image: {path}\n")
        except Exception as e:
            self._write_console_message(f"Failed to load image: {e}\n", error=True)

    def _show_selected_image_in_directory(self, text_widget):
        try:
            selected_text = text_widget.selection_get()
        except tk.TclError:
            return

        if not self._is_valid_image_path(selected_text):
            return

        path = self._shell_unquote_path(selected_text)
        if not os.path.isabs(path):
            path = os.path.join(self._shell_procmon.cwd(deep=True), path)

        _showindirectory.show_in_directory(path)

    @staticmethod
    def _get_selected_text(text_widget: tk.Text):
        try:
            return text_widget.selection_get()
        except tk.TclError:
            return None


# AppKit prints this to stderr when a window-move ends without a start.
# Tk already drops the event when it can. The log that still gets out is
# written by AppKit itself, so it never reaches the console pane.
_APPKIT_MOVE_WARNING = b'Window move completed without beginning'


def _consume_stderr(buf: bytes, marker: bytes = _APPKIT_MOVE_WARNING) -> tuple[bytes, bytes]:
    """Return bytes to forward, and the tail that might still be the warning."""
    out = []
    while True:
        found = buf.find(marker)
        if found == -1:
            keep = len(marker) - 1
            if len(buf) > keep:
                out.append(buf[:-keep])
                buf = buf[-keep:]
            return b''.join(out), buf
        line_start = buf.rfind(b'\n', 0, found) + 1
        line_end = buf.find(b'\n', found)
        if line_end == -1:
            if line_start:
                out.append(buf[:line_start])
                buf = buf[line_start:]
            return b''.join(out), buf
        if line_start:
            out.append(buf[:line_start])
        buf = buf[line_end + 1:]


def _flush_held_stderr(buf: bytes, marker: bytes = _APPKIT_MOVE_WARNING) -> bytes:
    """Release a held tail once no more bytes are arriving."""
    if marker not in buf:
        return buf
    return buf[:buf.rfind(b'\n', 0, buf.find(marker)) + 1]


def _silence_appkit_window_move_warning() -> None:
    """Drop AppKit's window-move warning from this process's stderr."""
    if sys.platform != 'darwin' or getattr(_silence_appkit_window_move_warning, 'installed', False):
        return
    try:
        saved = os.dup(2)
        read_fd, write_fd = os.pipe()
    except OSError:
        return
    os.set_inheritable(saved, False)
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    os.set_blocking(read_fd, False)

    def write_all(data: bytes) -> None:
        while data:
            wrote = os.write(saved, data)
            data = data[wrote:]

    def forward():
        buf = b''
        try:
            while True:
                ready, _, _ = select.select([read_fd], [], [], 0.05)
                if not ready:
                    pending = _flush_held_stderr(buf)
                    if pending:
                        write_all(pending)
                    buf = b''
                    continue
                try:
                    chunk = os.read(read_fd, 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    pending = _flush_held_stderr(buf)
                    if pending:
                        write_all(pending)
                    return
                buf += chunk
                out, buf = _consume_stderr(buf)
                if out:
                    write_all(out)
        except OSError:
            return

    threading.Thread(target=forward, name='stderr-filter', daemon=True).start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    _silence_appkit_window_move_warning.installed = True


def main(args: collections.abc.Sequence[str]):
    _silence_appkit_window_move_warning()
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
