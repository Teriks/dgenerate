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

import tkinter as tk

import dgenerate.console.finddialog as _finddialog
from dgenerate.console.mousewheelbind import bind_mousewheel, un_bind_mousewheel
import dgenerate.console.themetext as _themetext

class HelpDialog:

    def __init__(self,
                 parent,
                 title: str,
                 help_text: str,
                 size: tuple = (850,600),
                 dock_to_right: bool = True,
                 position_widget=None
                 ):
        """
        Create and show a help dialog.
        
        :param parent: Parent widget for the dialog hierarchy
        :param title: Title for the dialog window
        :param help_text: Text content to display in the dialog
        :param size: Window size.
        :param dock_to_right: If True, dock the dialog to the right of the position_widget
        :param position_widget: Widget to use for positioning calculations (defaults to parent)
        """
        self.parent = parent
        self.position_widget = position_widget if position_widget is not None else parent
        self.title = title
        self.help_text = help_text
        self.dock_to_right = dock_to_right
        self.size = size

        self.top = None
        self.text_widget = None
        self.context_menu = None

        self._create_dialog()

    def _create_dialog(self):
        """Create the help dialog window."""
        self.top = tk.Toplevel(self.parent)
        self.top.title(self.title)
        self.top.attributes('-topmost', 1)
        self.top.attributes('-topmost', 0)

        frame = tk.Frame(self.top)
        frame.pack(expand=True, fill='both')

        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.text_widget = _themetext.ThemeText(
            frame,
            wrap=tk.NONE,
            state='disabled',
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )

        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, self.help_text + '\n\n')
        self.text_widget.config(state='disabled')

        self.text_widget.bind(
            '<Control-f>',
            lambda e: _finddialog.open_find_dialog(
                self.top,
                f'Find In {self.title}',
                self.text_widget))

        v_scrollbar.config(command=self.text_widget.yview)
        h_scrollbar.config(command=self.text_widget.xview)
        self.text_widget.pack(expand=True, fill='both')

        self._create_context_menu()
        self._position_dialog()
        self._bind_events()

    def _create_context_menu(self):
        """Create the right-click context menu for the text widget."""
        self.context_menu = tk.Menu(self.text_widget, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self._copy_text)
        self.context_menu.add_command(label="Copy All", command=self._copy_all_text)
        self.context_menu.add_command(
            label='Find',
            accelerator='Ctrl+F',
            command=lambda:
            _finddialog.open_find_dialog(
                self.top,
                f'Find In {self.title}',
                self.text_widget))

        # Bind right-click to show context menu
        self.text_widget.bind("<Button-3>", self._show_context_menu)

    def _show_context_menu(self, event):
        """Show the context menu at the cursor position."""
        try:
            # Check if there's selected text to enable/disable copy
            if self.text_widget.tag_ranges(tk.SEL):
                self.context_menu.entryconfig("Copy", state="normal")
            else:
                self.context_menu.entryconfig("Copy", state="disabled")

            self.context_menu.tk_popup(event.x_root, event.y_root)
        except tk.TclError:
            pass
        finally:
            self.context_menu.grab_release()

    def _copy_text(self):
        """Copy selected text to clipboard."""
        try:
            if self.text_widget.tag_ranges(tk.SEL):
                selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.text_widget.clipboard_clear()
                self.text_widget.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def _copy_all_text(self):
        """Copy all text to clipboard."""
        try:
            all_text = self.text_widget.get("1.0", tk.END)
            self.text_widget.clipboard_clear()
            self.text_widget.clipboard_append(all_text.rstrip('\n'))
        except tk.TclError:
            pass

    def _position_dialog(self):
        """Position the dialog window."""
        width, height = self.size

        if self.dock_to_right:
            # Dock to the right of the position widget
            pos_x = self.position_widget.winfo_rootx()
            pos_y = self.position_widget.winfo_rooty()
            pos_width = self.position_widget.winfo_width()

            x = pos_x + pos_width
            y = pos_y
        else:
            # Center on the position widget
            pos_x = self.position_widget.winfo_rootx()
            pos_y = self.position_widget.winfo_rooty()
            pos_width = self.position_widget.winfo_width()
            pos_height = self.position_widget.winfo_height()

            x = pos_x + (pos_width // 2) - (width // 2)
            y = pos_y + (pos_height // 2) - (height // 2)

        self.top.geometry(f'{width}x{height}+{x}+{y}')

    def _bind_events(self):
        """Bind mouse wheel and cleanup events."""
        bind_mousewheel(self.top.bind, self._on_help_mouse_wheel)

        og_destroy = self.top.destroy

        def new_destroy():
            un_bind_mousewheel(self.top.bind)
            og_destroy()

        self.top.destroy = new_destroy

    @staticmethod
    def _on_help_mouse_wheel(event):
        """Handle mouse wheel events."""
        return "break"


def show_help_dialog(parent,
                     title: str,
                     help_text: str,
                     size: tuple = (850,600),
                     dock_to_right: bool = True,
                     position_widget=None) -> HelpDialog:
    """
    Convenience function to create and show a help dialog.
    
    :param parent: Parent widget for the dialog hierarchy
    :param title: Title for the dialog window
    :param help_text: Text content to display in the dialog
    :param size: Window size.
    :param dock_to_right: If ``True``, dock the dialog to the right of the position_widget
    :param position_widget: Widget to use for positioning calculations (defaults to parent)
    :return: HelpDialog instance
    """
    return HelpDialog(parent, title, help_text, size, dock_to_right, position_widget)
