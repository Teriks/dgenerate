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
import tkinter.font as tkfont

import tklinenums

import dgenerate.console.resources as _resources
import dgenerate.console.syntaxhighlighter as _syntaxhighlighter
import dgenerate.console.fonts as _fonts
import dgenerate.pygments
import dgenerate.textprocessing


class DgenerateCodeView(tk.Frame):
    THEMES = dict(_resources.get_themes())

    def __init__(self, master=None, undo=False, **kwargs):
        super().__init__(master, **kwargs)

        text_args = {}
        if undo:
            text_args['undo'] = True
            text_args['autoseparators'] = True
            text_args['maxundo'] = -1

        self.text = tk.Text(self, **text_args)
        self._highlighting = True

        self.indentation = " " * 4
        
        # Column selection state
        self._column_mode = False
        self._column_anchor = None
        self._column_start = None
        self._column_end = None
        self._column_select_tag = "column_selection"
        self._native_selection_disabled = False

        self._line_numbers = tklinenums.TkLineNumbers(
            self,
            self.text,
            borderwidth=0,
        )

        self.y_scrollbar = tk.Scrollbar(self, orient='vertical', command=self.text.yview)
        self.x_scrollbar = tk.Scrollbar(self, orient='horizontal', command=self.text.xview)

        self.y_scrollbar.grid(row=0, column=2, sticky="ns")
        self.x_scrollbar.grid(row=1, column=1, sticky="we")

        self.text.grid(row=0, column=1, sticky='nsew')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._syntax_highlighter = _syntaxhighlighter.TkTextHighlighter(
            self.text, dgenerate.pygments.DgenerateLexer())

        # Use cross-platform font selection if no font is specified
        if 'font' not in kwargs:
            default_font_family, default_font_size = _fonts.get_default_monospace_font(10)
            kwargs['font'] = (default_font_family, default_font_size)

        self.text.configure(
            yscrollcommand=self._vertical_scroll,
            xscrollcommand=self._horizontal_scroll,
            font=kwargs['font'],
            tabs=tkfont.Font(font=kwargs["font"]).measure(" " * 4),
        )

        def key_release(event):
            keysym = event.keysym.lower() if event.keysym else ''
            char = event.char if event.char else ''

            if keysym or char:
                # cannot be picky here for proper
                # cross platform behavior, any key triggers
                if self._highlighting:
                    self._syntax_highlighter.highlight_visible()

            self.text.focus_set()

        self.text.bind('<<Modified>>',
                       lambda e: (self._line_numbers.redraw(),
                                  self.text.edit_modified(False)))
        self.text.bind('<KeyRelease>', key_release)
        self.text.bind('<Tab>', self._indent)
        self.text.bind('<Shift-Tab>', self._unindent)
        
        # Column selection bindings
        self.text.bind('<Alt-ButtonPress-1>', self._start_column_select)
        self.text.bind('<Alt-B1-Motion>', self._update_column_select)
        self.text.bind('<Alt-ButtonRelease-1>', self._end_column_select)
        
        # Column editing key bindings
        self.text.bind('<Alt-Left>', lambda e: self._column_move('left'))
        self.text.bind('<Alt-Right>', lambda e: self._column_move('right'))
        self.text.bind('<Alt-Up>', lambda e: self._column_move('up'))
        self.text.bind('<Alt-Down>', lambda e: self._column_move('down'))

        # These need to be completely disabled
        # to avoid weird focus bugs
        self.text.bind('<Alt_L>', lambda e: "break")
        self.text.bind('<Alt_R>', lambda e: "break")

        # Input handling in column mode
        self.text.bind('<KeyPress>', self._column_key_press)
        
        # Comment selected text with # key
        self.text.bind('<KeyPress-numbersign>', self._comment_selected_text)
        
        # Keep normal bindings but exit column mode if needed
        self.text.bind('<ButtonPress-1>', self._exit_column_mode)

        # Handle undo/redo/paste events
        self.text.bind('<<Undo>>', lambda e: self._handle_undo(e))
        self.text.bind('<<Redo>>', lambda e: self._handle_redo(e))

        replace = self.text.replace

        def replace_new(*a, **kwargs):
            replace(*a, **kwargs)
            if self._highlighting:
                self._syntax_highlighter.highlight_visible()

        self.text.replace = replace_new

        insert = self.text.insert

        def insert_new(*a, **kwargs):
            insert(*a, **kwargs)
            if self._highlighting:
                self._syntax_highlighter.highlight_visible()

        self.text.insert = insert_new

        delete = self.text.delete

        def delete_new(*a, **kwargs):
            delete(*a, **kwargs)
            if self._highlighting:
                self._syntax_highlighter.highlight_visible()

        self.text.delete = delete_new
        
    def _start_column_select(self, event):
        """Begin column selection mode."""
        self._exit_column_mode(None)  # Clear any existing column selection
        self._column_mode = True
        
        # Update column selection colors to match current selection colors
        select_bg = self.text.cget('selectbackground')
        select_fg = self.text.cget('selectforeground')
        self.text.tag_configure(self._column_select_tag, background=select_bg, foreground=select_fg)
        
        # Get the index at the current mouse position
        index = self.text.index(f"@{event.x},{event.y}")
        self._column_anchor = index
        self._column_start = index
        self._column_end = index

        # Set cursor at the start of the selection
        self.text.mark_set("insert", self._column_start)
        
        # Explicitly set focus to ensure keypresses are captured
        self.text.focus_set()
        
        # Disable native text selection
        self._disable_native_selection()

        # Prevent default handling
        return "break"
        
    def _update_column_select(self, event):
        """Update the column selection as the mouse moves."""
        if not self._column_mode:
            return
            
        # Get the new index at mouse position
        current_index = self.text.index(f"@{event.x},{event.y}")
        self._column_end = current_index
        
        # Update the visual selection
        self._update_column_highlighting()

        # Always keep cursor at the start of selection
        self.text.mark_set("insert", self._column_start)

        # Prevent default handling
        return "break"
        
    def _end_column_select(self, event):
        """Finalize the column selection."""
        if not self._column_mode:
            return
            
        # Update the selection one last time
        current_index = self.text.index(f"@{event.x},{event.y}")
        self._column_end = current_index
        self._update_column_highlighting()

        # Always keep cursor at the start of selection
        self.text.mark_set("insert", self._column_start)
        
        # Explicitly set focus to the text widget to ensure keypresses are captured
        self.text.focus_set()

        # Prevent default handling
        return "break"
        
    def _disable_native_selection(self):
        """Disable the native text selection temporarily."""
        if not self._native_selection_disabled:
            # Store the original selection background
            self._original_select_bg = self.text.cget('selectbackground')
            self._original_select_fg = self.text.cget('selectforeground')
            
            # Change selection color to transparent to hide native selection
            self.text.config(selectbackground=self.text.cget('background'), 
                             selectforeground=self.text.cget('foreground'))
            self._native_selection_disabled = True
            
    def _restore_native_selection(self):
        """Restore the native text selection."""
        if self._native_selection_disabled:
            # Restore the original selection colors
            self.text.config(selectbackground=self._original_select_bg,
                             selectforeground=self._original_select_fg)
            self._native_selection_disabled = False
            
    def _update_column_highlighting(self):
        """Update visual highlighting for column selection."""
        if not self._column_mode:
            return
            
        # Clear existing column selection highlights
        self.text.tag_remove(self._column_select_tag, "1.0", "end")
        
        # Get the column boundaries
        start_line, start_col = map(int, self._column_start.split('.'))
        end_line, end_col = map(int, self._column_end.split('.'))
        
        # Ensure start is before end in both dimensions
        if start_line > end_line:
            start_line, end_line = end_line, start_line
            # Swap the actual start/end points so _column_start always points to the beginning
            self._column_start, self._column_end = self._column_end, self._column_start
        if start_col > end_col:
            start_col, end_col = end_col, start_col
            # If only columns were swapped, update _column_start and _column_end
            if int(self._column_start.split('.')[0]) == start_line:
                self._column_start = f"{start_line}.{start_col}"
            if int(self._column_end.split('.')[0]) == end_line:
                self._column_end = f"{end_line}.{end_col}"
            
        # Apply highlighting to each line in the selection
        for line in range(start_line, end_line + 1):
            line_length = int(self.text.index(f"{line}.end").split('.')[1])
            
            # Skip lines that are too short for the selection
            if line_length < start_col:
                continue
                
            # Apply tag to the column range on this line
            sel_end_col = min(end_col, line_length)
            self.text.tag_add(self._column_select_tag, 
                            f"{line}.{start_col}", 
                            f"{line}.{sel_end_col}")

        # Keep cursor at the start
        self.text.mark_set("insert", self._column_start)

    def _exit_column_mode(self, event):
        """Exit column selection mode."""
        if self._column_mode:
            self.text.tag_remove(self._column_select_tag, "1.0", "end")
            self._column_mode = False
            self._column_anchor = None
            self._column_start = None
            self._column_end = None
            
            # Restore native selection
            self._restore_native_selection()
        
    def _column_move(self, direction):
        """Handle arrow key movement in column selection mode."""
        if not self._column_mode:
            # Start column mode if not already in it
            self._column_mode = True
            
            # Update column selection colors to match current selection colors
            select_bg = self.text.cget('selectbackground')
            select_fg = self.text.cget('selectforeground')
            self.text.tag_configure(self._column_select_tag, background=select_bg, foreground=select_fg)
            
            self._column_anchor = self.text.index("insert")
            self._column_start = self._column_anchor
            self._column_end = self._column_anchor
            
            # Disable native text selection
            self._disable_native_selection()
            
            # Ensure text widget has focus
            self.text.focus_set()
        
        # Calculate new position based on direction
        current_line, current_col = map(int, self._column_end.split('.'))
        
        if direction == 'left':
            if current_col > 0:
                current_col -= 1
        elif direction == 'right':
            current_col += 1
        elif direction == 'up':
            if current_line > 1:
                current_line -= 1
        elif direction == 'down':
            current_line += 1
            
        # Update the end position
        self._column_end = f"{current_line}.{current_col}"
        
        # Update the visual selection
        self._update_column_highlighting()

        # Keep cursor at the start of selection
        self.text.mark_set("insert", self._column_start)
        
        # Ensure the cursor is visible
        self.text.see(self._column_start)
        
        # Prevent default handling
        return "break"
        
    def _column_key_press(self, event):
        """Handle typing in column selection mode."""

        if not self._column_mode:
            return None
            
        # Only handle regular character input or backspace/delete
        if event.keysym in ('Alt_L', 'Alt_R', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R',
                         'Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Page_Up', 'Page_Down'):
            return None
            
        # Get the column boundaries
        start_line, start_col = map(int, self._column_start.split('.'))
        end_line, end_col = map(int, self._column_end.split('.'))
        
        # Ensure start is before end
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if start_col > end_col:
            start_col, end_col = end_col, start_col
            
        # Handle different key actions
        if event.keysym == "BackSpace":
            # Delete to the left in each line
            if start_col > 0:  # Can only backspace if not at start of line
                for line in range(start_line, end_line + 1):
                    line_length = int(self.text.index(f"{line}.end").split('.')[1])
                    if line_length >= start_col:  # Only if the line is long enough
                        self.text.delete(f"{line}.{start_col-1}", f"{line}.{start_col}")
                
                # Update selection bounds
                self._column_start = f"{start_line}.{start_col-1}"
                self._column_end = f"{end_line}.{end_col-1}"
                self._update_column_highlighting()
                
            return "break"
            
        elif event.keysym == "Delete":
            # Delete to the right in each line
            for line in range(start_line, end_line + 1):
                line_length = int(self.text.index(f"{line}.end").split('.')[1])
                if line_length >= start_col:  # Only if the line is long enough
                    if end_col < line_length:  # Only if we're not at the end
                        self.text.delete(f"{line}.{start_col}", f"{line}.{start_col+1}")
            
            # Selection bounds don't change for delete
            self._update_column_highlighting()
            return "break"
            
        elif event.char and ord(event.char) >= 32:  # Printable character
            # Use the helper method to insert the character
            self._handle_column_character(event.char)
            return "break"
            
        return None

    def _handle_column_character(self, char):
        """Insert a character at the column position in all selected lines."""
        if not self._column_mode:
            return
            
        # Get the column boundaries
        start_line, start_col = map(int, self._column_start.split('.'))
        end_line, end_col = map(int, self._column_end.split('.'))
        
        # Ensure start is before end
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if start_col > end_col:
            start_col, end_col = end_col, start_col
            
        # Insert the character at the same column position in each line
        for line in range(start_line, end_line + 1):
            line_length = int(self.text.index(f"{line}.end").split('.')[1])
            
            # If line is too short, pad with spaces
            if line_length < start_col:
                padding = ' ' * (start_col - line_length)
                self.text.insert(f"{line}.{line_length}", padding)
                
            # Insert the character
            self.text.insert(f"{line}.{start_col}", char)
            
        # Update selection bounds
        self._column_start = f"{start_line}.{start_col+1}"
        self._column_end = f"{end_line}.{end_col+1}"
        self._update_column_highlighting()
        
    def _handle_undo(self, event):
        self._exit_column_mode(None)
        
    def _handle_redo(self, event):
        self._exit_column_mode(None)

    def gen_undo_event(self):
        self.text.event_generate('<<Undo>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_redo_event(self):
        self.text.event_generate('<<Redo>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_cut_event(self):
        if self._column_mode:
            self._column_cut()
            return
        self.text.event_generate('<<Cut>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()
            
    def _column_cut(self):
        """Cut the selected column text to clipboard."""
        column_text = self._get_column_text()
        if column_text:
            self.clipboard_clear()
            self.clipboard_append("\n".join(column_text))
            self._column_delete()
            
    def _column_delete(self):
        """Delete the currently selected column text."""
        if not self._column_mode:
            return
            
        # Get the column boundaries
        start_line, start_col = map(int, self._column_start.split('.'))
        end_line, end_col = map(int, self._column_end.split('.'))
        
        # Ensure start is before end
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if start_col > end_col:
            start_col, end_col = end_col, start_col
        
        # Delete selected text in each line
        for line in range(start_line, end_line + 1):
            line_length = int(self.text.index(f"{line}.end").split('.')[1])
            
            # Skip lines that are too short
            if line_length < start_col:
                continue
                
            # Delete the column range on this line
            sel_end_col = min(end_col, line_length)
            self.text.delete(f"{line}.{start_col}", f"{line}.{sel_end_col}")
            
        # Update selection after deletion
        self._column_end = f"{start_line}.{start_col}"
        self._update_column_highlighting()
        
    def _get_column_text(self):
        """Get the text from the current column selection as a list of strings."""
        if not self._column_mode:
            return []
            
        # Get the column boundaries
        start_line, start_col = map(int, self._column_start.split('.'))
        end_line, end_col = map(int, self._column_end.split('.'))
        
        # Ensure start is before end
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if start_col > end_col:
            start_col, end_col = end_col, start_col
            
        # Collect text from each line
        result = []
        for line in range(start_line, end_line + 1):
            line_length = int(self.text.index(f"{line}.end").split('.')[1])
            
            # Skip lines that are too short
            if line_length < start_col:
                result.append("")
                continue
                
            # Get text from this line's column range
            sel_end_col = min(end_col, line_length)
            text = self.text.get(f"{line}.{start_col}", f"{line}.{sel_end_col}")
            result.append(text)
            
        return result

    def gen_paste_event(self):
        self.text.event_generate('<<Paste>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_delete_event(self):
        if self._column_mode:
            self._column_delete()
            return
        self.text.event_generate('<Delete>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_selectall_event(self):
        self._exit_column_mode(None)  # Exit column mode
        self.text.event_generate('<<SelectAll>>')

    def gen_copy_event(self):
        if self._column_mode:
            column_text = self._get_column_text()
            if column_text:
                self.clipboard_clear()
                self.clipboard_append("\n".join(column_text))
            return
        self.text.event_generate('<<Copy>>')

    def format_code(self):
        """Format the code in the text widget."""
        self._exit_column_mode(None)  # Exit column mode before formatting
        text = self.text.get("1.0", "end-1c")
        self.text.replace("1.0", "end-1c",
                          '\n'.join(dgenerate.textprocessing.format_dgenerate_config(
                              iter(text.splitlines()),
                              indentation=self.indentation)))

    def _indent(self, event):
        """Indent the selected text or the current line."""
        # Exit column mode if active
        if self._column_mode:
            self._exit_column_mode(None)
            
        # Get the current selection
        sel = self.text.tag_ranges('sel')
        if sel:
            # If there is a selection, get the start and end lines
            start_line = self.text.index("%s linestart" % sel[0])
            end_line = self.text.index("%s lineend" % sel[1])

            # Iterate over each line in the selection
            for line in range(int(start_line.split('.')[0]), int(end_line.split('.')[0]) + 1):
                # Insert an indentation at the start of the line
                self.text.insert(f"{line}.0", self.indentation)
        else:
            # If there is no selection, insert an indentation at the cursor position
            self.text.insert('insert', self.indentation)

        # Prevent the default Tab behavior
        return 'break'

    def _unindent(self, event):
        """Unindent the selected text or the current line."""
        # Exit column mode if active
        if self._column_mode:
            self._exit_column_mode(None)
            
        # Get the current selection
        sel = self.text.tag_ranges('sel')
        if sel:
            # If there is a selection, get the start and end lines
            start_line = self.text.index("%s linestart" % sel[0])
            end_line = self.text.index("%s lineend" % sel[1])

            # Iterate over each line in the selection
            for line in range(int(start_line.split('.')[0]), int(end_line.split('.')[0]) + 1):
                # Get the start of the line
                start_of_line = f"{line}.0"

                # Get the indentation at the start of the line
                indentation = self.text.get(start_of_line, f"{line}.{len(self.indentation)}")

                # If it matches the indentation style, delete it
                if indentation == self.indentation:
                    self.text.delete(start_of_line, f"{line}.{len(self.indentation)}")
                # If the first character is a tab, delete it
                elif self.text.get(start_of_line, f"{line}.1") == '\t':
                    self.text.delete(start_of_line, f"{line}.1")
                else:
                    # remove all indentation
                    self.text.delete(
                        start_of_line,
                        f"{line}.{len(indentation) - len(indentation.lstrip(' '))}"
                    )

        # Prevent the default Shift-Tab behavior
        return 'break'

    def _horizontal_scroll(self, first: str | float, last: str | float):
        self.x_scrollbar.set(first, last)

    def _vertical_scroll(self, first: str | float, last: str | float):
        self.y_scrollbar.set(first, last)
        self._line_numbers.redraw()
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def set_theme(self, color_theme: dict[str, dict[str, str | int]] | str | None) -> None:
        """Set the color theme for syntax highlighting."""
        if color_theme is None or color_theme.lower() == 'none':
            self._highlighting = False
            self._syntax_highlighter.remove_tags()
            self.text.configure(fg='black', bg='white')
        else:
            self._highlighting = True
            if isinstance(color_theme, str):
                color_theme = DgenerateCodeView.THEMES[color_theme]

            self._syntax_highlighter.set_theme(color_theme)

    def enable_line_numbers(self):
        self._line_numbers.grid(row=0, column=0, sticky='ns')

    def disable_line_numbers(self):
        self._line_numbers.grid_forget()

    def enable_word_wrap(self):
        self.text.config(wrap='word')

    def disable_word_wrap(self):
        self.text.config(wrap='none')

    def _comment_selected_text(self, event):
        """Comment out selected text by adding # to the beginning of each line."""
        # Only handle when we're not in column mode and have a selection
        if self._column_mode:
            # In column mode, let the column key press handler handle it
            # We need to manually insert the # character in column mode
            self._handle_column_character('#')
            return "break"
            
        # Check if there is selected text
        try:
            selected_text = self.text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            # No selection, let normal typing happen
            return None
            
        # Get the line ranges of the selection
        sel_start = self.text.index(tk.SEL_FIRST)
        sel_end = self.text.index(tk.SEL_LAST)
        
        start_line = int(sel_start.split('.')[0])
        end_line = int(sel_end.split('.')[0])
        
        # Add # to the beginning of each line
        for line_num in range(start_line, end_line + 1):
            line_start = f"{line_num}.0"
            
            # If this is the first line and the selection doesn't start at the beginning
            # we need to check if we should add a comment to this line
            if line_num == start_line:
                # Get the selection start column
                start_col = int(sel_start.split('.')[1])
                # Only comment this line if selection starts at beginning or line is fully selected
                if start_col > 0:
                    # Check if the entire line content is selected
                    line_content = self.text.get(line_start, f"{line_num}.end")
                    line_length = len(line_content)
                    
                    # If the selection doesn't cover most of the line, skip commenting this line
                    if start_col > 2 and line_length - start_col > 2:
                        continue
            
            # Insert the # character at the beginning of the line
            self.text.insert(line_start, "# ")
        
        # Prevent the default behavior (which would insert another #)
        return "break"

    def clear(self):
        """Clear all text content and reset the widget state."""
        # Exit column mode if active
        self._exit_column_mode(None)
        
        # Clear all text content
        self.text.delete("1.0", "end")
        
        # Clear any selection
        self.text.tag_remove("sel", "1.0", "end")

        # Clear syntax highlighting tags if highlighting is enabled
        if self._highlighting:
            self._syntax_highlighter.remove_tags()
        
        # Line numbers will automatically update due to the <<Modified>> event
        # Force a redraw to ensure they're updated immediately
        self._line_numbers.redraw()