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

import tklinenums

import dgenerate.console.resources as _resources
import dgenerate.console.syntaxhighlighter as _syntaxhighlighter
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

        kwargs.setdefault('font', ('Courier', 10))

        self.text.configure(
            yscrollcommand=self._vertical_scroll,
            xscrollcommand=self._horizontal_scroll,
            font=kwargs['font'],
            tabs=tk.font.Font(font=kwargs["font"]).measure(" " * 4),
        )

        def key_release(event):
            keysym = event.keysym.lower() if event.keysym else ''
            char = event.char if event.char else ''

            if keysym or char:
                # cannot be picky here for proper
                # cross platform behavior, any key triggers
                if self._highlighting:
                    self._syntax_highlighter.highlight_visible()

        self.text.bind('<<Modified>>',
                       lambda e: (self._line_numbers.redraw(),
                                  self.text.edit_modified(False)))
        self.text.bind('<KeyRelease>', key_release)
        self.text.bind('<Tab>', self._indent)
        self.text.bind('<Shift-Tab>', self._unindent)

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

    def gen_undo_event(self):
        self.text.event_generate('<<Undo>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_redo_event(self):
        self.text.event_generate('<<Redo>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_cut_event(self):
        self.text.event_generate('<<Cut>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_paste_event(self):
        self.text.event_generate('<<Paste>>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_delete_event(self):
        self.text.event_generate('<Delete>')
        if self._highlighting:
            self._syntax_highlighter.highlight_visible()

    def gen_selectall_event(self):
        self.text.event_generate('<<SelectAll>>')

    def gen_copy_event(self):
        self.text.event_generate('<<Copy>>')

    def format_code(self):
        text = self.text.get("1.0", "end-1c")
        self.text.replace("1.0", "end-1c",
                          '\n'.join(dgenerate.textprocessing.format_dgenerate_config(
                              iter(text.splitlines()),
                              indentation=self.indentation)))

    def _indent(self, event):
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
