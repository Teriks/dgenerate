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

import pygments
import tklinenums
import toml

import dgenerate.console.resources as _resources
import dgenerate.pygments


class ColorSchemeParser:
    def __init__(self, text_widget, lexer):
        self.lexer = lexer
        self.text = text_widget

        self._editor_keys_map = {
            "background": "bg",
            "foreground": "fg",
            "selectbackground": "select_bg",
            "selectforeground": "select_fg",
            "inactiveselectbackground": "inactive_select_bg",
            "insertbackground": "caret",
            "insertwidth": "caret_width",
            "borderwidth": "border_width",
            "highlightthickness": "focus_border_width",
        }
        self._extras = {
            "Error": "error",
            "Literal.Date": "date",
        }
        self._keywords = {
            "Keyword.Constant": "constant",
            "Keyword.Declaration": "declaration",
            "Keyword.Namespace": "namespace",
            "Keyword.Pseudo": "pseudo",
            "Keyword.Reserved": "reserved",
            "Keyword.Type": "type",
        }
        self._names = {
            "Name.Attribute": "attr",
            "Name.Builtin": "builtin",
            "Name.Builtin.Pseudo": "builtin_pseudo",
            "Name.Class": "class",
            "Name.Constant": "constant",
            "Name.Decorator": "decorator",
            "Name.Entity": "entity",
            "Name.Exception": "exception",
            "Name.Function": "function",
            "Name.Function.Magic": "magic_function",
            "Name.Label": "label",
            "Name.Namespace": "namespace",
            "Name.Tag": "tag",
            "Name.Variable": "variable",
            "Name.Variable.Class": "class_variable",
            "Name.Variable.Global": "global_variable",
            "Name.Variable.Instance": "instance_variable",
            "Name.Variable.Magic": "magic_variable",
        }
        self._strings = {
            "Literal.String.Affix": "affix",
            "Literal.String.Backtick": "backtick",
            "Literal.String.Char": "char",
            "Literal.String.Delimeter": "delimeter",
            "Literal.String.Doc": "doc",
            "Literal.String.Double": "double",
            "Literal.String.Escape": "escape",
            "Literal.String.Heredoc": "heredoc",
            "Literal.String.Interpol": "interpol",
            "Literal.String.Regex": "regex",
            "Literal.String.Single": "single",
            "Literal.String.Symbol": "symbol",
        }
        self._numbers = {
            "Literal.Number.Bin": "binary",
            "Literal.Number.Float": "float",
            "Literal.Number.Hex": "hex",
            "Literal.Number.Integer": "integer",
            "Literal.Number.Integer.Long": "long",
            "Literal.Number.Oct": "octal",
        }
        self._comments = {
            "Comment.Hashbang": "hashbang",
            "Comment.Multiline": "multiline",
            "Comment.Preproc": "preproc",
            "Comment.PreprocFile": "preprocfile",
            "Comment.Single": "single",
            "Comment.Special": "special",
        }
        self._generic = {
            "Generic.Emph": "emphasis",
            "Generic.Error": "error",
            "Generic.Heading": "heading",
            "Generic.Strong": "strong",
            "Generic.Subheading": "subheading",
        }

    @staticmethod
    def _parse_table(source, map_, fallback=None):
        result = {}
        if source is not None:
            for token, key in map_.items():
                value = source.get(key)
                if value is None:
                    value = fallback
                result[token] = value
        elif fallback is not None:
            for token in map_:
                result[token] = fallback
        return result

    def _parse_theme(self, color_scheme):
        editor = {}
        if "editor" in color_scheme:
            editor_settings = color_scheme["editor"]
            for tk_name, key in self._editor_keys_map.items():
                editor[tk_name] = editor_settings.get(key)

        assert "general" in color_scheme, "General table must present in color scheme"
        general = color_scheme["general"]

        error = general.get("error")
        escape = general.get("escape")
        punctuation = general.get("punctuation")
        general_comment = general.get("comment")
        general_keyword = general.get("keyword")
        general_name = general.get("name")
        general_string = general.get("string")

        tags = {
            "Error": error,
            "Escape": escape,
            "Punctuation": punctuation,
            "Comment": general_comment,
            "Keyword": general_keyword,
            "Keyword.Other": general_keyword,
            "Literal.String": general_string,
            "Literal.String.Other": general_string,
            "Name.Other": general_name,
        }

        tags.update(**self._parse_table(color_scheme.get("keyword"), self._keywords, general_keyword))
        tags.update(**self._parse_table(color_scheme.get("name"), self._names, general_name))
        tags.update(
            **self._parse_table(
                color_scheme.get("operator"),
                {"Operator": "symbol", "Operator.Word": "word"},
            )
        )
        tags.update(**self._parse_table(color_scheme.get("string"), self._strings, general_string))
        tags.update(**self._parse_table(color_scheme.get("number"), self._numbers))
        tags.update(**self._parse_table(color_scheme.get("comment"), self._comments, general_comment))
        tags.update(**self._parse_table(color_scheme.get("generic"), self._generic))
        tags.update(**self._parse_table(color_scheme.get("extras"), self._extras))

        return editor, tags

    def set_theme(self, color_theme: dict[str, dict[str, str | int]] | str):
        if isinstance(color_theme, str):
            color_theme = toml.load(color_theme)

        editor, tags = self._parse_theme(color_theme)
        self.text.configure(**editor)
        for key, value in tags.items():
            if isinstance(value, str):
                self.text.tag_configure(f"Token.{key}", foreground=value)
        self.highlight_all()

    def highlight_all(self) -> None:
        # Remove existing syntax highlighting tags
        for tag in self.text.tag_names(index=None):
            if tag.startswith("Token"):
                self.text.tag_remove(tag, "1.0", "end")

        # Get the index range of the visible lines
        first_visible_index = self.text.index("@0,0")
        last_visible_line_num = int(self.text.index("@0,%d" % self.text.winfo_height()).split(".")[0]) + 1
        last_visible_index = f"{last_visible_line_num}.0"

        # Extend the range to the start of the text or the first empty line above the visible text
        while first_visible_index != "1.0" and self.text.get(f"{first_visible_index} - 1 line").strip():
            first_visible_index = self.text.index(f"{first_visible_index} - 1 line")

        # Extend the range to the end of the text or the first empty line below the visible text
        while last_visible_index != self.text.index("end") and self.text.get(last_visible_index).strip():
            last_visible_index = self.text.index(f"{last_visible_index} + 1 line")

        # Get the text of the extended range
        lines = self.text.get(first_visible_index, last_visible_index)
        line_offset = lines.count("\n") - lines.lstrip().count("\n")
        start_index = str(self.text.index(f"{first_visible_index} + {line_offset} lines"))

        # Apply syntax highlighting to the extended range
        for token, text in pygments.lex(lines, self.lexer):
            token = str(token)
            end_index = self.text.index(f"{start_index} + {len(text)} chars")
            if token not in {"Token.Text.Whitespace", "Token.Text"}:
                self.text.tag_add(token, start_index, end_index)
            start_index = end_index

    def highlight_block(self, index: str) -> None:
        index = self.text.index(index).split('.')[0] + '.' + '0'

        # Find the start of the block
        start_index = index
        while start_index != "1.0" and self.text.get(f"{start_index} - 1 line").strip():
            start_index = self.text.index(f"{start_index} - 1 line")

        # Find the end of the block
        end_index = index
        while end_index != self.text.index("end") and self.text.get(end_index).strip():
            end_index = self.text.index(f"{end_index} + 1 line")

        # Remove existing highlights in the block
        for tag in self.text.tag_names(index=None):
            if tag.startswith("Token"):
                self.text.tag_remove(tag, start_index, end_index)

        # Get the text of the block
        lines = self.text.get(start_index, end_index)
        line_offset = lines.count("\n") - lines.lstrip().count("\n")
        start_index = str(self.text.index(f"{start_index} + {line_offset} lines"))

        # Apply syntax highlighting to the block
        for token, text in pygments.lex(lines, self.lexer):
            token = str(token)
            end_index = self.text.index(f"{start_index} + {len(text)} chars")
            if token not in {"Token.Text.Whitespace", "Token.Text"}:
                self.text.tag_add(token, start_index, end_index)
            start_index = end_index


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

        self._color_scheme = ColorSchemeParser(self.text, dgenerate.pygments.DgenerateLexer())

        kwargs.setdefault('font', ('monospaced', 11))

        self.text.configure(
            yscrollcommand=self._vertical_scroll,
            xscrollcommand=self._horizontal_scroll,
            font=kwargs['font'],
            tabs=tk.font.Font(font=kwargs["font"]).measure(" " * 4),
        )

        def key_release(event):
            if len(event.char) == 1 and event.char.isprintable():
                self._color_scheme.highlight_block('insert')

        self.text.bind('<<Modified>>', lambda e: (self._line_numbers.redraw(), self.text.edit_modified(False)))
        self.text.bind('<KeyRelease>', key_release)
        self.text.bind('<Tab>', self._indent)
        self.text.bind('<Shift-Tab>', self._unindent)

        replace = self.text.replace

        def replace_new(*a, **kwargs):
            replace(*a, **kwargs)
            self._color_scheme.highlight_all()

        self.text.replace = replace_new

        insert = self.text.insert

        def insert_new(*a, **kwargs):
            insert(*a, **kwargs)
            self._color_scheme.highlight_all()

        self.text.insert = insert_new

        delete = self.text.delete

        def delete_new(*a, **kwargs):
            delete(*a, **kwargs)
            self._color_scheme.highlight_all()

        self.text.delete = delete_new

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

        # Prevent the default Shift-Tab behavior
        return 'break'

    def _horizontal_scroll(self, first: str | float, last: str | float):
        self.x_scrollbar.set(first, last)

    def _vertical_scroll(self, first: str | float, last: str | float):
        self.y_scrollbar.set(first, last)
        self._line_numbers.redraw()
        self._color_scheme.highlight_all()

    def set_theme(self, color_theme: dict[str, dict[str, str | int]] | str | None) -> None:
        if isinstance(color_theme, str):
            color_theme = DgenerateCodeView.THEMES[color_theme]

        self._color_scheme.set_theme(color_theme)

    def enable_line_numbers(self):
        self._line_numbers.grid(row=0, column=0, sticky='ns')

    def disable_line_numbers(self):
        self._line_numbers.grid_forget()

    def enable_word_wrap(self):
        self.text.config(wrap='word')

    def disable_word_wrap(self):
        self.text.config(wrap='none')
