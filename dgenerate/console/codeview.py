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

import chlorophyll
import toml

import dgenerate.pygments


class DgenerateCodeView(tk.Frame):
    THEMES = {'dgenerate':
                  '''
                  # Editor colors
                  [editor]
                  bg = "#FFFFFF" # Background color: white
                  fg = "#000000" # Foreground color: black
                  select_bg = "#CCE8FF" # Selected text background: soft blue
                  select_fg = "#4B4B4B" # Selected text background: medium gray
                  inactive_select_bg = "#E0E0E0" # Inactive selected text background: light grey
                  caret = "#000000" # Caret color: black
                  caret_width = 1
                  border_width = 0
                  focus_border_width = 0
                  
                  # General syntax colors
                  [general]
                  comment = "#0000CD" # Comments: darker slate grey
                  error = "#B22222" # Errors: dark red
                  escape = "#5B5BFF" # Escapes: darker violet
                  keyword = "#0000CD" # Keywords: dark pink
                  name = "#1C608B" # Names: darker blue
                  string = "#006400" # Strings: darker teal
                  punctuation = "#404040" # Punctuation: dark grey
                  
                  # Keyword colors
                  [keyword]
                  constant = "#556B2F" # Constants: dark olive green
                  declaration = "#8B4500" # Declarations: dark amber
                  namespace = "#8B2500" # Namespaces: darker orange
                  pseudo = "#BC8F8F" # Pseudo keywords: rosy brown
                  reserved = "#696969" # Reserved keywords: dim grey
                  type = "#483D8B" # Types: dark slate blue
                  
                  # Name colors
                  [name]
                  attr = "#1C608B" # Attributes: darker blue
                  builtin = "#B22222" # Built-in functions: dark pink
                  builtin_pseudo = "#B22222" # Built-in pseudo classes: dark red
                  class = "#8B2500" # Classes: darker orange
                  class_variable = "#5F9EA0" # Class variables: cadet blue
                  constant = "#8B4500" # Constants: dark amber
                  decorator = "#1C608B" # Decorators: darker blue
                  entity = "#1B7B7B" # Entities: darker teal
                  exception = "#556B2F" # Exceptions: dark olive green
                  function = "#483D8B" # Functions: dark slate blue
                  global_variable = "#696969" # Global variables: dim grey
                  instance_variable = "#BC8F8F" # Instance variables: rosy brown
                  label = "#1C608B" # Labels: darker blue
                  magic_function = "#A00073" # Magic functions: dark pink
                  magic_variable = "#B22222" # Magic variables: dark red
                  namespace = "#8B2500" # Namespace: darker orange
                  tag = "#556B2F" # Tags: dark olive green
                  variable = "#8B4500" # Variables: dark amber
                  
                  # Operator colors
                  [operator]
                  symbol = "#614051" # Symbols: dim grey
                  word = "#BC8F8F" # Word operators: rosy brown
                  
                  # String colors
                  [string]
                  affix = "#006400" # Affix: dark green
                  char = "#006400" # Char: dark green
                  delimeter = "#006400" # Delimeter: dark green
                  doc = "#006400" # Doc: dark green
                  double = "#006400" # Double: dark green
                  escape = "#006400" # Escape: dark green
                  heredoc = "#006400" # Heredoc: dark green
                  interpol = "#696969" # Interpol: dim gray
                  regex = "#006400" # Regex: dark green
                  single = "#006400" # Single: dark green
                  symbol = "#006400" # Symbol: dark green
                  
                  # Number colors
                  [number]
                  binary = "#FF6347" # Binary: tomato
                  float = "#FF4500" # Float: gold
                  hex = "#FF4500" # Integer: orange red
                  integer = "#FF4500" # Integer: orange red
                  long = "#FF8C00" # Long: dark orange
                  octal = "#FF1493" # Octal: deep pink
                  
                  # Comment colors
                  [comment]
                  hashbang = "#5F9EA0" # Hashbang: medium blue
                  multiline = "#5F9EA0" # Multiline: medium blue
                  preproc = "#5F9EA0" # Preproc: rosy brown
                  preprocfile = "#5F9EA0" # Preprocfile: darker teal
                  single = "#5F9EA0" # Single: medium blue
                  special = "#5F9EA0" # Special: dark amber
                  ''',
              'readthedocs':
                  '''
                  # Editor colors
                  [editor]
                  bg = "#f8f8f8" # Background color: light grey
                  fg = "#000000" # Foreground color: black
                  select_bg = "#ffffcc" # Selected text background: light yellow
                  select_fg = "#000000" # Selected text foreground: black
                  inactive_select_bg = "#efefef" # Inactive selected text background: very light grey
                  caret = "#000000" # Caret color: black
                  caret_width = 1
                  border_width = 0
                  focus_border_width = 0
                  
                  # General syntax colors
                  [general]
                  comment = "#3D7B7B" # Comments: teal, italic
                  error = "#FF0000" # Errors: red, border
                  keyword = "#008000" # Keywords: green, bold
                  operator = "#666666" # Operators: dark grey
                  name = "#000000" # Names: black
                  string = "#BA2121" # Strings: red-brown
                  punctuation = "#000000" # Punctuation: black
                  
                  # Keyword colors
                  [keyword]
                  constant = "#008000" # Constants: green, bold
                  declaration = "#008000" # Declarations: green, bold
                  namespace = "#008000" # Namespaces: green, bold
                  pseudo = "#008000" # Pseudo keywords: green
                  reserved = "#008000" # Reserved keywords: green, bold
                  type = "#B00040" # Types: dark pink
                  
                  # Name colors
                  [name]
                  attr = "#687822" # Attributes: olive
                  builtin = "#008000" # Built-in functions: green
                  builtin_pseudo = "#008000" # Built-in pseudo classes: green
                  class = "#0000FF" # Classes: blue, bold
                  class_variable = "#19177C" # Class variables: dark blue
                  constant = "#880000" # Constants: maroon
                  decorator = "#AA22FF" # Decorators: violet
                  entity = "#717171" # Entities: grey
                  exception = "#CB3F38" # Exceptions: red
                  function = "#0000FF" # Functions: blue
                  global_variable = "#19177C" # Global variables: dark blue
                  instance_variable = "#19177C" # Instance variables: dark blue
                  label = "#767600" # Labels: dark yellow
                  magic_function = "#0000FF" # Magic functions: blue
                  magic_variable = "#19177C" # Magic variables: dark blue
                  namespace = "#0000FF" # Namespace: blue, bold
                  tag = "#008000" # Tags: green, bold
                  variable = "#19177C" # Variables: dark blue
                  
                  # Operator colors
                  [operator]
                  symbol = "#666666" # Symbols: dark grey
                  word = "#AA22FF" # Word operators: violet
                  
                  # String colors
                  [string]
                  affix = "#BA2121" # Affix: red-brown
                  char = "#BA2121" # Char: red-brown
                  delimeter = "#BA2121" # Delimeter: red-brown
                  doc = "#BA2121" # Doc: red-brown, italic
                  double = "#BA2121" # Double: red-brown
                  escape = "#AA5D1F" # Escape: brown, bold
                  heredoc = "#BA2121" # Heredoc: red-brown
                  interpol = "#A45A77" # Interpol: pink, bold
                  regex = "#A45A77" # Regex: pink
                  single = "#BA2121" # Single: red-brown
                  symbol = "#19177C" # Symbol: dark blue
                  
                  # Number colors
                  [number]
                  binary = "#666666" # Binary: dark grey
                  float = "#666666" # Float: dark grey
                  hex = "#666666" # Hex: dark grey
                  integer = "#666666" # Integer: dark grey
                  long = "#666666" # Long: dark grey
                  octal = "#666666" # Octal: dark grey
                  
                  # Comment colors
                  [comment]
                  hashbang = "#3D7B7B" # Hashbang: teal, italic
                  multiline = "#3D7B7B" # Multiline: teal, italic
                  preproc = "#9C6500" # Preproc: brown
                  preprocfile = "#3D7B7B" # Preprocfile: teal, italic
                  single = "#3D7B7B" # Single: teal, italic
                  special = "#3D7B7B" # Special: teal, italic
                  '''}

    def __init__(self, master=None, undo=False, **kwargs):
        super().__init__(master, **kwargs)

        text_args = {}
        if undo:
            text_args['undo'] = True
            text_args['autoseparators'] = True
            text_args['maxundo'] = -1

        self.text = chlorophyll.CodeView(self, wrap='word',
                                         autohide_scrollbar=False,
                                         color_scheme=toml.loads(DgenerateCodeView.THEMES['dgenerate']),
                                         lexer=dgenerate.pygments.DgenerateLexer(), **text_args)
        font = tk.font.Font(font=self.text['font'])
        self.text.config(tabs=font.measure(' ' * 4))
        self.text.pack(side='left', fill='both', expand=True)

        self.text._cmd_proxy = lambda *args: None
        self.text.highlight_line = lambda *args: None

        def modified(e):
            self.text.edit_modified(False)
            self.text.highlight_all()

        self.text.bind('<<Modified>>', modified)

        ors = self.text.replace

        def replace(*a, **kwargs):
            ors(*a, **kwargs)
            self.text.highlight_all()

        self.text.replace = replace

        ors2 = self.text.insert

        def insert(*a, **kwargs):
            ors2(*a, **kwargs)
            self.text.highlight_all()

        self.text.insert = insert

        self.text._line_numbers.grid_forget()

    def set_theme(self, theme):
        if theme in DgenerateCodeView.THEMES:
            self.text._set_color_scheme(toml.loads(DgenerateCodeView.THEMES[theme]))
        else:
            self.text._set_color_scheme(theme)

    def enable_line_numbers(self):
        self.text._line_numbers.grid(row=0, column=0, sticky="ns")

    def disable_line_numbers(self):
        self.text._line_numbers.grid_forget()

    def enable_word_wrap(self):
        self.text.config(wrap='word')

    def disable_word_wrap(self):
        self.text.config(wrap='none')
