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
import dgenerate.console.themetext as _themetext
import dgenerate.console.fonts as _fonts


class ScrolledText(tk.Frame):
    def __init__(self, master=None, undo=False, **kwargs):
        super().__init__(master, **kwargs)

        text_args = {}
        if undo:
            text_args['undo'] = True
            text_args['autoseparators'] = True
            text_args['maxundo'] = -1

        self.text = _themetext.ThemeText(
            self,
            wrap='word',
            **text_args
        )

        # Use cross-platform font selection if no font is specified
        if 'font' not in kwargs:
            default_font_family, default_font_size = _fonts.get_default_monospace_font(10)
            kwargs['font'] = (default_font_family, default_font_size)

        self.text.configure(
            font=kwargs['font'],
            tabs=tkfont.Font(font=kwargs["font"]).measure(" " * 4),
        )

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
