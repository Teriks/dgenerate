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

import re
import tkinter as tk

_TOKEN = re.compile(
    r'(?P<csi>\x1b\[(?P<params>[0-?]*)[ -/]*(?P<final>[@-~]))'
    r'|(?P<osc>\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?)'
    r'|(?P<esc>\x1b[@-Z\\-_]?)'
    r'|(?P<ctrl>[\r\n\b])'
    r'|(?P<ignored>[\x00-\x08\x0b-\x1f\x7f])'
    r'|(?P<text>[^\x00-\x08\x0a-\x1f\x7f]+)'
)


class TerminalText:
    """
    Writes process output into a :py:class:`tkinter.Text` the way a terminal would.

    A text mark is the cursor. Carriage return, newline, backspace, cursor
    movement (``ESC[A`` ``ESC[B`` ``ESC[C`` ``ESC[D`` ``ESC[G``) and line erase
    (``ESC[K``) move or edit text at the cursor, so progress bars such as tqdm,
    including nested bars, redraw in place. Other escape sequences are dropped.

    stdout and stderr arrive through separate pipes and their relative order is
    lost. When output switches streams mid-line, the new stream starts on a
    fresh line at the bottom instead of writing over the other stream's line.
    """

    MARK = 'terminal_cursor'

    def __init__(self, text: tk.Text):
        """
        :param text: the output widget, its state must be ``NORMAL`` while writing
        """
        self._text = text
        self._stream = None
        self.reset()

    def reset(self):
        """
        Put the cursor at the end of the widget content.

        Call this after the widget content is replaced outside this writer.
        """
        self._text.mark_set(self.MARK, 'end-1c')
        self._text.mark_gravity(self.MARK, tk.RIGHT)
        self._stream = None

    def write(self, data: str, tag: str | None = None, stream=None):
        """
        Write output at the cursor.

        :param data: decoded output, may contain control characters and escape sequences
        :param tag: text tag for inserted characters, ``None`` for no tag
        :param stream: identity of the source stream, such as ``'stdout'``
        """
        if stream != self._stream:
            if self._stream is not None:
                self._to_fresh_line()
            self._stream = stream

        tags = (tag,) if tag else ()
        for match in _TOKEN.finditer(data):
            kind = match.lastgroup
            if kind == 'text':
                self._put(match.group('text'), tags)
            elif kind == 'ctrl':
                self._control(match.group('ctrl'))
            elif kind == 'csi':
                self._csi(match.group('params'), match.group('final'))

    def _cursor(self) -> tuple[int, int]:
        line, col = self._text.index(self.MARK).split('.')
        return int(line), int(col)

    def _last_line(self) -> int:
        return int(self._text.index('end-1c').split('.')[0])

    def _line_length(self, line: int) -> int:
        return int(self._text.index(f'{line}.end').split('.')[1])

    def _move(self, line: int, col: int):
        line = max(1, line)
        last = self._last_line()
        while line > last:
            self._text.insert('end-1c', '\n')
            last += 1
        length = self._line_length(line)
        if col > length:
            self._text.insert(f'{line}.end', ' ' * (col - length))
        self._text.mark_set(self.MARK, f'{line}.{max(0, col)}')

    def _to_fresh_line(self):
        last = self._last_line()
        if self._line_length(last):
            self._text.insert('end-1c', '\n')
        self._text.mark_set(self.MARK, 'end-1c')

    def _put(self, chars: str, tags: tuple):
        line, col = self._cursor()
        overwrite = min(len(chars), self._line_length(line) - col)
        if overwrite > 0:
            self._text.delete(self.MARK, f'{self.MARK}+{overwrite}c')
        self._text.insert(self.MARK, chars, tags)

    def _control(self, char: str):
        line, col = self._cursor()
        if char == '\r':
            self._text.mark_set(self.MARK, f'{line}.0')
        elif char == '\n':
            self._move(line + 1, 0)
        elif char == '\b' and col > 0:
            self._text.mark_set(self.MARK, f'{line}.{col - 1}')

    def _csi(self, params: str, final: str):
        line, col = self._cursor()
        values = [int(value) for value in re.findall(r'\d+', params)]
        count = values[0] if values else None
        if final == 'A':
            self._move(line - (count or 1), col)
        elif final == 'B':
            self._move(line + (count or 1), col)
        elif final == 'C':
            self._move(line, col + (count or 1))
        elif final == 'D':
            self._move(line, col - (count or 1))
        elif final == 'G':
            self._move(line, (count or 1) - 1)
        elif final == 'K':
            mode = count or 0
            if mode == 0:
                self._text.delete(self.MARK, f'{line}.end')
            elif mode == 1:
                self._text.delete(f'{line}.0', self.MARK)
                self._text.insert(f'{line}.0', ' ' * col)
                self._text.mark_set(self.MARK, f'{line}.{col}')
            elif mode == 2:
                self._text.delete(f'{line}.0', f'{line}.end')
                self._move(line, col)


__all__ = ['TerminalText']
