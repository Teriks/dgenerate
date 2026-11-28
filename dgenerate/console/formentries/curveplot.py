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

"""Plot, pointer, keyboard, and undo handling shared by the schedule editors."""

import collections
import math
import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk

import dgenerate.console.formentries.entry as _entry
import dgenerate.console.sigmaschedule as _schedule
import dgenerate.console.textentry as _t_entry

HINT = 'Drag a point, or select one and use the arrow keys. Ctrl+Z undoes.'

_TEXT_CLASSES = ('Entry', 'TEntry', 'Spinbox', 'TSpinbox', 'TCombobox')

_Frame = collections.namedtuple('_Frame', 'left top span_x span_y axis_top')


def nice_step(span: float, count: int) -> float:
    """A 1, 2, 2.5, or 5 times a power of ten close to ``span / count``."""
    raw = span / max(count, 1)
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    return min(
        (factor * magnitude for factor in (1, 2, 2.5, 5, 10)),
        key=lambda step: abs(math.log(step / raw)))


def linear_ticks(top: float, count: int = 4) -> list[float]:
    step = nice_step(top, count)
    ticks = []
    index = 0
    while index * step <= top * (1 + 1e-9):
        ticks.append(index * step)
        index += 1
    return ticks


def log_ticks(bottom: float, top: float, most: int = 10) -> list[float]:
    """Powers of ten in range, with 2 and 5 between them when ``most`` allows."""
    low = math.floor(math.log10(bottom))
    high = math.ceil(math.log10(top))

    def ticks(multipliers):
        return [
            multiplier * 10 ** power
            for power in range(low, high + 1)
            for multiplier in multipliers
            if bottom <= multiplier * 10 ** power <= top]

    dense = ticks((1, 2, 5))
    return dense if len(dense) <= most else ticks((1,))


def tick_label(value: float) -> str:
    if value == 0:
        return '0'
    if abs(value) >= 1000:
        return f'{value:.0f}'
    return f'{value:.4g}' if abs(value) >= 1 else f'{value:.3g}'


class CurvePlot:
    """
    Mixin for a ``tk.Toplevel`` that edits one descending list of numbers.

    The dialog provides ``canvas``, ``values``, ``point_var``, ``values_var``,
    ``values_entry`` and ``_notice``, and implements the hooks below the
    divider. Point edits go through an undo stack. Changes made with the
    dialog's own controls rebuild the curve and clear that stack.
    """

    def _init_plot(self):
        self._drag_index = None
        self._drag_top = None
        self._drag_bottom = None
        self._drag_log = None
        self._drag_left = None
        self._pending = None
        self._selected = None
        self._hovered = None
        self._undo = []
        self._redo = []
        self._typed_notice = ''
        self._shown_values = ''
        self._font = None
        try:
            self._px = max(1.0, float(self.winfo_fpixels('1i')) / 96.0)
        except tk.TclError:
            self._px = 1.0

    def _build_plot(self, parent, row: int) -> int:
        """Canvas, status, hint, and values rows from ``row``. Returns the next row."""
        self.canvas = tk.Canvas(
            parent, height=int(250 * self._px), highlightthickness=0, borderwidth=0,
            takefocus=1)
        self.canvas.grid(row=row, column=0, columnspan=3, pady=(4, 2), sticky='nsew')
        parent.rowconfigure(row, weight=1)
        canvas = self.canvas
        canvas.bind('<Configure>', lambda _event: self._draw())
        canvas.bind('<Button-1>', self._press)
        canvas.bind('<B1-Motion>', self._drag)
        canvas.bind('<ButtonRelease-1>', lambda _event: self._release())
        canvas.bind('<Motion>', self._hover)
        canvas.bind('<Leave>', self._leave)
        canvas.bind('<FocusIn>', lambda _event: self._draw())
        canvas.bind('<FocusOut>', lambda _event: self._draw())
        for key, direction in (('Up', 1), ('Down', -1)):
            canvas.bind(f'<{key}>', lambda _event, d=direction: self._nudge(d, False))
            canvas.bind(f'<Shift-{key}>', lambda _event, d=direction: self._nudge(d, True))
        canvas.bind('<Left>', lambda _event: self._select_by(-1))
        canvas.bind('<Right>', lambda _event: self._select_by(1))
        canvas.bind('<Home>', lambda _event: self._select_at(0))
        canvas.bind('<End>', lambda _event: self._select_at(len(self.values) - 1))
        canvas.bind('<Tab>', self._tab_out)
        self.bind('<Control-z>', self._undo_key)
        self.bind('<Control-y>', self._redo_key)
        self.bind('<Control-Z>', self._redo_key)
        row += 1

        self.point_var = tk.StringVar(value='')
        ttk.Label(parent, textvariable=self.point_var, anchor='w', wraplength=int(520 * self._px)).grid(
            row=row, column=0, columnspan=3, sticky='w')
        row += 1

        ttk.Label(parent, text=HINT, anchor='w', wraplength=int(520 * self._px)).grid(
            row=row, column=0, columnspan=3, pady=(0, 4), sticky='w')
        row += 1

        ttk.Label(parent, text='Values', anchor='e').grid(
            row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.values_var = tk.StringVar(value='')
        self.values_entry = _t_entry.TextEntry(parent, textvariable=self.values_var)
        self.values_entry.grid(row=row, column=1, columnspan=2, padx=_entry.ROW_XPAD, sticky='ew')
        self.values_entry.bind('<Return>', self._commit_key)
        self.values_entry.bind('<KP_Enter>', self._commit_key)
        self.values_entry.bind('<FocusOut>', lambda _event: self._commit_typed())
        self.values_entry.bind('<KeyRelease>', lambda _event: self._check_typed())
        self.values_entry.bind('<Escape>', self._revert_typed)
        return row + 1

    # --- geometry ---------------------------------------------------------

    def _log_axis(self) -> bool:
        # The pointer-to-value mapping must not change mid-drag.
        if self._drag_log is not None:
            return self._drag_log
        return self._wants_log_axis()

    def _wants_log_axis(self) -> bool:
        return False

    def _axis_bottom(self) -> float:
        return 0.0

    def _plot_font(self):
        if self._font is None:
            self._font = tkfont.nametofont('TkDefaultFont')
        return self._font

    def _floor(self, top: float) -> float:
        return min(self._axis_bottom(), top / 2)

    def _y_ticks(self, top: float) -> list[float]:
        if self._log_axis():
            height = self._size()[1]
            most = max(int(height // (self._plot_font().metrics('linespace') * 1.5)), 3)
            return log_ticks(self._floor(top), top, most)
        return linear_ticks(top)

    def _size(self) -> tuple[int, int]:
        """Canvas size, or its requested size before it is mapped."""
        canvas = self.canvas
        width, height = int(canvas.winfo_width()), int(canvas.winfo_height())
        if width <= 1 or height <= 1:
            width, height = int(canvas.winfo_reqwidth()), int(canvas.winfo_reqheight())
        return max(width, 2), max(height, 2)

    def _frame(self) -> _Frame:
        width, height = self._size()
        top = max(self._axis_top(), 1e-6)
        font = self._plot_font()
        if self._drag_left is not None:
            left = self._drag_left
        else:
            widths = [font.measure(tick_label(tick)) for tick in self._y_ticks(top)]
            left = max(widths + [0]) + int(12 * self._px)
        right = int(16 * self._px)
        upper = int(14 * self._px)
        lower = font.metrics('linespace') + int(10 * self._px)
        return _Frame(
            left, upper,
            max(width - left - right, 1),
            max(height - upper - lower, 1),
            top)

    def _fraction(self, value: float, top: float) -> float:
        """Height of ``value`` on the axis, 0 at the bottom and 1 at the top."""
        if self._log_axis():
            bottom = self._floor(top)
            value = min(max(value, bottom), top)
            return math.log(value / bottom) / math.log(top / bottom)
        return min(max(value, 0.0), top) / top

    def _y_to_value(self, y: float, top: float, pad_y: float, span_y: float) -> float:
        t = min(max(1 - (y - pad_y) / span_y, 0.0), 1.0)
        if self._log_axis():
            bottom = self._floor(top)
            return bottom * (top / bottom) ** t
        return t * top

    def _value_y(self, value: float, frame: _Frame) -> float:
        return frame.top + (1 - self._fraction(value, frame.axis_top)) * frame.span_y

    def _curve(self, values, frame: _Frame):
        count = max(len(values) - 1, 1)
        return [
            (frame.left + (index / count) * frame.span_x, self._value_y(value, frame))
            for index, value in enumerate(values)]

    def _points(self):
        frame = self._frame()
        return self._curve(self.values, frame), frame.axis_top, frame.top, frame.span_y

    # --- drawing ----------------------------------------------------------

    def _x_ticks(self, count: int, span_x: float) -> list[int]:
        font = self._plot_font()
        room = font.measure(str(count)) + int(18 * self._px)
        most = max(int(span_x // room), 2)
        stride = max(1, math.ceil(count / most))
        ticks = list(range(0, count, stride))
        if ticks[-1] != count - 1:
            if len(ticks) > 1 and count - 1 - ticks[-1] < stride:
                ticks.pop()
            ticks.append(count - 1)
        return ticks

    def _active_index(self):
        if self._drag_index is not None:
            return self._drag_index
        if self._hovered is not None:
            return self._hovered
        return self._selected

    def _draw(self):
        if not self.canvas.winfo_exists():
            return
        canvas = self.canvas
        canvas.delete('all')
        if self._selected is not None and self._selected >= len(self.values):
            self._selected = None
        if len(self.values) < 2:
            return
        from dgenerate.console import themedialog
        color = themedialog.foreground() or '#202020'
        muted = themedialog.edge() or '#b4b4b4'
        background = canvas.cget('background')
        font = self._plot_font()
        frame = self._frame()
        right = frame.left + frame.span_x
        bottom = frame.top + frame.span_y

        for tick in self._y_ticks(frame.axis_top):
            y = self._value_y(tick, frame)
            canvas.create_line(frame.left, y, right, y, fill=muted, dash=(2, 4))
            canvas.create_text(
                frame.left - int(6 * self._px), y, text=tick_label(tick),
                anchor='e', fill=color, font=font)
        canvas.create_line(frame.left, frame.top, frame.left, bottom, fill=muted)
        canvas.create_line(frame.left, bottom, right, bottom, fill=muted)
        count = max(len(self.values) - 1, 1)
        for index in self._x_ticks(len(self.values), frame.span_x):
            x = frame.left + (index / count) * frame.span_x
            canvas.create_line(x, bottom, x, bottom + int(3 * self._px), fill=muted)
            canvas.create_text(
                x, bottom + int(4 * self._px), text=str(index + 1),
                anchor='n', fill=color, font=font)

        ceiling = self._ceiling()
        if ceiling is not None and ceiling <= frame.axis_top:
            y = self._value_y(ceiling, frame)
            canvas.create_line(frame.left, y, right, y, fill=color, dash=(6, 4))
            canvas.create_text(
                right, y - int(2 * self._px), text=f'max {tick_label(ceiling)}',
                anchor='se', fill=color, font=font)

        radius = 4 * self._px
        reference = self._reference()
        if reference and len(reference) >= 2 and self._differs(reference):
            faded = self._blend(color, background, 0.5)
            start = self._curve(reference, frame)
            flat = [number for point in start for number in point]
            canvas.create_line(*flat, fill=faded, width=2, dash=(5, 3))
            if len(reference) == len(self.values):
                for (x, y), was, now in zip(start, reference, self.values):
                    if abs(was - now) > 1e-9:
                        canvas.create_oval(
                            x - radius, y - radius, x + radius, y + radius, outline=faded, width=2)

        coords = self._curve(self.values, frame)
        flat = [number for point in coords for number in point]
        canvas.create_line(*flat, fill=color, width=2)
        for x, y in coords:
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)

        if self._selected is not None:
            x, y = coords[self._selected]
            ring = radius + 4 * self._px
            canvas.create_oval(x - ring, y - ring, x + ring, y + ring, outline=color, width=2)

        active = self._active_index()
        if active is not None:
            x, y = coords[active]
            grow = radius + 2 * self._px
            canvas.create_oval(x - grow, y - grow, x + grow, y + grow, fill=color, outline=color)
            below = bool(reference) and len(reference) == len(self.values) and \
                reference[active] > self.values[active]
            self._label(x, y, self._format(self.values[active]), frame, color, background, below)

        if self._focused():
            width, height = self._size()
            canvas.create_rectangle(1, 1, width - 2, height - 2, outline=muted, dash=(1, 2))

    def _blend(self, first, second, amount: float) -> str:
        try:
            one = self.canvas.winfo_rgb(first)
            two = self.canvas.winfo_rgb(second)
        except tk.TclError:
            return first
        mixed = (round((a + (b - a) * amount) / 257) for a, b in zip(one, two))
        return '#{:02x}{:02x}{:02x}'.format(*mixed)

    def _focused(self) -> bool:
        try:
            return self.canvas.focus_get() is self.canvas
        except (KeyError, tk.TclError):
            # A combobox list that is open has focus and no Python widget.
            return False

    def _label(self, x, y, text, frame: _Frame, color, background, below=False):
        """Value tag beside a point, below it when ``below`` and there is room."""
        canvas = self.canvas
        gap = int(10 * self._px)
        anchor_x = 'w' if x < frame.left + frame.span_x * 0.7 else 'e'
        room_above = y > frame.top + gap * 2
        room_below = y < frame.top + frame.span_y - gap * 2
        anchor_y = 'n' if (below and room_below) or not room_above else 's'
        text_x = x + gap if anchor_x == 'w' else x - gap
        text_y = y - gap / 2 if anchor_y == 's' else y + gap / 2
        item = canvas.create_text(
            text_x, text_y, text=text, anchor=anchor_y + anchor_x, fill=color, font=self._plot_font())
        left, upper, right_edge, lower = canvas.bbox(item)
        box = canvas.create_rectangle(
            left - 3, upper - 1, right_edge + 3, lower + 1, fill=background, outline=color)
        canvas.tag_lower(box, item)

    def _differs(self, reference) -> bool:
        if len(reference) != len(self.values):
            return True
        return any(abs(left - right) > 1e-9 for left, right in zip(reference, self.values))

    # --- pointer and keyboard ---------------------------------------------

    def _nearest(self, event, reach: float):
        coords, *_rest = self._points()
        nearest = None
        distance = reach * self._px
        for index, (x, y) in enumerate(coords):
            gap = math.hypot(event.x - x, event.y - y)
            if gap <= distance:
                nearest = index
                distance = gap
        return nearest

    def _press(self, event):
        self.canvas.focus_set()
        nearest = self._nearest(event, 12)
        self._selected = nearest
        self._drag_index = nearest
        if nearest is None:
            self._show_point(None)
            self._draw()
            return
        self._pending = self._snapshot()
        self._drag_log = self._wants_log_axis()
        self._drag_top = self._axis_top()
        self._drag_bottom = self._axis_bottom() if self._drag_log else None
        self._drag_left = self._frame().left
        self.canvas.configure(cursor='sb_v_double_arrow')
        self._move(event)

    def _drag(self, event):
        if self._drag_index is not None:
            self._move(event)

    def _move(self, event):
        _coords, top, pad_y, span_y = self._points()
        if event.y < pad_y and self._ceiling() is None:
            self._drag_top = max(min(top * 1.25, self._drag_limit()), top)
            top = self._drag_top
        if event.y > pad_y + span_y and self._log_axis():
            # A log axis has no 0. Dragging past the floor lowers it.
            self._drag_bottom = max(self._axis_bottom() / 1.25, 1e-6)
        index = self._drag_index
        self._place(index, self._y_to_value(event.y, top, pad_y, span_y))
        self._changed(index)

    def _release(self):
        if self._pending is not None and self._snapshot() != self._pending:
            self._undo.append(self._pending)
            self._redo.clear()
        self._pending = None
        self._drag_index = None
        self._drag_top = None
        self._drag_bottom = None
        self._drag_log = None
        self._drag_left = None
        self.canvas.configure(cursor='')
        self._draw()

    def _hover(self, event):
        if self._drag_index is not None:
            return
        nearest = self._nearest(event, 14)
        self.canvas.configure(cursor='hand2' if nearest is not None else '')
        if nearest != self._hovered:
            self._hovered = nearest
            self._show_point(nearest)
            self._draw()

    def _leave(self, _event):
        if self._drag_index is not None or self._hovered is None:
            return
        self._hovered = None
        self._show_point(None)
        self._draw()

    def _select_at(self, index: int):
        if self.values:
            self._selected = min(max(index, 0), len(self.values) - 1)
            self._show_point(self._selected)
            self._draw()
        return 'break'

    def _select_by(self, offset: int):
        start = self._selected if self._selected is not None else (-1 if offset > 0 else len(self.values))
        return self._select_at(start + offset)

    def _tab_out(self, _event):
        self.canvas.tk_focusNext().focus_set()
        return 'break'

    def _nudge(self, direction: int, large: bool):
        if not self.values or self._drag_index is not None:
            return 'break'
        if self._selected is None:
            self._selected = 0
        index = self._selected
        before = self._snapshot()
        self._place(index, self._nudged(index, direction, large))
        if self._snapshot() != before:
            self._undo.append(before)
            self._redo.clear()
        self._changed(index)
        return 'break'

    # --- undo -------------------------------------------------------------

    def _forget_history(self):
        self._undo.clear()
        self._redo.clear()

    def _typing(self, event) -> bool:
        try:
            return event.widget.winfo_class() in _TEXT_CLASSES
        except (AttributeError, tk.TclError):
            return False

    def _undo_key(self, event):
        if self._typing(event):
            return None
        self.undo()
        return 'break'

    def _redo_key(self, event):
        if self._typing(event):
            return None
        self.redo()
        return 'break'

    def undo(self):
        if not self._undo or self._drag_index is not None:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self._after_restore()

    def redo(self):
        if not self._redo or self._drag_index is not None:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self._after_restore()

    def _after_restore(self):
        self._clear_typed()
        self._sync_values()
        self._show_point(self._selected)
        self._draw()

    # --- values field -----------------------------------------------------

    def _sync_values(self):
        self._shown_values = ', '.join(self._format(value) for value in self.values)
        self.values_var.set(self._shown_values)
        self._clear_typed()

    def _typed_text(self) -> str:
        return self.values_var.get().strip()

    def _check_typed(self):
        text = self._typed_text()
        if text == self._shown_values:
            self._clear_typed()
            return
        try:
            self._parse_typed(text)
        except _schedule.SigmaScheduleError as error:
            self._typed_notice = str(error)
            _entry.invalid_colors(self.values_entry)
        else:
            self._clear_typed()
            return
        self._show_point(None)

    def _clear_typed(self):
        if self._typed_notice:
            self._typed_notice = ''
            self._show_point(self._selected)
        _entry.valid_colors(self.values_entry)

    def _commit_key(self, _event):
        self._commit_typed()
        return 'break'

    def _commit_typed(self) -> bool:
        """Apply typed values. ``False`` leaves the field marked invalid."""
        text = self._typed_text()
        if text == self._shown_values:
            self._clear_typed()
            return True
        try:
            values = self._parse_typed(text)
        except _schedule.SigmaScheduleError as error:
            self._typed_notice = str(error)
            _entry.invalid_colors(self.values_entry)
            self._show_point(None)
            return False
        before = self._snapshot()
        self._accept_typed(values)
        if self._snapshot() != before:
            self._undo.append(before)
            self._redo.clear()
        self._sync_values()
        self._show_point(self._selected)
        self._draw()
        return True

    def _revert_typed(self, _event):
        if self._typed_text() == self._shown_values:
            return None
        self.values_var.set(self._shown_values)
        self._clear_typed()
        return 'break'

    # --- status -----------------------------------------------------------

    def _show_point(self, index):
        notice = self._typed_notice or self._notice
        if notice:
            self.point_var.set(notice)
            return
        if index is None:
            index = self._selected
        if index is None or not self.values:
            self.point_var.set('')
            return
        text = f'Point {index + 1} of {len(self.values)}: {self._format(self.values[index])}'
        reference = self._reference()
        if reference and len(reference) == len(self.values) and abs(reference[index] - self.values[index]) > 1e-9:
            text += f' (started at {self._format(reference[index])})'
        self.point_var.set(text)

    # --- hooks ------------------------------------------------------------

    def _axis_top(self) -> float:
        raise NotImplementedError

    def _ceiling(self):
        raise NotImplementedError

    def _drag_limit(self) -> float:
        """Highest the axis grows when a drag runs past the top without a ceiling."""
        return max(self.values, default=1.0) * 2

    def _reference(self) -> list | None:
        """Curve the edits started from, drawn behind them."""
        raise NotImplementedError

    def _format(self, value) -> str:
        raise NotImplementedError

    def _place(self, index: int, proposed: float):
        """Move one point to ``proposed`` and mark the curve edited."""
        raise NotImplementedError

    def _nudged(self, index: int, direction: int, large: bool) -> float:
        raise NotImplementedError

    def _changed(self, index: int):
        """Refresh after a point edit."""
        self._sync_values()
        self._show_point(index)
        self._draw()

    def _snapshot(self):
        raise NotImplementedError

    def _restore(self, state):
        raise NotImplementedError

    def _parse_typed(self, text: str) -> list:
        """Typed values, or ``SigmaScheduleError`` naming the problem."""
        raise NotImplementedError

    def _accept_typed(self, values: list):
        raise NotImplementedError
