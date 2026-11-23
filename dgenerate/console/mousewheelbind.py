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

# Tk 8.7 and Tk 9 report a trackpad two-finger gesture as TouchpadScroll.
# MouseWheel is only a physical wheel after that split, which is why a
# MacBook trackpad stopped scrolling widgets that bind MouseWheel alone.
_TOUCHPAD_SEQUENCE = '<TouchpadScroll>'
_touchpad_supported = None
# Finger travel, in the event's pixel deltas, before a spinbox or zoom
# control takes one step. Canvas scrolling uses the pixels directly.
_TOUCHPAD_NOTCH = 24
_touchpad_remainder = {}


def precise_scroll_deltas(packed):
    """
    Unpack a TouchpadScroll ``%D`` value the way ``tk::PreciseScrollDeltas`` does.

    :param packed: The event delta, a signed 32-bit packing of delta X and delta Y
    :return: ``(delta_x, delta_y)`` in pixels. Positive Y matches a MouseWheel
        scroll-up.
    """
    value = int(packed) & 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    delta_x = value >> 16
    low = value & 0xFFFF
    delta_y = low if low < 0x8000 else low - 0x10000
    return delta_x, delta_y


def _sequences(modifier):
    if modifier is None:
        return ('<MouseWheel>', '<Button-4>', '<Button-5>')
    return (
        f'<{modifier}-MouseWheel>',
        f'<{modifier}-Button-4>',
        f'<{modifier}-Button-5>',
    )


def _bind_sequence(bind_func, sequence, callback=None):
    try:
        if callback is None:
            bind_func(sequence)
        else:
            bind_func(sequence, callback)
    except tk.TclError:
        return False
    return True


def _mark_touchpad(callback):
    def _on_touchpad(event):
        delta_x, delta_y = precise_scroll_deltas(getattr(event, 'delta', 0) or 0)
        event.touchpad = True
        event.delta_x = delta_x
        event.delta_y = delta_y
        return callback(event)

    return _on_touchpad


def bind_mousewheel(bind_func, callback, modifier=None):
    """
    Bind wheel and trackpad scrolling to ``callback``.

    A trackpad event arrives with ``event.touchpad`` set and ``delta_x`` /
    ``delta_y`` already unpacked. ``event.delta`` on that event is still the
    packed Tk value, so callers should use :func:`scroll_direction` or
    :func:`handle_canvas_scroll` instead of reading it.
    """
    global _touchpad_supported

    for sequence in _sequences(modifier):
        bind_func(sequence, callback)

    if modifier is not None or _touchpad_supported is False:
        return

    bound = _bind_sequence(bind_func, _TOUCHPAD_SEQUENCE, _mark_touchpad(callback))
    if _touchpad_supported is None:
        _touchpad_supported = bound


def un_bind_mousewheel(bind_func, modifier=None):
    for sequence in _sequences(modifier):
        _bind_sequence(bind_func, sequence)
    if modifier is None and _touchpad_supported:
        _bind_sequence(bind_func, _TOUCHPAD_SEQUENCE)


def scroll_direction(event):
    """
    Direction of one wheel or trackpad event.

    :return: Positive to scroll up or zoom in, negative to scroll down or
        zoom out, and ``0`` when the event should be ignored. Trackpad
        movement is accumulated until it amounts to one notch, so a
        spinbox or zoom control does not step on every finger sample.
    """
    if getattr(event, 'touchpad', False):
        delta_x = getattr(event, 'delta_x', 0)
        delta_y = getattr(event, 'delta_y', 0)
        if delta_y == 0 or abs(delta_x) > abs(delta_y):
            return 0
        key = id(getattr(event, 'widget', None))
        total = _touchpad_remainder.get(key, 0) + delta_y
        if abs(total) < _TOUCHPAD_NOTCH:
            _touchpad_remainder[key] = total
            return 0
        steps = int(total / _TOUCHPAD_NOTCH)
        _touchpad_remainder[key] = total - (steps * _TOUCHPAD_NOTCH)
        if steps > 1:
            return 1
        if steps < -1:
            return -1
        return steps

    delta = getattr(event, 'delta', 0) or 0
    if delta:
        if abs(delta) >= 120:
            steps = int(delta / 120)
            return steps if steps else (1 if delta > 0 else -1)
        return 1 if delta > 0 else -1

    number = getattr(event, 'num', None)
    if number == 4:
        return 1
    if number == 5:
        return -1
    return 0


# One pending scroll per canvas. A thumb drag or a free-spinning wheel
# posts an event per step, and each step moves every widget in the form.
# Applying them as they arrive leaves the view catching up after the
# pointer has stopped. The burst is folded into one move.
_queued_yview = {}


def yview_command(canvas):
    """A scrollbar ``command``.

    A thumb drag is applied immediately so the form follows the pointer.
    A wheel burst is collapsed into one move, so a free spin does not keep
    the view catching up after the wheel has stopped.
    """

    def command(*args):
        if args and args[0] == 'moveto':
            drop_queued_yview(canvas)
            try:
                canvas.yview_moveto(args[1])
            except tk.TclError:
                pass
            return
        _enqueue_yview(canvas, args)

    return command


def drop_queued_yview(canvas):
    """Forget a scroll that has not been applied yet."""
    state = _queued_yview.pop(canvas, None)
    if state is None or state['job'] is None:
        return
    try:
        canvas.after_cancel(state['job'])
    except tk.TclError:
        pass


def _enqueue_yview(canvas, args):
    state = _queued_yview.get(canvas)
    if state is None:
        state = {'job': None, 'op': None}
        _queued_yview[canvas] = state

        def drop(_event, canvas=canvas):
            drop_queued_yview(canvas)

        try:
            canvas.bind('<Destroy>', drop, add='+')
        except tk.TclError:
            _queued_yview.pop(canvas, None)
            return
    state['op'] = _merge_yview(state['op'], args)
    if state['job'] is not None:
        return
    try:
        state['job'] = canvas.after(0, _flush_yview, canvas)
    except tk.TclError:
        _queued_yview.pop(canvas, None)


def _merge_yview(op, args):
    kind = args[0]
    if kind == 'moveto':
        return ('moveto', float(args[1]))
    count = int(float(args[1]))
    what = args[2] if len(args) > 2 else 'units'
    if op is not None and op[0] == 'scroll' and op[2] == what:
        return ('scroll', op[1] + count, what)
    return ('scroll', count, what)


def _flush_yview(canvas):
    state = _queued_yview.get(canvas)
    if state is None:
        return
    state['job'] = None
    op = state['op']
    state['op'] = None
    if op is None:
        return
    try:
        if op[0] == 'moveto':
            canvas.yview_moveto(op[1])
        elif op[2] == 'pixels':
            _scroll_pixels(canvas, -op[1])
        else:
            canvas.yview_scroll(op[1], op[2])
    except tk.TclError:
        _queued_yview.pop(canvas, None)


def _scroll_pixels(canvas, delta_y):
    """``delta_y`` positive moves the view up. Older Tk has no pixel units."""
    try:
        canvas.yview_scroll(-int(delta_y), 'pixels')
        return
    except tk.TclError:
        pass
    region = canvas.bbox('all')
    if region is None:
        return
    span = region[3] - region[1]
    if span <= 0:
        return
    top = canvas.yview()[0]
    canvas.yview_moveto(min(1.0, max(0.0, top - (delta_y / span))))


def _scroll_canvas_pixels(canvas, delta_y):
    """Scroll by ``delta_y`` pixels. Positive Y moves the view up."""
    _enqueue_yview(canvas, ('scroll', -int(delta_y), 'pixels'))


def handle_canvas_scroll(canvas: tk.Canvas, event: tk.Event):
    canvas_x = canvas.winfo_rootx()
    canvas_y = canvas.winfo_rooty()
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if not (canvas_x <= event.x_root <= canvas_x + canvas_width and
            canvas_y <= event.y_root <= canvas_y + canvas_height):
        return

    viewable_region = canvas.bbox("all")
    if viewable_region is None:
        return

    content_height = viewable_region[3] - viewable_region[1]
    if content_height <= canvas_height:
        return

    if getattr(event, 'touchpad', False):
        delta_x = getattr(event, 'delta_x', 0)
        delta_y = getattr(event, 'delta_y', 0)
        if delta_y == 0 or abs(delta_x) > abs(delta_y):
            return
        _scroll_canvas_pixels(canvas, delta_y)
        return

    steps = scroll_direction(event)
    if steps:
        _enqueue_yview(canvas, ('scroll', -steps, 'units'))
