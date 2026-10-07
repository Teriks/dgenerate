import types
import unittest

import dgenerate.console.mousewheelbind as _wheel

try:
    import tkinter as tk

    _root = tk.Tk()
    _root.withdraw()
    _root.destroy()
except Exception:
    tk = None


class TestPreciseScrollDeltas(unittest.TestCase):
    def test_vertical_down_is_negative_y(self):
        self.assertEqual(_wheel.precise_scroll_deltas(0xFFFD), (0, -3))

    def test_mixed_signs(self):
        self.assertEqual(_wheel.precise_scroll_deltas(-65531), (-1, 5))
        self.assertEqual(_wheel.precise_scroll_deltas(0x0002FFFC), (2, -4))

    def test_unsigned_high_bit(self):
        self.assertEqual(_wheel.precise_scroll_deltas(0xFFFF0005), (-1, 5))


class TestScrollDirection(unittest.TestCase):
    def test_windows_notch_and_macos_tick(self):
        self.assertEqual(_wheel.scroll_direction(types.SimpleNamespace(delta=240, num=0)), 2)
        self.assertEqual(_wheel.scroll_direction(types.SimpleNamespace(delta=-120, num=0)), -1)
        self.assertEqual(_wheel.scroll_direction(types.SimpleNamespace(delta=1, num=0)), 1)
        self.assertEqual(_wheel.scroll_direction(types.SimpleNamespace(delta=0, num=5)), -1)

    def test_trackpad_accumulates_to_one_notch(self):
        widget = object()
        event = types.SimpleNamespace(touchpad=True, delta_x=0, delta_y=10, widget=widget)
        self.assertEqual(_wheel.scroll_direction(event), 0)
        self.assertEqual(_wheel.scroll_direction(event), 0)
        self.assertEqual(_wheel.scroll_direction(event), 1)

    def test_horizontal_trackpad_is_ignored(self):
        event = types.SimpleNamespace(touchpad=True, delta_x=12, delta_y=-2, widget=object())
        self.assertEqual(_wheel.scroll_direction(event), 0)


@unittest.skipIf(tk is None, 'Tk is not available')
class TestCanvasScroll(unittest.TestCase):
    def test_small_mouse_delta_scrolls_one_unit(self):
        root = tk.Tk()
        try:
            root.geometry('200x80+20+20')
            canvas = tk.Canvas(root, width=180, height=60, highlightthickness=0)
            canvas.pack()
            canvas.create_rectangle(0, 0, 20, 800, outline='')
            root.update()
            canvas.configure(scrollregion=canvas.bbox('all'))
            event = types.SimpleNamespace(
                delta=-1,
                num=0,
                x_root=canvas.winfo_rootx() + 4,
                y_root=canvas.winfo_rooty() + 4,
            )
            before = canvas.yview()[0]
            _wheel.handle_canvas_scroll(canvas, event)
            self.assertGreater(canvas.yview()[0], before)
        finally:
            root.destroy()

    def test_trackpad_pixels_move_the_view(self):
        root = tk.Tk()
        try:
            root.geometry('200x80+20+20')
            canvas = tk.Canvas(root, width=180, height=60, highlightthickness=0)
            canvas.pack()
            canvas.create_rectangle(0, 0, 20, 800, outline='')
            root.update()
            canvas.configure(scrollregion=canvas.bbox('all'))
            event = types.SimpleNamespace(
                delta=0,
                num=0,
                touchpad=True,
                delta_x=0,
                delta_y=-40,
                x_root=canvas.winfo_rootx() + 4,
                y_root=canvas.winfo_rooty() + 4,
            )
            before = canvas.yview()[0]
            _wheel.handle_canvas_scroll(canvas, event)
            self.assertGreater(canvas.yview()[0], before)
        finally:
            root.destroy()

    def test_binding_touchpad_sequence_does_not_raise(self):
        root = tk.Tk()
        try:
            root.withdraw()
            seen = []
            _wheel.bind_mousewheel(root.bind, seen.append)
            _wheel.un_bind_mousewheel(root.unbind)
        finally:
            root.destroy()
