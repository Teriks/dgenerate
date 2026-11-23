import platform
import unittest
import ctypes

from dgenerate.console import themechrome

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class ThemeChromeTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_mac_keeps_the_native_frame(self):
        from dgenerate.console import themechrome

        themechrome._CUSTOM = False
        try:
            window = tk.Toplevel(_ROOT)
            self.addCleanup(window.destroy)
        finally:
            themechrome._CUSTOM = platform.system() in ('Windows', 'Linux')
        self.assertFalse(window.overrideredirect())
        self.assertIsNone(getattr(window, '_chrome', None))

    def test_dialog_gets_a_themed_frame_and_client(self):
        if platform.system() not in ('Windows', 'Linux'):
            self.skipTest('custom frames are windows and linux only')
        from dgenerate.console import themechrome

        window = tk.Toplevel(_ROOT)
        self.addCleanup(window.destroy)
        window.title('Insert Recipe')
        body = tk.Frame(window)
        window.rowconfigure(0, weight=1)
        window.update_idletasks()

        self.assertFalse(window.overrideredirect())
        self.assertIs(body.master, window._chrome_client)
        self.assertEqual(window._chrome['text'].cget('text'), 'Insert Recipe')
        buttons = window._chrome['buttons']
        family = themechrome._caption_family()
        self.assertNotIn('min', buttons)
        self.assertNotIn('max', buttons)
        if family:
            import tkinter.font as tkfont
            self.assertEqual(
                tkfont.Font(font=buttons['close'].cget('font')).actual('family'), family)
            self.assertIn('-10', str(buttons['close'].cget('font')))
        else:
            self.assertIsNotNone(buttons['close'].cget('image'))
            self.assertEqual(buttons['close'].winfo_reqwidth(), themechrome._CAPTION_BUTTON)
        self.assertEqual(int(window._chrome['north'].cget('height')), 1)
        self.assertEqual(int(window._chrome['west'].cget('width')), 1)
        window.configure(padx=5, pady=8)
        self.assertEqual(int(str(window._chrome_client.cget('padx')).split()[0]), 5)
        self.assertEqual(int(str(window._chrome_client.cget('pady')).split()[0]), 8)
        self.assertEqual(int(str(window.cget('padx')).split()[0]), 0)

        window.deiconify()
        window.geometry('180x100+0+0')
        window.update_idletasks()
        window._chrome['placed'] = False
        _ROOT.geometry('600x400+200+160')
        _ROOT.update_idletasks()
        themechrome._center_if_default(window)
        window.update_idletasks()
        self.assertGreater(window.winfo_x(), 0)
        self.assertGreater(window.winfo_y(), 0)
        pad = window._chrome['icon'].pack_info()['padx']
        left = pad[0] if isinstance(pad, tuple) else pad
        self.assertEqual(int(left), 8)
        self.assertEqual(themechrome._hwnd(window) is not None or platform.system() != 'Windows', True)

        self.assertEqual(
            themechrome._move_hit(themechrome._HTCAPTION, False),
            themechrome._HTCLIENT)
        self.assertEqual(themechrome._move_hit(themechrome._HTTOP, False), themechrome._HTTOP)
        self.assertEqual(
            themechrome._resize_command(themechrome._HTBOTTOMLEFT),
            themechrome._SC_SIZE | themechrome._WMSZ_BOTTOMLEFT)
        self.assertEqual(
            themechrome._resize_command(themechrome._HTBOTTOMRIGHT),
            themechrome._SC_SIZE | themechrome._WMSZ_BOTTOMRIGHT)
        self.assertEqual(
            themechrome._resize_command(themechrome._HTBOTTOM),
            themechrome._SC_SIZE | themechrome._WMSZ_BOTTOM)
        self.assertLess(
            themechrome._resize_command(themechrome._HTBOTTOMRIGHT) & 0xFFF0,
            themechrome._SC_MOVE)
        self.assertIsNone(themechrome._resize_command(themechrome._HTCAPTION))

        window.resizable(False, False)
        self.assertEqual(window._chrome['resize'], (0, 0))
        if platform.system() == 'Windows':
            self.assertEqual(window._chrome['north'].winfo_manager(), '')
        else:
            self.assertEqual(window._chrome['north'].winfo_manager(), 'pack')

        window.geometry('220x140+30+40')
        window.update_idletasks()

        class _Event:
            pass

        event = _Event()
        event.x_root = window.winfo_x() + 8
        event.y_root = window.winfo_y() + 8
        if platform.system() == 'Windows':
            themechrome._press_drag(window, event)
            self.assertIsNone(window._chrome.get('drag'))
        else:
            themechrome._press_drag(window, event)
            event.x_root += 15
            event.y_root += 10
            themechrome._drag(window, event)
            window.update_idletasks()
            self.assertEqual(window.winfo_x(), 45)
            self.assertEqual(window.winfo_y(), 50)
        themechrome._release_pointer(window)

    def test_opening_height_is_not_padded_by_the_stock_caption(self):
        if platform.system() != 'Windows':
            self.skipTest('caption padding is windows only')
        window = tk.Toplevel(_ROOT)
        self.addCleanup(window.destroy)
        window.withdraw()
        tk.Label(window, text='one line').pack(anchor='w', padx=8, pady=8)
        window.update_idletasks()
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        window.minsize(width, height)
        window.geometry(f'{width}x{height}+40+40')
        window.deiconify()
        window.update()
        self.assertEqual(window.winfo_height(), height)

        window.geometry(f'{width}x{height + 40}+40+40')
        window.update()
        self.assertEqual(window.winfo_height(), height + 40)

        plain = tk.Toplevel(_ROOT)
        self.addCleanup(plain.destroy)
        plain.withdraw()
        tk.Label(plain, text='plain').pack()
        plain.update_idletasks()
        plain_height = plain.winfo_reqheight()
        plain.geometry(f'{plain.winfo_reqwidth()}x{plain_height}+80+80')
        plain.deiconify()
        plain.update()
        self.assertEqual(plain.winfo_height(), plain_height)

        owned = tk.Toplevel(_ROOT)
        self.addCleanup(owned.destroy)
        owned.withdraw()
        owned.resizable(False, False)
        owned.transient(_ROOT)
        tk.Label(owned, text='owned').pack(padx=8, pady=8)
        owned.update_idletasks()
        owned_height = owned.winfo_reqheight()
        owned.geometry(f'+120+120')
        owned.deiconify()
        owned.update()
        self.assertEqual(owned.winfo_height(), owned_height)

    def test_windows_keeps_a_native_resize_border(self):
        if platform.system() != 'Windows':
            self.skipTest('native hit testing is windows only')
        window = tk.Toplevel(_ROOT)
        self.addCleanup(window.destroy)
        window.deiconify()
        window.geometry('240x180+120+80')
        window.update_idletasks()
        hook = window._size_hook
        self.assertIsNotNone(hook)
        self.assertFalse(window.overrideredirect())
        self.assertGreaterEqual(hook.band, 4)
        self.assertEqual(window.winfo_width(), 240)
        self.assertEqual(window.winfo_height(), 180)
        user32 = ctypes.windll.user32
        user32.GetWindowLongPtrW.restype = ctypes.c_void_p
        user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
        user32.SendMessageW.restype = ctypes.c_ssize_t
        user32.SendMessageW.argtypes = (
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t)
        style = int(user32.GetWindowLongPtrW(ctypes.c_void_p(hook.hwnd), -16) or 0)
        self.assertFalse(style & themechrome._WS_CHILD)
        self.assertTrue(style & themechrome._WS_THICKFRAME)

        class _Point(ctypes.Structure):
            _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

        outer = ctypes.c_void_p(hook.hwnd)
        window_rect = themechrome._Rect()
        client = themechrome._Rect()
        user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(themechrome._Rect))
        user32.GetClientRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(themechrome._Rect))
        user32.ClientToScreen.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Point))
        user32.GetWindowRect(outer, ctypes.byref(window_rect))
        user32.GetClientRect(outer, ctypes.byref(client))
        origin = _Point(0, 0)
        user32.ClientToScreen(outer, ctypes.byref(origin))
        inset = hook.band
        self.assertEqual(origin.x - window_rect.left, inset)
        self.assertEqual(origin.y - window_rect.top, 0)
        self.assertEqual(window_rect.right - (origin.x + client.right), inset)
        self.assertEqual(window_rect.bottom - (origin.y + client.bottom), inset)

        sized = themechrome._NcCalcSizeParams()
        sized.rgrc[0] = themechrome._Rect(0, 0, 240, 180)
        result = int(user32.SendMessageW(
            outer, themechrome._WM_NCCALCSIZE, 1, ctypes.addressof(sized)))
        self.assertEqual(result, 0)
        self.assertEqual(
            (sized.rgrc[0].left, sized.rgrc[0].top, sized.rgrc[0].right, sized.rgrc[0].bottom),
            (inset, 0, 240 - inset, 180 - inset))

        def hit(x, y, target=None):
            packed = (int(x) & 0xFFFF) | ((int(y) & 0xFFFF) << 16)
            return int(user32.SendMessageW(
                target or outer, themechrome._WM_NCHITTEST, 0, packed))

        left, top, right, bottom = themechrome._window_rect(hook.hwnd)
        self.assertEqual(hit(left, top + 80), themechrome._HTLEFT)
        self.assertEqual(hit(right - 1, top + 80), themechrome._HTRIGHT)
        self.assertEqual(hit((left + right) // 2, bottom - 1), themechrome._HTBOTTOM)
        self.assertEqual(hit(left + inset + 24, top), themechrome._HTTOP)
        self.assertEqual(hit(left + inset + 24, top + inset + 6), themechrome._HTCLIENT)
        hook.buttons = ((right - 36, top + 2, right - inset - 2, top + 24),)
        self.assertEqual(hit(right - 20, top + 10), themechrome._HTCLIENT)
        title = window._chrome['title']
        title_x = title.winfo_rootx() + 30
        title_y = title.winfo_rooty() + max(title.winfo_height() - 4, inset + 2)
        self.assertEqual(
            hit(title_x, title_y, ctypes.c_void_p(int(title.winfo_id()))),
            themechrome._HTCLIENT)
        limits = themechrome._MinMaxInfo()
        user32.SendMessageW(
            outer, themechrome._WM_GETMINMAXINFO, 0, ctypes.addressof(limits))
        self.assertGreaterEqual(limits.ptMaxTrackSize.x, themechrome._TRACK_LIMIT)
        self.assertGreaterEqual(limits.ptMaxTrackSize.y, themechrome._TRACK_LIMIT)

    def test_a_bare_override_window_drops_the_frame(self):
        if platform.system() not in ('Windows', 'Linux'):
            self.skipTest('custom frames are windows and linux only')
        tip = tk.Toplevel(_ROOT)
        self.addCleanup(tip.destroy)
        self.assertIsNotNone(tip._chrome)
        tip.overrideredirect(True)
        self.assertIsNone(tip._chrome)
        themechrome._pointer_motion(tip, type('Event', (), {'x_root': 1, 'y_root': 1})())
        tip.deiconify()
        tip.update()
        tip.event_generate('<B1-Motion>', x=3, y=3)
        label = tk.Label(tip, text='tip')
        self.assertIs(label.master, tip)

    def test_main_window_requests_a_taskbar_button(self):
        if platform.system() != 'Windows':
            self.skipTest('taskbar style is windows only')
        import ctypes
        from dgenerate.console import themechrome

        app = tk.Tk()
        self.addCleanup(app.destroy)
        app.withdraw()
        app.geometry('240x120+80+80')
        themechrome.install(app)
        self.assertIn('min', app._chrome['buttons'])
        self.assertIn('max', app._chrome['buttons'])
        themechrome._apply_taskbar_style(app)
        user32 = ctypes.windll.user32
        user32.GetWindowLongPtrW.restype = ctypes.c_void_p
        user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
        hwnd = ctypes.c_void_p(themechrome._hwnd(app))
        exstyle = int(user32.GetWindowLongPtrW(hwnd, themechrome._GWL_EXSTYLE) or 0)
        self.assertTrue(exstyle & themechrome._WS_EX_APPWINDOW)
        self.assertFalse(exstyle & themechrome._WS_EX_TOOLWINDOW)

    def test_owned_dialog_minimizes_to_the_taskbar(self):
        if platform.system() != 'Windows':
            self.skipTest('owned-window minimize is windows only')
        app = tk.Toplevel(_ROOT)
        self.addCleanup(app.destroy)
        app.geometry('320x200+120+120')
        dialog = tk.Toplevel(app)
        self.addCleanup(dialog.destroy)
        dialog.transient(app)
        dialog.geometry('220x140+180+180')
        themechrome.install(app)
        themechrome.install(dialog)
        app.update_idletasks()
        dialog.update_idletasks()
        themechrome._apply_taskbar_style(dialog)
        dialog.update()
        hwnd = themechrome._hwnd(dialog)
        owner = themechrome._window_long(hwnd, themechrome._GWLP_HWNDPARENT)
        self.assertEqual(owner, themechrome._hwnd(app))
        themechrome._minimize(dialog)
        dialog.update()
        self.assertEqual(themechrome._hwnd(dialog), hwnd)
        user32 = ctypes.windll.user32
        user32.IsIconic.argtypes = (ctypes.c_void_p,)
        user32.IsIconic.restype = ctypes.c_int
        user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(themechrome._Rect))
        user32.GetWindowRect.restype = ctypes.c_int

        def iconic_rect():
            rect = themechrome._Rect()
            user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect))
            return rect

        def assert_taskbar_minimized():
            self.assertTrue(user32.IsIconic(ctypes.c_void_p(hwnd)))
            self.assertEqual(themechrome._window_long(hwnd, themechrome._GWLP_HWNDPARENT), 0)
            exstyle = themechrome._window_long(hwnd, themechrome._GWL_EXSTYLE)
            self.assertTrue(exstyle & themechrome._WS_EX_APPWINDOW)
            self.assertFalse(exstyle & themechrome._WS_EX_TOOLWINDOW)
            self.assertLess(iconic_rect().left, -1000)

        assert_taskbar_minimized()
        app.focus_force()
        app.update()
        dialog.update()
        assert_taskbar_minimized()
        themechrome._watch_minimized(dialog)
        themechrome._user32.ShowWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
        themechrome._user32.ShowWindow.restype = ctypes.c_int
        themechrome._user32.ShowWindow(ctypes.c_void_p(hwnd), 9)
        dialog.update()
        themechrome._watch_minimized(dialog)
        hwnd = themechrome._hwnd(dialog)
        self.assertFalse(user32.IsIconic(ctypes.c_void_p(hwnd)))
        self.assertEqual(
            themechrome._window_long(hwnd, themechrome._GWLP_HWNDPARENT),
            themechrome._hwnd(app))
        themechrome._minimize(dialog)
        dialog.update()
        self.assertEqual(themechrome._hwnd(dialog), hwnd)
        assert_taskbar_minimized()

    def test_managed_frame_gets_theme_chrome_and_gives_it_back(self):
        if platform.system() not in ('Windows', 'Linux'):
            self.skipTest('painted frames are windows and linux only')
        frame = tk.Frame(_ROOT)
        label = tk.Label(frame, text='log')
        label.pack(fill=tk.BOTH, expand=True)

        def drop():
            themechrome.release(frame)
            try:
                _ROOT.wm_forget(frame)
            except tk.TclError:
                pass
            try:
                frame.destroy()
            except tk.TclError:
                pass

        self.addCleanup(drop)
        _ROOT.update_idletasks()
        _ROOT.wm_manage(frame)
        frame.tk.call('wm', 'withdraw', frame._w)
        themechrome.adopt(frame)
        self.assertNotEqual(themechrome._hwnd(frame), themechrome._hwnd(_ROOT))
        self.assertIs(label.master, frame)
        self.assertEqual(str(label.pack_info()['in']), str(frame._chrome_client))
        if platform.system() == 'Windows':
            themechrome._apply_taskbar_style(frame)
            user32 = ctypes.windll.user32
            user32.GetWindowLongPtrW.restype = ctypes.c_void_p
            user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
            hwnd = ctypes.c_void_p(themechrome._hwnd(frame))
            exstyle = int(user32.GetWindowLongPtrW(hwnd, themechrome._GWL_EXSTYLE) or 0)
            self.assertTrue(exstyle & themechrome._WS_EX_APPWINDOW)
            self.assertFalse(exstyle & themechrome._WS_EX_TOOLWINDOW)
        themechrome.release(frame)
        self.assertIsNone(getattr(frame, '_chrome', None))
        self.assertEqual(str(label.pack_info()['in']), str(frame))

    def test_managed_frame_covers_the_native_caption(self):
        if platform.system() != 'Windows':
            self.skipTest('native caption covering is windows only')
        app = tk.Tk()
        self.addCleanup(app.destroy)
        app.withdraw()
        app.geometry('640x420+40+40')
        themechrome.install(app)
        frame = tk.Frame(app)
        label = tk.Label(frame, text='log')
        label.pack(fill=tk.BOTH, expand=True)

        def drop():
            themechrome.release(frame)
            try:
                app.wm_forget(frame)
            except tk.TclError:
                pass

        self.addCleanup(drop)
        app.update_idletasks()
        app.wm_manage(frame)
        frame.tk.call('wm', 'geometry', frame._w, '480x300+90+90')
        themechrome.adopt(frame)
        frame.update_idletasks()
        self.assertIs(frame.winfo_children()[-1], label)
        hook = getattr(frame, '_size_hook', None)
        self.assertIsNotNone(hook)
        self.assertNotEqual(hook.hwnd, themechrome._hwnd(app))
        user32 = ctypes.windll.user32

        class RECT(ctypes.Structure):
            _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                        ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

        window_rect, client_rect = RECT(), RECT()
        hwnd = ctypes.c_void_p(hook.hwnd)
        user32.GetWindowRect(hwnd, ctypes.byref(window_rect))
        user32.GetClientRect(hwnd, ctypes.byref(client_rect))
        window_height = window_rect.bottom - window_rect.top
        client_height = client_rect.bottom - client_rect.top
        self.assertGreater(window_height, 0)
        self.assertLess(window_height - client_height, 24)

    def test_nested_size_message_does_not_enter_python(self):
        if platform.system() != 'Windows' or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest('the subclass thunk is 64-bit windows only')
        address = themechrome._subclass_address()
        native = themechrome._NativeChrome()
        native.inset = 4
        native.top_inset = 1
        native.hwnd = 0x55
        rect = (ctypes.c_long * 4)(10, 20, 210, 180)
        prototype = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_ssize_t)
        call = prototype(address)
        themechrome._dispatch_flag.value = 1
        try:
            result = call(0x55, 0x83, 1, ctypes.addressof(rect), 1, ctypes.addressof(native))
        finally:
            themechrome._dispatch_flag.value = 0
        self.assertEqual(result, 0)
        self.assertEqual(list(rect), [14, 21, 206, 176])
