import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None


class ThemeMenuTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def test_apply_colors_menus_created_before_and_after(self):
        from dgenerate.console.thememenu import ThemeMenu, apply

        early = ThemeMenu(_ROOT, tearoff=0)
        early.add_command(label='File')
        early.add_checkbutton(label='Wrap Text')
        self.addCleanup(early.destroy)

        apply(_ROOT, '#282a36', '#f8f8f2', '#44475a', '#f8f8f2')
        self.assertEqual(str(early.cget('background')), '#282a36')
        self.assertEqual(str(early.cget('foreground')), '#f8f8f2')
        self.assertEqual(str(early.cget('activebackground')), '#44475a')
        self.assertEqual(str(early.cget('activeforeground')), '#f8f8f2')

        late = ThemeMenu(_ROOT, tearoff=0)
        self.addCleanup(late.destroy)
        self.assertEqual(str(late.cget('background')), '#282a36')
        self.assertNotEqual(str(late.cget('disabledforeground')), '#f8f8f2')

    def test_check_radio_and_relabel(self):
        from dgenerate.console.thememenu import ThemeMenu

        menu = ThemeMenu(_ROOT, tearoff=0)
        self.addCleanup(menu.destroy)
        flag = tk.BooleanVar(master=_ROOT, value=False)
        choice = tk.StringVar(master=_ROOT, value='none')
        menu.add_checkbutton(label='Wrap Text', variable=flag)
        menu.add_radiobutton(label='Dracula', variable=choice, value='dracula')
        menu.add_command(label='Save')
        nested = ThemeMenu(menu, tearoff=0)
        self.addCleanup(nested.destroy)
        menu.add_cascade(label='More', menu=nested)

        menu._entries[0].invoke()
        self.assertTrue(flag.get())
        menu.entryconfigure('Wrap Text', state=tk.DISABLED)
        menu._entries[0].invoke()
        self.assertTrue(flag.get())
        menu._entries[1].invoke()
        self.assertEqual(choice.get(), 'dracula')
        menu.entryconfigure(menu.index('end') - 1, label='Save As')
        self.assertEqual(menu._entries[2].label, 'Save As')
        self.assertIs(nested._toplevel(), _ROOT)

    def test_hover_after_menu_closes_does_not_open_a_child(self):
        from dgenerate.console.thememenu import ThemeMenu

        menu = ThemeMenu(_ROOT, tearoff=0)
        self.addCleanup(menu.destroy)
        child = ThemeMenu(menu, tearoff=0)
        self.addCleanup(child.destroy)
        child.add_command(label='Recipe')
        menu.add_cascade(label='Insert Code', menu=child)
        menu._hover(menu._entries[0])
        self.assertEqual(menu._popup, None)

    def test_hover_switches_top_level_menu(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import MenuBar, ThemeMenu

        bar = MenuBar(_ROOT)
        self.addCleanup(bar.destroy)
        file_menu = ThemeMenu(bar, tearoff=0)
        edit_menu = ThemeMenu(bar, tearoff=0)
        self.addCleanup(file_menu.destroy)
        self.addCleanup(edit_menu.destroy)
        file_menu.add_command(label='New')
        edit_menu.add_command(label='Cut')
        nested = ThemeMenu(edit_menu, tearoff=0)
        self.addCleanup(nested.destroy)
        nested.add_command(label='Find')
        edit_menu.add_cascade(label='Find / Replace', menu=nested)
        bar.add_cascade(label='File', menu=file_menu)
        bar.add_cascade(label='Edit', menu=edit_menu)
        bar.pack(side=tk.TOP, fill=tk.X)
        _ROOT.update_idletasks()

        file_button, edit_button = bar._slots[0][0], bar._slots[1][0]
        bar._open(file_button, file_menu)
        _ROOT.update()
        self.assertIn(file_menu, thememenu._open)
        bar._hover(edit_button, edit_menu)
        _ROOT.update()
        self.assertIn(edit_menu, thememenu._open)
        self.assertNotIn(file_menu, thememenu._open)
        self.assertIs(edit_menu._popup, _ROOT.grab_current())
        thememenu._close_all()

    def test_foreground_watch_closes_when_the_window_is_not_in_front(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import MenuBar, ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        bar = MenuBar(parent)
        menu = ThemeMenu(bar, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='New')
        bar.add_cascade(label='File', menu=menu)
        bar.pack(side=tk.TOP, fill=tk.X)
        parent.update_idletasks()
        bar._open(bar._slots[0][0], menu)
        parent.update()
        self.assertIn(menu, thememenu._open)
        job = thememenu._foreground_job
        original = thememenu._foreground
        thememenu._foreground = lambda widget: False
        try:
            thememenu._close_if_background()
        finally:
            thememenu._foreground = original
            if job is not None:
                parent.after_cancel(job)
                if thememenu._foreground_job == job:
                    thememenu._foreground_job = None
        self.assertNotIn(menu, thememenu._open)

    def test_menu_stays_open_while_its_window_is_still_active(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import MenuBar, ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        bar = MenuBar(parent)
        menu = ThemeMenu(bar, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='New')
        bar.add_cascade(label='File', menu=menu)
        bar.pack(side=tk.TOP, fill=tk.X)
        parent.update_idletasks()
        bar._open(bar._slots[0][0], menu)
        parent.update()
        self.assertIn(menu, thememenu._open)
        original_foreground = thememenu._foreground
        original_active = thememenu._net_active_window
        thememenu._foreground = thememenu._linux_foreground
        thememenu._net_active_window = lambda: int(parent.winfo_id())
        try:
            thememenu._close_if_background()
            self.assertIn(menu, thememenu._open)
            thememenu._net_active_window = lambda: 0
            thememenu._close_if_background()
            self.assertIn(menu, thememenu._open)
        finally:
            thememenu._foreground = original_foreground
            thememenu._net_active_window = original_active
            thememenu._close_all()

    def test_unknown_active_window_keeps_an_unfocused_menu(self):
        from dgenerate.console import thememenu

        class Elsewhere:
            def winfo_toplevel(self):
                return self

        class Idle:
            def focus_get(self):
                return None

            def winfo_toplevel(self):
                return self

        class Busy:
            def focus_get(self):
                return Elsewhere()

            def winfo_toplevel(self):
                return self

        original = thememenu._net_active_window
        thememenu._net_active_window = lambda: None
        try:
            self.assertTrue(thememenu._linux_foreground(Idle()))
            self.assertFalse(thememenu._linux_foreground(Busy()))
        finally:
            thememenu._net_active_window = original

    def test_menu_closes_when_its_window_is_unmapped(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import MenuBar, ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        bar = MenuBar(parent)
        menu = ThemeMenu(bar, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='New')
        bar.add_cascade(label='File', menu=menu)
        bar.pack(side=tk.TOP, fill=tk.X)
        parent.update_idletasks()
        bar._open(bar._slots[0][0], menu)
        parent.update()
        self.assertIn(menu, thememenu._open)
        parent.event_generate('<Unmap>')
        self.assertNotIn(menu, thememenu._open)

    def test_right_click_outside_closes_the_menu(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import MenuBar, ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        bar = MenuBar(parent)
        menu = ThemeMenu(bar, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='New')
        bar.add_cascade(label='File', menu=menu)
        bar.pack(side=tk.TOP, fill=tk.X)
        body = tk.Frame(parent, width=200, height=120)
        body.pack()
        parent.update_idletasks()
        bar._open(bar._slots[0][0], menu)
        parent.update()
        self.assertIn(menu, thememenu._open)
        body.event_generate('<Button-3>', x=20, y=20)
        parent.update()
        self.assertNotIn(menu, thememenu._open)

    def test_right_click_that_opens_a_context_menu_does_not_close_it(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        parent.geometry('320x200+40+40')
        body = tk.Text(parent, width=20, height=5)
        body.pack()
        outside = tk.Frame(parent, width=40, height=20)
        outside.pack()
        menu = ThemeMenu(body, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='Copy')
        body.bind('<Button-3>', lambda event: menu.tk_popup(event.x_root, event.y_root))
        parent.deiconify()
        parent.update()
        body.event_generate('<Button-3>', x=8, y=8)
        self.assertIn(menu, thememenu._open)
        parent.update()
        outside.event_generate('<Button-3>', x=4, y=4)
        parent.update()
        self.assertNotIn(menu, thememenu._open)

    def test_clicking_the_window_frame_closes_a_context_menu(self):
        import platform

        if platform.system() not in ('Windows', 'Linux'):
            self.skipTest('custom frames are windows and linux only')

        from dgenerate.console import themechrome
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        parent.geometry('320x200+80+80')
        parent.deiconify()
        parent.update()
        menu = ThemeMenu(parent, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='Copy')
        title = parent._chrome['text']
        menu.tk_popup(title.winfo_rootx(), title.winfo_rooty() + 80)
        parent.update()
        self.assertIn(menu, thememenu._open)
        if platform.system() == 'Windows':
            import ctypes
            user32 = ctypes.windll.user32
            user32.SendMessageW.restype = ctypes.c_ssize_t
            user32.SendMessageW.argtypes = (
                ctypes.c_void_p, ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t)
            hook = parent._size_hook
            left, top, right, bottom = themechrome._window_rect(hook.hwnd)
            packed = ((left + hook.band + 30) & 0xFFFF) | ((top + 8) & 0xFFFF) << 16
            self.assertEqual(
                int(user32.SendMessageW(
                    ctypes.c_void_p(hook.hwnd), themechrome._WM_NCHITTEST, 0, packed)),
                themechrome._HTCLIENT)
        title.event_generate('<Button-1>', x=4, y=4)
        parent.update()
        self.assertNotIn(menu, thememenu._open)

        menu.tk_popup(title.winfo_rootx(), title.winfo_rooty() + 80)
        parent.update()
        close = parent._chrome['buttons']['close']
        close.event_generate('<Button-1>', x=2, y=2)
        close.event_generate('<ButtonRelease-1>', x=2, y=2)
        parent.update()
        self.assertTrue(parent.winfo_exists())
        self.assertNotIn(menu, thememenu._open)
        if platform.system() == 'Windows':
            left, top, right, bottom = themechrome._window_rect(hook.hwnd)
            packed = ((left + hook.band + 30) & 0xFFFF) | ((top + 8) & 0xFFFF) << 16
            self.assertEqual(
                int(user32.SendMessageW(
                    ctypes.c_void_p(hook.hwnd), themechrome._WM_NCHITTEST, 0, packed)),
                themechrome._HTCLIENT if themechrome._remote_session() else themechrome._HTCAPTION)

    def test_context_menu_highlights_and_keeps_its_grab(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        parent.geometry('320x200+40+40')
        entry = tk.Entry(parent)
        entry.pack()
        menu = ThemeMenu(entry, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='Copy')
        menu.add_command(label='Paste')
        parent.deiconify()
        parent.update()
        menu.tk_popup(80, 80)
        parent.update()
        self.assertIn(menu, thememenu._open)
        menu.grab_release()
        self.assertIn(menu, thememenu._open)
        self.assertIs(menu._popup, parent.grab_current())
        row = menu._entries[1].row
        thememenu._track(
            row.winfo_rootx() + row.winfo_width() // 2,
            row.winfo_rooty() + row.winfo_height() // 2)
        self.assertIs(menu._active, menu._entries[1])
        row = menu._entries[0].row
        menu._popup._menu_origin = (5000, 5000)
        thememenu._track(5000 + row.winfo_x() + 2, 5000 + row.winfo_y() + 2)
        self.assertIs(menu._active, menu._entries[0])
        thememenu._close_all()

    def test_destroyed_owner_closes_the_context_menu(self):
        from dgenerate.console import thememenu
        from dgenerate.console.thememenu import ThemeMenu

        parent = tk.Toplevel(_ROOT)
        self.addCleanup(parent.destroy)
        entry = tk.Entry(parent)
        entry.pack()
        menu = ThemeMenu(entry, tearoff=0)
        self.addCleanup(menu.destroy)
        menu.add_command(label='Copy')
        parent.update_idletasks()
        menu.tk_popup(80, 80)
        parent.update()
        self.assertIn(menu, thememenu._open)
        entry.destroy()
        parent.update()
        self.assertNotIn(menu, thememenu._open)
        thememenu._close_if_background()
        thememenu._on_motion(type('Event', (), {'x_root': 0, 'y_root': 0})())

    def test_menu_bar_is_native_on_macos_only(self):
        import platform

        from dgenerate.console import thememenu

        self.assertFalse(issubclass(thememenu.ThemeMenu, tk.Menu))
        if platform.system() in ('Windows', 'Linux'):
            self.assertTrue(thememenu._CUSTOM)
            self.assertTrue(issubclass(thememenu.MenuBar, tk.Frame))
        else:
            self.assertFalse(thememenu._CUSTOM)
            self.assertTrue(issubclass(thememenu.MenuBar, tk.Menu))
