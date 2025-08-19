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

import os
import threading
import tkinter as tk
import typing
from typing import Optional, Callable

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.resources as _resources
import dgenerate.console.showindirectory as _showindirectory
from dgenerate.console.imageviewer import ImageViewer


class TearOffImagePane:
    def __init__(self,
                 master: tk.Widget,
                 on_error: Optional[Callable[[str], None]] = None,
                 on_info: Optional[Callable[[str], None]] = None,
                 get_cwd: Optional[Callable[[], str]] = None,
                 initial_window_geometry: str = '512x512',
                 window_title: str = 'Image Preview'):
        self._parent_container = master
        self._on_error = on_error or (lambda s: None)
        self._on_info = on_info or (lambda s: None)
        self._get_working_directory = get_cwd or (lambda: os.getcwd())
        self._initial_window_geometry = initial_window_geometry
        self._window_title = window_title

        self._pane_visible = tk.BooleanVar(value=False)
        self._window_visible = tk.BooleanVar(value=False)

        self._pane_frame = None
        self._pane_viewer = None
        self._pane_context_menu = None

        self._window = None
        self._window_viewer = None
        self._window_context_menu = None
        self._window_last_pos = None
        self._race_lock = threading.RLock()

        self._last_right_clicked_coords = None

        def _update_pane_visibility(*args):
            master.after('idle', self._update_pane_visibility)

        def _update_window_visibility(*args):
            master.after('idle', self._update_window_visibility)

        self._pane_visible.trace_add('write', _update_pane_visibility)
        self._window_visible.trace_add('write', _update_window_visibility)

        self._create_pane()

    def _create_pane(self):
        self._pane_frame = tk.Frame(self._parent_container, bg='black')
        self._pane_viewer = ImageViewer(self._pane_frame, bg='black')
        self._pane_viewer.pack(fill=tk.BOTH, expand=True)

        self._pane_viewer.on_error = lambda s: self._on_error(s + '\n')
        self._pane_viewer.on_info = lambda s: self._on_info(s + '\n')

        self._pane_context_menu = self._create_context_menu(self._pane_frame, self._pane_viewer, 'pane')

        self._pane_viewer.bind_event(
            '<Button-3>',
            lambda e: self._show_context_menu(e, self._pane_viewer, self._pane_context_menu)
        )

    def _create_window(self, initial_image_path=None, initial_view_state=None):
        root = self._parent_container.winfo_toplevel()
        self._window = tk.Toplevel(root)

        geometry = self._initial_window_geometry
        if self._window_last_pos is not None:
            geometry += '+{}+{}'.format(*self._window_last_pos)
        self._window.geometry(geometry)
        self._window.title(self._window_title)

        window_frame = tk.Frame(self._window, bg='black')
        window_frame.pack(fill=tk.BOTH, expand=True)

        self._window_viewer = ImageViewer(window_frame, bg='black')
        self._window_viewer.pack(fill=tk.BOTH, expand=True)

        self._window_viewer.on_error = lambda s: self._on_error(s + '\n')
        self._window_viewer.on_info = lambda s: self._on_info(s + '\n')

        self._window.update_idletasks()

        if initial_image_path is not None:
            self._window_viewer.load_image(
                initial_image_path,
                fit=False,
                view_state=initial_view_state
            )

        self._window.lift()
        self._window.focus_force()

        self._window_context_menu = self._create_context_menu(self._window, self._window_viewer, 'window')

        self._window_viewer.bind_event(
            '<Button-3>',
            lambda e: self._show_context_menu(e, self._window_viewer, self._window_context_menu)
        )

        def on_closing():
            self._window_visible.set(False)
            self._window.withdraw()

        self._window.protocol('WM_DELETE_WINDOW', on_closing)

    def _update_pane_visibility(self):
        if not self._pane_visible.get():
            if self._pane_frame and self._pane_frame.winfo_manager():
                self._parent_container.remove(self._pane_frame)
        else:
            if self._window_visible.get():
                self._window_visible.set(False)

            current_image_path = None
            current_view_state = None

            if self._window is not None and self._window_viewer is not None:
                current_image_path = self._window_viewer.get_image_path()

                if current_image_path == self._pane_viewer.get_image_path():
                    current_view_state = self._window_viewer.get_view_state()

            self._parent_container.add(self._pane_frame)

            if current_image_path is not None:
                self._pane_viewer.load_image(
                    current_image_path,
                    fit=False,
                    view_state=current_view_state
                )
                self._pane_viewer.update_idletasks()
                self._pane_viewer.update()

    def _update_window_visibility(self):
        if not self._window_visible.get():
            if self._window is not None:
                self._window.withdraw()
        else:
            if self._pane_visible.get():
                self._pane_visible.set(False)

            current_image_path = self._pane_viewer.get_image_path()
            current_view_state = self._pane_viewer.get_view_state()

            if self._window is not None:
                if current_image_path != self._window_viewer.get_image_path():
                    current_view_state = None

                self._window.deiconify()
                self._window.update_idletasks()
                self._window.lift()
                self._window.focus_force()

                if current_image_path is not None:
                    self._window_viewer.load_image(
                        current_image_path,
                        fit=False,
                        view_state=current_view_state
                    )
                return

            self._create_window(current_image_path, current_view_state)

    def _create_context_menu(self,
                             master: tk.Misc,
                             image_viewer: ImageViewer,
                             menu_type: typing.Literal['pane', 'window'] = 'pane') -> tk.Menu:
        context_menu = tk.Menu(master, tearoff=0)

        self._install_common_context_options(context_menu, image_viewer)

        if _showindirectory.is_supported():
            def open_in_directory():
                file_path = image_viewer.get_image_path()
                if file_path:
                    _showindirectory.show_in_directory(file_path)

            context_menu.add_command(label='Show In Directory', command=open_in_directory)

            original_popup = context_menu.tk_popup

            def patched_popup(*args, **kwargs):
                path = image_viewer.get_image_path()
                if path is not None and os.path.exists(path):
                    context_menu.entryconfigure('Show In Directory', state=tk.NORMAL)
                else:
                    context_menu.entryconfigure('Show In Directory', state=tk.DISABLED)
                original_popup(*args, **kwargs)

            context_menu.tk_popup = patched_popup
            context_menu.add_separator()

        if menu_type == 'pane':
            context_menu.add_command(
                label='Hide Image Pane',
                command=lambda: self._pane_visible.set(False))

            context_menu.add_command(
                label='Make Window',
                command=lambda: self._window_visible.set(True))
        elif menu_type == 'window':
            context_menu.add_command(
                label='Make Pane',
                command=lambda: self._pane_visible.set(True))

        context_menu.add_separator()

        context_menu.add_command(
            label='Help',
            command=lambda: image_viewer.request_help()
        )

        return context_menu

    def _install_common_context_options(self, context_menu: tk.Menu, image_viewer: ImageViewer):
        context_menu.add_command(
            label='Copy Coordinates "x"',
            command=lambda: self._copy_coordinates_from_menu('x', image_viewer))

        context_menu.add_command(
            label='Copy Coordinates CSV',
            command=lambda: self._copy_coordinates_from_menu(',', image_viewer))

        context_menu.add_separator()

        context_menu.add_command(
            label='Copy Bounding Box "x"',
            command=lambda: self._start_bbox_selection_from_menu('x', image_viewer)
        )

        context_menu.add_command(
            label='Copy Bounding Box CSV',
            command=lambda: self._start_bbox_selection_from_menu(',', image_viewer)
        )

        context_menu.add_separator()

        context_menu.add_command(
            label='View Actual Size',
            command=lambda: self._reset_image_view(image_viewer))

        context_menu.add_command(
            label='Zoom to Fit',
            command=lambda: self._zoom_image_to_fit(image_viewer))

        context_menu.add_separator()

        context_menu.add_command(
            label='Load Image',
            command=self._load_image_manually)

        context_menu.add_command(
            label='Copy Path',
            command=lambda: self._copy_image_path_from_menu(image_viewer))

    def _show_context_menu(self, event, image_viewer: ImageViewer, context_menu: tk.Menu):
        image_x, image_y = image_viewer.get_coordinates_at_cursor(event.x, event.y)
        if image_x is not None and image_y is not None:
            self._last_right_clicked_coords = (image_x, image_y)
        else:
            self._last_right_clicked_coords = None

        context_menu.entryconfigure('Load Image', state=tk.NORMAL)

        if self._last_right_clicked_coords is not None:
            context_menu.entryconfigure('Copy Coordinates "x"', state=tk.NORMAL)
            context_menu.entryconfigure('Copy Coordinates CSV', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Coordinates "x"', state=tk.DISABLED)
            context_menu.entryconfigure('Copy Coordinates CSV', state=tk.DISABLED)

        if image_viewer.has_image():
            context_menu.entryconfigure('Copy Bounding Box "x"', state=tk.NORMAL)
            context_menu.entryconfigure('Copy Bounding Box CSV', state=tk.NORMAL)
            context_menu.entryconfigure('View Actual Size', state=tk.NORMAL)
            context_menu.entryconfigure('Zoom to Fit', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Bounding Box "x"', state=tk.DISABLED)
            context_menu.entryconfigure('Copy Bounding Box CSV', state=tk.DISABLED)
            context_menu.entryconfigure('View Actual Size', state=tk.DISABLED)
            context_menu.entryconfigure('Zoom to Fit', state=tk.DISABLED)

        if image_viewer.get_image_path() is not None:
            context_menu.entryconfigure('Copy Path', state=tk.NORMAL)
        else:
            context_menu.entryconfigure('Copy Path', state=tk.DISABLED)

        root = self._parent_container.winfo_toplevel()
        context_menu.tk_popup(root.winfo_pointerx(), root.winfo_pointery())

    def _copy_coordinates_from_menu(self, separator, image_viewer):
        if self._last_right_clicked_coords is None:
            return

        x, y = self._last_right_clicked_coords
        coordinate_text = f"{x}{separator}{y}"

        try:
            root = self._parent_container.winfo_toplevel()
            root.clipboard_clear()
            root.clipboard_append(coordinate_text)
            self._on_info(f"Coordinates copied to clipboard: {coordinate_text}\n")
        except Exception as e:
            self._on_error(f"Failed to copy coordinates to clipboard: {e}\n")

    def _start_bbox_selection_from_menu(self, separator, image_viewer):
        image_viewer.start_bbox_selection(separator)

    def _reset_image_view(self, image_viewer):
        image_viewer.reset_view()

    def _zoom_image_to_fit(self, image_viewer):
        image_viewer.zoom_to_fit()

    def _copy_image_path_from_menu(self, image_viewer):
        image_viewer.copy_path()

    def _load_image_manually(self):
        f = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']),
            initialdir=self._get_working_directory())

        if f is None:
            return

        try:
            self.load_image(f)
            self._on_info(f"Manually loaded image: {f}\n")
        except Exception as e:
            self._on_error(f"Failed to load image: {e}\n")

    def load_image(self, image_path: str, fit: bool = True):
        try:
            self._pane_viewer.load_image(image_path, fit=fit)

            if self._window is not None and self._window_viewer is not None:
                self._window_viewer.load_image(image_path, fit=fit)
        except Exception as e:
            self._on_error(f"Error loading image: {e}\n")

    def set_pane_visible(self, visible: bool):
        self._pane_visible.set(visible)

    def set_window_visible(self, visible: bool):
        self._window_visible.set(visible)

    def is_pane_visible(self) -> bool:
        return self._pane_visible.get()

    def is_window_visible(self) -> bool:
        return self._window_visible.get()

    def get_pane_visible_var(self) -> tk.BooleanVar:
        return self._pane_visible

    def get_window_visible_var(self) -> tk.BooleanVar:
        return self._window_visible

    def get_current_image_path(self) -> Optional[str]:
        if self._pane_visible.get():
            return self._pane_viewer.get_image_path()
        elif self._window_visible.get() and self._window_viewer is not None:
            return self._window_viewer.get_image_path()
        else:
            return self._pane_viewer.get_image_path()

    def cleanup(self):
        if self._pane_viewer is not None:
            self._pane_viewer.cleanup()

        if self._window_viewer is not None:
            self._window_viewer.cleanup()

        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
