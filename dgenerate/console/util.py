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

import inspect


def position_toplevel(master,
                      toplevel: tk.Toplevel,
                      size: tuple[int, int] | None = None,
                      position: tuple[int, int] | None = None):
    """
    Position a top level tkinter window on a parent window.

    If no position is provided, the window will be centered on the parent

    :param master: the parent window
    :param toplevel: the toplevel
    :param size: optional toplevel size
    :param position: optional toplevel position
    :return:
    """

    toplevel.withdraw()

    master.update_idletasks()
    toplevel.update_idletasks()

    prefix = '' if size is None else f'{size[0]}x{size[1]}'

    if position is None:
        window_width = master.winfo_width()
        window_height = master.winfo_height()
        top_level_width = toplevel.winfo_reqwidth() if not size else size[0]
        top_level_height = toplevel.winfo_reqheight() if not size else size[1]

        position_top = master.winfo_y() + (window_height // 2) - (top_level_height // 2)
        position_left = master.winfo_x() + (window_width // 2) - (top_level_width // 2)

        toplevel.geometry(f"{prefix}+{position_left}+{position_top}")
    else:
        toplevel.geometry(f"{prefix}+{position[0]}+{position[1]}")

    toplevel.deiconify()


class DialogState:
    """State class for managing singleton dialog window state"""

    def __init__(self, save_position: bool = True, save_size: bool = False):
        self.save_position = save_position
        self.save_size = save_size
        self.last_pos = None
        self.last_size = None
        self.cur_window = None


def create_singleton_dialog(
        master: tk.Tk | tk.Toplevel,
        dialog_class: type[tk.Toplevel],
        state: DialogState,
        dialog_kwargs: dict = None
) -> tk.Toplevel:
    """Create a singleton dialog window that saves its position and size.

    This function ensures only one instance of a dialog window exists at a time.
    If the dialog is already open, it will be brought to the front and focused instead of creating
    a new instance.

    :param master: The parent window (Tk or Toplevel instance)
    :param dialog_class: The dialog class to instantiate (must inherit from Toplevel)
    :param state: The dialog state to use for position/size persistence
    :param dialog_kwargs: Additional keyword arguments to pass to the dialog constructor
    :type master: Union[tk.Tk, tk.Toplevel]
    :type dialog_class: Type[tk.Toplevel]
    :type state: DialogState
    :type dialog_kwargs: Optional[Dict[str, Any]]
    :return: The dialog window instance
    :rtype: tk.Toplevel
    :raises RuntimeError: If the dialog class doesn't support size or position parameters when they are enabled in state
    """
    if state.cur_window is not None:
        state.cur_window.lift()
        state.cur_window.focus_set()
        return state.cur_window

    if dialog_kwargs is None:
        dialog_kwargs = {}

    sig = inspect.signature(dialog_class.__init__)
    params = sig.parameters

    if state.save_position and state.last_pos is not None:
        if 'position' not in params:
            raise RuntimeError(f"Dialog class {dialog_class.__name__} does not support position parameter")
        dialog_kwargs['position'] = state.last_pos

    if state.save_size and state.last_size is not None:
        if 'size' not in params:
            raise RuntimeError(f"Dialog class {dialog_class.__name__} does not support size parameter")
        dialog_kwargs['size'] = state.last_size

    dialog = dialog_class(master=master, **dialog_kwargs)
    state.cur_window = dialog

    og_destroy = dialog.destroy

    def destroy():
        if state.save_position:
            state.last_pos = (dialog.winfo_x(), dialog.winfo_y())
        if state.save_size:
            state.last_size = (dialog.winfo_width(), dialog.winfo_height())
        state.cur_window = None
        og_destroy()

    dialog.destroy = destroy

    def protocol_delete():
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", protocol_delete)
    return dialog
