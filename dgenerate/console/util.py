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
