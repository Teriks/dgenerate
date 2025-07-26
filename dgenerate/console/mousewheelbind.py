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


def bind_mousewheel(bind_func, callback, modifier=None):
    if modifier is None:
        bind_func("<MouseWheel>", callback)
        bind_func("<Button-4>", callback)  # Linux
        bind_func("<Button-5>", callback)  # Linux
    else:
        bind_func(f"<{modifier}-MouseWheel>", callback)
        bind_func(f"<{modifier}-Button-4>", callback)  # Linux
        bind_func(f"<{modifier}-Button-5>", callback)  # Linux


def un_bind_mousewheel(bind_func, modifier=None):
    if modifier is None:
        bind_func("<MouseWheel>")
        bind_func("<Button-4>")  # Linux
        bind_func("<Button-5>")  # Linux
    else:
        bind_func(f"<{modifier}-MouseWheel>")
        bind_func(f"<{modifier}-Button-4>")  # Linux
        bind_func(f"<{modifier}-Button-5>")  # Linux


def handle_canvas_scroll(canvas: tk.Canvas, event: tk.Event):
    canvas_x = canvas.winfo_rootx()
    canvas_y = canvas.winfo_rooty()
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if not (canvas_x <= event.x_root <= canvas_x + canvas_width and
            canvas_y <= event.y_root <= canvas_y + canvas_height):
        return

    viewable_region = canvas.bbox("all")
    if viewable_region is not None:
        canvas_height = canvas.winfo_height()
        content_height = viewable_region[3] - viewable_region[1]

        if content_height <= canvas_height:
            return

        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
