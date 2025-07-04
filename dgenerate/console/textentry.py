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


class TextEntry(tk.Entry):
    """
    A tk.Entry widget with a right-click context menu providing copy, cut, paste, and delete operations.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Cut", command=self.__context_cut)
        self.context_menu.add_command(label="Copy", command=self.__context_copy)
        self.context_menu.add_command(label="Paste", command=self.__context_paste)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Delete", command=self.__context_delete)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All", command=self.__context_select_all)

        self.bind("<Button-3>", self.__show_context_menu)
    
    def __show_context_menu(self, event):
        self.focus()
        self.__update_menu_state()
        
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def __update_menu_state(self):
        has_selection = self.selection_present()

        has_clipboard = False
        try:
            self.clipboard_get()
            has_clipboard = True
        except tk.TclError:
            pass

        self.context_menu.entryconfig("Cut", state="normal" if has_selection else "disabled")
        self.context_menu.entryconfig("Copy", state="normal" if has_selection else "disabled")
        self.context_menu.entryconfig("Paste", state="normal" if has_clipboard else "disabled")
        self.context_menu.entryconfig("Delete", state="normal" if has_selection else "disabled")
    
    def __context_cut(self):
        if self.selection_present():
            self.clipboard_clear()
            self.clipboard_append(self.selection_get())
            super().delete(tk.SEL_FIRST, tk.SEL_LAST)
    
    def __context_copy(self):
        if self.selection_present():
            self.clipboard_clear()
            self.clipboard_append(self.selection_get())
    
    def __context_paste(self):
        try:
            text = self.clipboard_get()
            if self.selection_present():
                super().delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.insert(tk.INSERT, text)
        except tk.TclError:
            pass
    
    def __context_delete(self):
        if self.selection_present():
            super().delete(tk.SEL_FIRST, tk.SEL_LAST)
    
    def __context_select_all(self):
        self.select_range(0, tk.END) 