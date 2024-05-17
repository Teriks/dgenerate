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


import re
import tkinter as tk
import typing


class _FindDialog(tk.Toplevel):
    def __init__(self,
                 master,
                 name, text_widget: tk.Text,
                 position: tuple[int, int] = None,
                 width: typing.Optional[int] = None,
                 find_text='',
                 replace_text='',
                 replace_mode=False,
                 case_sensitive=False,
                 regex_mode=False):
        super().__init__(master=master)

        self.master = master
        self.title(name)
        self.text_widget = text_widget
        self.transient(master)
        self.replace_mode = replace_mode

        self.find_entry = tk.Text(self, height=1, width=40)
        if find_text:
            self.find_entry.insert('1.0', find_text)
        self.find_entry.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self.find_entry.bind('<KeyRelease>', lambda e: self._validate_find())

        self.replace_entry = None
        if replace_mode:
            self.replace_entry = tk.Text(self, height=1, width=40)
            if replace_text:
                self.replace_entry.insert('1.0', replace_text)
            self.replace_entry.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
            self.replace_entry.bind('<KeyRelease>', lambda e: self._validate_replacement())

        find_frame = tk.Frame(self)
        find_frame.grid(row=2, column=0, sticky='ew')

        self.case_var = tk.BooleanVar(value=case_sensitive)
        self.case_check = tk.Checkbutton(find_frame, text="Case Sensitive", variable=self.case_var)
        self.case_check.pack(side='left', padx=2, pady=2)

        self.regex_var = tk.BooleanVar(value=regex_mode)
        self.regex_check = tk.Checkbutton(find_frame, text="Regex Search", variable=self.regex_var)
        self.regex_check.pack(side='left', padx=2, pady=2)

        self.previous_button = tk.Button(find_frame, text='Previous', command=self._find_previous)
        self.previous_button.pack(side='left', padx=2, pady=2)

        self.next_button = tk.Button(find_frame, text='Next', command=self._find_next)
        self.next_button.pack(side='left', padx=2, pady=2)

        self.replace_button = None
        if replace_mode:
            self.replace_button = tk.Button(find_frame, text='Replace', command=self._replace_text)
            self.replace_button.pack(side='left', padx=2, pady=2)

        self.find_all_button = tk.Button(find_frame, text='Find All', command=self._find_all)
        self.find_all_button.pack(side='left', padx=2, pady=2)

        self.replace_all_button = None
        if replace_mode:
            self.replace_all_button = tk.Button(find_frame, text='Replace All', command=self._replace_all_text)
            self.replace_all_button.pack(side='left', padx=2, pady=2)

        self.text_widget.bind('<Button-1>', self._reset_search)

        self.grid_columnconfigure(0, weight=1)

        self.last_find = None

        self.resizable(True, False)

        self.withdraw()

        self.master.update_idletasks()
        self.update_idletasks()

        self.minsize(self.winfo_reqwidth(), 0)

        if position is None:
            window_width = self.master.winfo_width()
            window_height = self.master.winfo_height()
            top_level_width = self.winfo_width()
            top_level_height = self.winfo_height()

            position_top = self.master.winfo_y() + window_height // 2 - top_level_height // 2
            position_left = self.master.winfo_x() + window_width // 2 - top_level_width // 2
            if width is None:
                self.geometry(f"+{position_left}+{position_top}")
            else:
                self.geometry(f"{width}x{self.winfo_reqheight()}+{position_left}+{position_top}")
        else:
            if width is None:
                self.geometry("+{}+{}".format(*position))
            else:
                self.geometry(f"{width}x{self.winfo_reqheight()}+{position[0]}+{position[1]}")

        self.deiconify()

        self.find_entry.focus_set()

    def _reset_search(self, event=None):
        self.text_widget.tag_remove('found', '1.0', tk.END)
        self.text_widget.mark_set('insert', '@%d,%d' % (event.x, event.y))
        self.last_find = self.text_widget.index('insert')
        self.last_find_end = None
        return 'break'

    def _text_index_to_absolute(self, index):
        line, char = map(int, self.text_widget.index(index).split('.'))
        absolute_index = 0
        for i in range(1, line):
            line_text = self.text_widget.get(f"{i}.0", f"{i}.end")
            absolute_index += len(line_text) + 1
        absolute_index += char
        return absolute_index

    def _find_next(self):
        s = self.find_entry.get("1.0", "end-1c")
        start_idx = '1.0' if not self.last_find else self.last_find + '+1c'
        while True:
            if self.regex_var.get():
                pattern = re.compile(s, re.I if not self.case_var.get() else 0)
                match = pattern.search(self.text_widget.get('1.0', tk.END), self._text_index_to_absolute(start_idx))
                if match:
                    idx = self.text_widget.index(f'1.0 + {match.start()} chars')
                    end_idx = f'{idx}+{len(match.group())}c'
                    self.text_widget.tag_remove('found', '1.0', tk.END)
                    self.text_widget.tag_add('found', idx, end_idx)
                    self.text_widget.mark_set(tk.INSERT, end_idx)
                    self.text_widget.see(idx)
                    self.text_widget.tag_config('found', foreground='white', background='blue')
                    self.last_find = idx
                    self.last_find_end = end_idx
                    break
                elif start_idx == '1.0':
                    break
                else:
                    start_idx = '1.0'
            else:
                idx = self.text_widget.search(s, start_idx, nocase=not self.case_var.get(), stopindex=tk.END)
                if idx:
                    end_idx = f'{idx}+{len(s)}c'
                    self.text_widget.tag_remove('found', '1.0', tk.END)
                    self.text_widget.tag_add('found', idx, end_idx)
                    self.text_widget.mark_set(tk.INSERT, end_idx)
                    self.text_widget.see(idx)
                    self.text_widget.tag_config('found', foreground='white', background='blue')
                    self.last_find = idx
                    self.last_find_end = end_idx
                    break
                elif start_idx == '1.0':
                    break
                else:
                    start_idx = '1.0'

    def _find_previous(self):
        if not self.last_find:
            return
        s = self.find_entry.get("1.0", "end-1c")
        pattern = re.compile(re.escape(s) if not self.regex_var.get() else s, re.I if not self.case_var.get() else 0)
        while True:
            text = self.text_widget.get('1.0', self.last_find)
            matches = [(m.start(), m.end(), m.group()) for m in pattern.finditer(text)]
            if matches:
                start, end, match = matches[-1]
                idx = self.text_widget.index(f'1.0 + {start} chars')
                end_idx = f'{idx}+{len(match)}c'
                self.text_widget.tag_remove('found', '1.0', tk.END)
                self.text_widget.tag_add('found', idx, end_idx)
                self.text_widget.mark_set(tk.INSERT, idx)
                self.text_widget.see(idx)
                self.text_widget.tag_config('found', foreground='white', background='blue')
                self.last_find = idx
                self.last_find_end = end_idx
                break
            elif self.last_find == self.text_widget.index(tk.END):
                break
            else:
                self.last_find = self.text_widget.index(tk.END)
                self.last_find_end = None

    def _find_all(self):
        self.text_widget.tag_remove('found', '1.0', tk.END)
        self.last_find = '1.0'
        self.last_find_end = None

        s = self.find_entry.get("1.0", "end-1c")
        if self.regex_var.get():
            pattern = re.compile(s, re.I if not self.case_var.get() else 0)
            matches = pattern.finditer(self.text_widget.get('1.0', tk.END))
            for match in matches:
                idx = self.text_widget.index(f'1.0 + {match.start()} chars')
                end_idx = f'{idx}+{len(match.group())}c'
                self.text_widget.tag_add('found', idx, end_idx)
                self.text_widget.tag_config('found', foreground='white', background='blue')
        else:
            start_idx = '1.0'
            while True:
                idx = self.text_widget.search(s, start_idx, nocase=not self.case_var.get(), stopindex=tk.END)
                if idx:
                    end_idx = f'{idx}+{len(s)}c'
                    self.text_widget.tag_add('found', idx, end_idx)
                    self.text_widget.tag_config('found', foreground='white', background='blue')
                    start_idx = end_idx
                else:
                    break

    def _replace_all_text(self):
        find = self.find_entry.get("1.0", "end-1c")
        replacement = self.replace_entry.get("1.0", "end-1c")
        in_text = self.text_widget.get('1.0', "end-1c")

        self.last_find = '1.0'
        self.last_find_end = None
        self.text_widget.edit_separator()

        if self.regex_var.get():
            pattern = re.compile(find, re.I if not self.case_var.get() else 0)
            self.text_widget.replace('1.0', tk.END, pattern.sub(replacement, in_text))
        else:
            self.text_widget.replace('1.0', tk.END, in_text.replace(find, replacement))

    def _replace_text(self):
        if not self.last_find or not self.last_find_end:
            return

        find = self.find_entry.get("1.0", "end-1c")
        replacement = self.replace_entry.get("1.0", "end-1c")

        if self.regex_var.get():
            pattern = re.compile(find, re.I if not self.case_var.get() else 0)
            search_text = self.text_widget.get(self.last_find, self.last_find_end)
            match = pattern.search(search_text)
            if match:
                replacement = self._custom_expand(match, replacement)

        self.text_widget.edit_separator()
        self.text_widget.replace(self.last_find, self.last_find_end, replacement)

        self.last_find_end = None
        self._find_next()

    def _validate_find(self):
        s = self.find_entry.get("1.0", "end-1c")
        if self.regex_var.get():
            try:
                re.compile(s)
                self.find_entry.tag_remove("invalid", "1.0", tk.END)
            except re.error:
                self.find_entry.tag_add("invalid", "1.0", tk.END)
                self.find_entry.tag_config("invalid", background="red")
        else:
            self.find_entry.tag_remove("invalid", "1.0", tk.END)

    def _validate_replacement(self):
        s = self.find_entry.get("1.0", "end-1c")
        replacement = self.replace_entry.get("1.0", "end-1c")
        if self.regex_var.get():
            pattern = re.compile(s, re.I if not self.case_var.get() else 0)
            self.replace_entry.tag_remove("invalid", "1.0", tk.END)
            for match in re.finditer(r"\\(\d+)|\\{(\d+)}", replacement):
                group_num = int(list(filter(None, match.groups()))[0])
                if pattern.groups < group_num:
                    start = f"1.0 + {match.start()} chars"
                    end = f"{start} + {len(match.group())} chars"
                    self.replace_entry.tag_add("invalid", start, end)
            self.replace_entry.tag_config("invalid", background="red")
        else:
            self.replace_entry.tag_remove("invalid", "1.0", tk.END)

    @staticmethod
    def _custom_expand(match, format_string):

        occurrences = re.findall(r'\\(\d+)|\\{(\d+)}', format_string)

        for group in occurrences:
            group_num = list(filter(None, group))[0]
            if group[0] != '':
                format_string = format_string.replace(f'\\{group_num}', match.group(int(group_num)))
            else:
                format_string = format_string.replace(f'\\{{{group_num}}}', match.group(int(group_num)))

        return format_string

    def destroy(self) -> None:
        try:
            self.text_widget.unbind('<Button-1>')
            self.text_widget.tag_remove('found', '1.0', tk.END)
        except tk.TclError:
            pass
        super().destroy()


class _FindReplaceDialogState:
    def __init__(self):
        self.find_dialog = None
        self.find_text = ''
        self.replace_text = ''
        self.regex_mode = False
        self.case_sensitive = False
        self.last_pos = None
        self.last_find_dialog_width = None
        self.last_find_replace_dialog_width = None

    def open_find_dialog(self, master, name, text_box):
        if self.find_dialog is not None:
            last_width = self.find_dialog.winfo_width()
            self.regex_mode = self.find_dialog.regex_var.get()
            self.case_sensitive = self.find_dialog.case_var.get()
            if self.find_dialog.text_widget is text_box and not self.find_dialog.replace_mode:
                try:
                    self.find_dialog.find_entry.focus_set()
                    return
                except tk.TclError:
                    pass
            else:
                self.find_text = self.find_dialog.find_entry.get('1.0', 'end-1c')

                if self.find_dialog.replace_mode:
                    self.replace_text = self.find_dialog.replace_entry.get('1.0', 'end-1c')
                    self.last_find_replace_dialog_width = last_width
                else:
                    self.last_find_dialog_width = last_width

                self.last_pos = (self.find_dialog.winfo_x(), self.find_dialog.winfo_y())
                self.find_dialog.destroy()
                self.find_dialog = None

        self.find_dialog = _FindDialog(
            master, name, text_box,
            position=self.last_pos,
            width=self.last_find_dialog_width,
            find_text=self.find_text,
            regex_mode=self.regex_mode,
            case_sensitive=self.case_sensitive)

        self.find_dialog.protocol("WM_DELETE_WINDOW", self._on_closing)

    def open_find_replace_dialog(self, master, name, text_box):
        if self.find_dialog is not None:
            last_width = self.find_dialog.winfo_width()
            self.regex_mode = self.find_dialog.regex_var.get()
            self.case_sensitive = self.find_dialog.case_var.get()
            if self.find_dialog.text_widget is text_box and self.find_dialog.replace_mode:
                try:
                    self.find_dialog.find_entry.focus_set()
                    return
                except tk.TclError:
                    pass
            else:
                self.find_text = self.find_dialog.find_entry.get('1.0', 'end-1c')
                self.last_pos = (self.find_dialog.winfo_x(), self.find_dialog.winfo_y())

                if self.find_dialog.replace_mode:
                    self.last_find_replace_dialog_width = last_width
                else:
                    self.last_find_dialog_width = last_width

                self.find_dialog.destroy()
                self.find_dialog = None

        self.find_dialog = _FindDialog(
            master, name, text_box,
            find_text=self.find_text,
            replace_text=self.replace_text,
            position=self.last_pos,
            width=self.last_find_replace_dialog_width,
            regex_mode=self.regex_mode,
            case_sensitive=self.case_sensitive,
            replace_mode=True)

        self.find_dialog.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.regex_mode = self.find_dialog.regex_var.get()
        self.case_sensitive = self.find_dialog.case_var.get()
        self.last_pos = (self.find_dialog.winfo_x(), self.find_dialog.winfo_y())

        if self.find_dialog.replace_mode:
            self.last_find_replace_dialog_width = self.find_dialog.winfo_width()
        else:
            self.last_find_dialog_width = self.find_dialog.winfo_width()

        self.find_text = self.find_dialog.find_entry.get('1.0', 'end-1c')
        if self.find_dialog.replace_mode:
            self.replace_text = self.find_dialog.replace_entry.get('1.0', 'end-1c')
        self.find_dialog.destroy()
        self.find_dialog = None


_dialog_state = _FindReplaceDialogState()


def open_find_dialog(master, name, text_box):
    _dialog_state.open_find_dialog(master, name, text_box)


def open_find_replace_dialog(master, name, text_box):
    _dialog_state.open_find_replace_dialog(master, name, text_box)
