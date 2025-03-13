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
import inspect
import json
import platform
import re
import tkinter as tk
import tkinter.ttk as ttk
import typing

import dgenerate.console.recipesformentries as _recipesformentries
import dgenerate.console.resources as _resources
import dgenerate.console.util as _util
from dgenerate.console.mousewheelbind import bind_mousewheel, un_bind_mousewheel


class _RecipeTemplateTag:
    classname: str
    config: dict
    placeholder: str

    def __init__(self, classname, config, placeholder):
        self.classname = classname
        self.config = config
        self.placeholder = placeholder


def _find_template_tags(text) -> list[_RecipeTemplateTag]:
    tag_pattern = r'@(\w+)\[\{'

    potential_starts = [(m.start(), m.group(1)) for m in re.finditer(tag_pattern, text)]

    tags = []

    for start_pos, classname in potential_starts:
        stack = []
        json_start = start_pos + len(classname) + 2
        json_end = json_start

        while json_end <= len(text):
            try:
                json.loads(text[json_start:json_end])
                break
            except json.JSONDecodeError:
                json_end += 1

        if not stack:
            json_content = text[json_start:json_end]
            try:
                parsed_json = json.loads(json_content)

                full_tag = text[start_pos:json_end + 1]

                tags.append(
                    _RecipeTemplateTag(
                        classname,
                        parsed_json,
                        full_tag,
                    ))

            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Invalid JSON in recipe template: {text[start_pos:json_end + 1]}")

    return tags


class _RecipesForm(tk.Toplevel):
    def __init__(
            self,
            insert: typing.Callable[[str], None],
            master=None,
            position: tuple[int, int] = None,
            size: tuple[int, int] = None
    ):
        super().__init__(master)
        self.title('Insert Recipe')

        # Get all entry classes
        self._entry_classes = {
            c.NAME: c for c in
            _recipesformentries.__dict__.values()
            if inspect.isclass(c) and issubclass(c, _recipesformentries._Entry)
        }

        self.minsize(width=600, height=200)

        self._templates = None
        self._dropdown = None
        self._templates_dict: dict[str, str] = _resources.get_recipes()
        self._template_names: list[str] = list(_resources.get_recipes().keys())

        self._current_template = tk.StringVar(value=self._template_names[0])
        self._entries: list[_recipesformentries._Entry] = []
        self._content: typing.Optional[str] = None

        self._insert = insert

        self.transient(master)

        self._create_scrollable_form()

        if platform.system() == 'Windows':
            default_size = (700, 600)
        else:
            default_size = (800, 600)

        _util.position_toplevel(
            master=master,
            toplevel=self,
            size=size if size else default_size,
            position=position
        )

    def _apply_templates(self) -> None:
        """Apply the templates to the form entries."""
        content = self._content
        missing_fields = False

        for entry in self._entries:
            content = entry.template(content)
            if entry.is_empty() and not entry.optional:
                entry.invalid()
                missing_fields = True
            elif not entry.is_valid():
                entry.invalid()
                missing_fields = True
            else:
                entry.valid()

        if not missing_fields:
            self._content = content
            value = ('\n'.join(line.lstrip() for line in self._content.splitlines())).strip()
            if value:
                self._insert(value)
            self.destroy()

    def _create_scrollable_form(self) -> None:
        """Create a scrollable form for the template entries."""
        outer_frame = tk.Frame(self, bd=3, relief="sunken")
        outer_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        self.canvas = tk.Canvas(outer_frame, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, padx=10, pady=10, highlightthickness=0)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')

        outer_frame.grid_rowconfigure(0, weight=1)
        outer_frame.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.bind("<Configure>", self._on_resize)

        self._dropdown = tk.OptionMenu(self, self._current_template, *self._template_names,
                                       command=lambda s: self._update_form(s, preserve_width=True))
        self._dropdown.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        self.grid_columnconfigure(0, weight=1)

        # Bind mouse wheel events
        bind_mousewheel(self.canvas.bind_all, self._on_mouse_wheel)

        self._update_form(self._current_template.get())

    def _on_mouse_wheel(self, event):
        """Scroll the canvas with the mouse wheel."""
        viewable_region = self.canvas.bbox("all")
        if viewable_region is not None:
            canvas_height = self.canvas.winfo_height()
            content_height = viewable_region[3] - viewable_region[1]

            if content_height <= canvas_height:
                return  # Do nothing if there's no overflow

            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                if event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")

    def _on_resize(self, event):
        """Update the canvas width to match the new window width."""
        canvas_width = self.winfo_width() - self.scrollbar.winfo_width() - 20
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _update_form(self, selection: str, preserve_width: bool = False) -> None:
        """Update the form based on the selected template."""
        self.canvas.yview_moveto(0)

        for widget in self.scrollable_frame.winfo_children():
            if widget != self._dropdown:
                widget.destroy()

        self._entries = []
        self._content = self._templates_dict[selection]
        self._templates = _find_template_tags(self._content)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        row_offset = 1
        for template in self._templates:
            classname = template.classname
            config = template.config

            if config.get('divider-before', False):
                separator = ttk.Separator(self.scrollable_frame, orient='horizontal')
                separator.grid(row=row_offset, column=0, sticky='ew', columnspan=100,
                               pady=_recipesformentries.DIVIDER_YPAD)
                row_offset += 1

            if classname in self._entry_classes:
                entry = self._entry_classes[classname](
                    recipe_form=self,
                    master=self.scrollable_frame,
                    row=row_offset,
                    config=config,
                    placeholder=template.placeholder
                )
                row_offset += entry.widget_rows
            else:
                raise RuntimeError(f'Unknown template placeholder: {classname}')

            if config.get('divider-after', False):
                separator = ttk.Separator(self.scrollable_frame, orient='horizontal')
                separator.grid(row=row_offset, column=0, sticky='ew', columnspan=100,
                               pady=_recipesformentries.DIVIDER_YPAD)
                row_offset += 1

            self._entries.append(entry)

        apply_button = tk.Button(self, text='Insert', command=self._apply_templates)
        apply_button.grid(row=3, column=0, padx=5, pady=5, columnspan=2)

    def destroy(self) -> None:
        un_bind_mousewheel(self.canvas.unbind_all)
        super().destroy()


_last_pos = None
_last_size = None
_cur_window = None


def request_recipe(master, insert: typing.Callable[[str], None]):
    global _last_size, _last_pos, _cur_window

    if _cur_window is not None:
        _cur_window.focus_set()
        return

    _cur_window = _RecipesForm(
        insert=insert,
        master=master,
        position=_last_pos,
        size=_last_size
    )

    og_destroy = _cur_window.destroy

    # noinspection PyUnresolvedReferences
    def destroy():
        global _last_size, _last_pos, _cur_window
        _last_size = (_cur_window.winfo_width(), _cur_window.winfo_height())
        _last_pos = (_cur_window.winfo_x(), _cur_window.winfo_y())
        _cur_window = None
        og_destroy()

    _cur_window.destroy = destroy

    def on_closing():
        _cur_window.destroy()

    _cur_window.protocol("WM_DELETE_WINDOW", on_closing)
