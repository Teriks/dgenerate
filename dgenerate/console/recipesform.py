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

import dgenerate.console.appearance as _appearance
import dgenerate.console.formentries as _formentries
import dgenerate.console.formentries.entry as _form_entry
import dgenerate.console.resources as _resources
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
from dgenerate.console.mousewheelbind import bind_mousewheel, handle_canvas_scroll, un_bind_mousewheel
from dgenerate.console.combobox import ComboBox

_dialog_state = _util.DialogState(save_position=True, save_size=True)

_FILTER_DEBOUNCE_MS = 120


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

            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in recipe template: {text[start_pos:json_end + 1]}") from e

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
            _formentries.__dict__.values()
            if inspect.isclass(c) and issubclass(c, _formentries._Entry)
        }

        self.minsize(width=600, height=200)

        self._templates = None
        self._dropdown = None
        self._filter_entry = None
        self._filter_frame = None
        self._filter_toggle = None
        self._filter_visible = False
        self._templates_dict: dict[str, str] = _resources.get_recipes()
        self._template_names: list[str] = list(self._templates_dict.keys())
        self._search_rows = [
            (name, name.lower(), self._templates_dict[name].lower())
            for name in self._template_names
        ]

        self._current_template = tk.StringVar(value=self._template_names[0])
        self._filter_var = tk.StringVar()
        self._filter_after_id: typing.Optional[str] = None
        self._entries: list[_formentries._Entry] = []
        self._content: typing.Optional[str] = None

        self._last_known_selection = None

        self._insert = insert

        self._id_map = dict()

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
            if entry._row_inactive:
                content = _form_entry.replace_first(content, entry.placeholder, '')
                continue
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

    def _filtered_template_names(self) -> list[str]:
        query = self._filter_var.get().strip().lower()
        if not query:
            return list(self._template_names)
        by_title = [name for name, title_l, _body_l in self._search_rows if query in title_l]
        if by_title:
            return by_title
        terms = query.split()
        return [
            name for name, _title_l, body_l in self._search_rows
            if all(term in body_l for term in terms)
        ]

    def _filter_query(self) -> str:
        return self._filter_var.get().strip()

    def _update_filter_toggle_label(self) -> None:
        if self._filter_toggle is None:
            return

        query = self._filter_query()
        if self._filter_visible:
            self._filter_toggle.configure(text='Hide Filter')
        elif query:
            display = query if len(query) <= 14 else query[:13] + '…'
            self._filter_toggle.configure(text=f'Filter: {display}')
        else:
            self._filter_toggle.configure(text='Filter')

    def _set_filter_visible(self, visible: bool) -> None:
        if self._filter_frame is None:
            return

        self._filter_visible = visible
        if visible:
            self._filter_frame.grid()
            self._filter_entry.focus_set()
            self._filter_entry.icursor(tk.END)
        else:
            self._filter_frame.grid_remove()
            if self._filter_query():
                self._filter_var.set('')
        self._update_filter_toggle_label()

    def _toggle_filter(self) -> None:
        self._set_filter_visible(not self._filter_visible)

    def _on_filter_escape(self, _event=None):
        if self._filter_query():
            self._filter_var.set('')
        else:
            self._set_filter_visible(False)
        return 'break'

    def _on_filter_shortcut(self, _event=None):
        self._set_filter_visible(True)
        return 'break'

    def _on_filter_changed(self, *_args) -> None:
        self._update_filter_toggle_label()
        if self._filter_after_id is not None:
            self.after_cancel(self._filter_after_id)
        self._filter_after_id = self.after(_FILTER_DEBOUNCE_MS, self._apply_recipe_filter)

    def _apply_recipe_filter(self) -> None:
        self._filter_after_id = None
        if self._dropdown is None:
            return

        filtered = self._filtered_template_names()
        if list(self._dropdown['values']) != filtered:
            self._dropdown['values'] = filtered

    def _create_scrollable_form(self) -> None:
        """Create a scrollable form for the template entries."""
        self._filter_frame = ttk.Frame(self)
        self._filter_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=(5, 0))
        self._filter_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self._filter_frame, text='Filter').grid(row=0, column=0, sticky='w')
        self._filter_entry = _t_entry.TextEntry(self._filter_frame, textvariable=self._filter_var)
        self._filter_entry.grid(row=0, column=1, sticky='ew', padx=(5, 0))
        self._filter_var.trace_add('write', self._on_filter_changed)
        self._filter_entry.bind('<Escape>', self._on_filter_escape)
        self._filter_frame.grid_remove()

        recipe_row = ttk.Frame(self)
        recipe_row.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        recipe_row.grid_columnconfigure(0, weight=1)

        self._dropdown = ComboBox(
            recipe_row,
            textvariable=self._current_template,
            values=self._template_names
        )
        self._dropdown.bind(
            '<<ComboboxSelected>>',
            lambda e: self._update_form(self._current_template.get())
        )
        self._dropdown.grid(row=0, column=0, sticky='ew')

        self._filter_toggle = ttk.Button(
            recipe_row,
            text='Filter',
            command=self._toggle_filter,
            width=14,
        )
        self._filter_toggle.grid(row=0, column=1, sticky='e', padx=(5, 0))

        self.bind('<Control-f>', self._on_filter_shortcut)
        self.bind('<Control-F>', self._on_filter_shortcut)

        outer_frame = ttk.Frame(self)
        outer_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=(5, 0))

        viewport = tk.Frame(outer_frame)
        _appearance.configure_viewport(viewport)
        viewport.grid(row=0, column=0, sticky='nsew')

        self.canvas = tk.Canvas(viewport, highlightthickness=0, borderwidth=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding=(10, 10))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.grid(row=0, column=1, sticky='ns')

        outer_frame.grid_rowconfigure(0, weight=1)
        outer_frame.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ttk.Button(self, text='Insert', command=self._apply_templates).grid(
            row=3, column=0, padx=5, pady=5)

        self.bind("<Configure>", self._on_resize)

        # Bind mouse wheel events
        self.bind_mousewheel()

        self._update_form(self._current_template.get())
        self._update_filter_toggle_label()
        _appearance.install(self)

    def bind_mousewheel(self):
        bind_mousewheel(self.canvas.bind_all, self._on_mouse_wheel)

    def unbind_mousewheel(self):
        un_bind_mousewheel(self.canvas.unbind_all)

    def _on_mouse_wheel(self, event):
        if 'popdown' in str(event.widget).lower():
            return "break"

        # Notify entries of scroll event
        for entry in self._entries:
            entry.on_form_scroll()
        handle_canvas_scroll(self.canvas, event)

    def _on_resize(self, event):
        canvas_width = self.winfo_width() - self.scrollbar.winfo_width() - 20
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def get_entry_by_id(self, entry_id: str) -> typing.Optional[_formentries._Entry]:
        return self._id_map.get(entry_id, None)

    def _update_form(self, selection: str) -> None:
        if self._last_known_selection == selection:
            return
        else:
            self._last_known_selection = selection

        self.canvas.yview_moveto(0)

        for widget in self.scrollable_frame.winfo_children():
            if widget != self._dropdown:
                widget.destroy()

        self._id_map.clear()

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
                               pady=_formentries.DIVIDER_YPAD)
                row_offset += 1

            if classname in self._entry_classes:
                entry = self._entry_classes[classname](
                    form=self,
                    master=self.scrollable_frame,
                    row=row_offset,
                    config=config,
                    placeholder=template.placeholder
                )

                entry.install_argument_help()

                entry_id = config.get('id', None)
                if entry_id is not None:
                    self._id_map[entry_id] = entry

                row_offset += entry.widget_rows
            else:
                raise RuntimeError(f'Unknown template placeholder: {classname}')

            if config.get('divider-after', False):
                separator = ttk.Separator(self.scrollable_frame, orient='horizontal')
                separator.grid(row=row_offset, column=0, sticky='ew', columnspan=100,
                               pady=_formentries.DIVIDER_YPAD)
                row_offset += 1

            self._entries.append(entry)

        self._bind_refresh()
        self.refresh_exclusions()
        _appearance.apply(self)

    @staticmethod
    def _entry_variables(entry) -> list:
        """Tk variables whose writes can change an entry's value or active state."""
        variables = [
            getattr(entry, name) for name in ('text_var', 'bool_var', 'enabled_var', 'uri_var')
            if getattr(entry, name, None) is not None
        ]
        variables.extend(getattr(entry, 'bool_vars', []))
        return variables

    def _bind_refresh(self) -> None:
        """
        Re-evaluate conditions and exclusions on any entry value change.

        'enabled-when' / 'visible-when' conditions read ``text_var`` or
        ``bool_var`` of their source entry, which are traced here too, so a
        single refresh path applies conditions before exclusions.
        """
        traced = set()
        for entry in self._entries:
            for variable in self._entry_variables(entry):
                if str(variable) in traced:
                    continue
                traced.add(str(variable))
                variable.trace_add('write', lambda *_args: self.refresh_exclusions())

    def _matches_condition(self, condition) -> bool:
        if 'all' in condition:
            return all(self._matches_condition(item) for item in condition['all'])

        source = self._id_map.get(condition.get('id'))
        if source is None:
            return False

        variable = getattr(source, 'text_var', None)
        if variable is None:
            variable = getattr(source, 'bool_var', None)
        if variable is None:
            return False

        value = variable.get()
        equals = condition.get('equals', '')
        if isinstance(value, bool):
            return value == (str(equals).lower() == 'true')
        return str(value).strip() == str(equals).strip()

    def refresh_exclusions(self) -> None:
        """Re-evaluate all enabled-when conditions, then exclusions."""
        enabled = {}
        for entry in self._entries:
            spec = entry.config.get('enabled-when') or entry.config.get('visible-when')
            enabled[entry] = self._matches_condition(spec) if spec else True

        # Exclusion is mutual, a recipe only needs to declare it on one
        # side of a pair, usually on the later entry.
        exclusions = {entry: [] for entry in self._entries}
        for entry in self._entries:
            exclude = entry.config.get('exclude', [])
            if isinstance(exclude, str):
                exclude = [exclude]
            for other_id in exclude:
                other_entry = self.get_entry_by_id(other_id)
                if other_entry is not None and other_entry is not entry:
                    exclusions[entry].append(other_entry)
                    exclusions[other_entry].append(entry)

        # Only entries left enabled by their conditions can exclude others.
        # Entries are visited in form order, so if two exclusive entries are
        # somehow both active, the first one wins and the other is grayed
        # out, rather than both graying each other out.
        for entry in self._entries:
            if enabled[entry] and entry.is_active():
                for other_entry in exclusions[entry]:
                    enabled[other_entry] = False

        # Rows only change state when their result differs, so an entry's
        # own widget states are not reset on every refresh.
        for entry in self._entries:
            entry.set_row_enabled(enabled[entry])

    def destroy(self) -> None:
        if self._filter_after_id is not None:
            self.after_cancel(self._filter_after_id)
            self._filter_after_id = None
        self.unbind_mousewheel()
        super().destroy()


def request_recipe(master, insert: typing.Callable[[str], None]):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_RecipesForm,
        state=_dialog_state,
        dialog_kwargs={'insert': insert}
    )
