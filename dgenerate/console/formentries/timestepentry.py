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
import tkinter.ttk as ttk

import dgenerate.console.formentries.curveplot as _curveplot
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.sigmaschedule as _schedule
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
from dgenerate.console.spinbox import IntSpinbox


class _TimestepBuildDialog(_curveplot.CurvePlot, tk.Toplevel):
    """Editor for one integer timestep list. Nothing is written until Insert."""

    def __init__(self, master, title, origin, steps, values, scheduler_names,
                 scheduler_name, use_flow_sigmas, on_accept, on_scheduler=None,
                 model_type=None):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self._init_plot()
        self._width = int(560 * self._px)
        self.model_type = model_type
        self.origin = list(origin)
        self.base = list(origin)
        self.values = list(values)
        self.scheduler_names = list(scheduler_names)
        self._on_scheduler = on_scheduler
        self._opened_name = scheduler_name
        self._opened_flow = bool(use_flow_sigmas)
        self._using_formula = False
        self.dirty = False
        self._notice = ''
        self._on_accept = on_accept
        self._ready = False
        self.min_steps = 2
        self.max_steps = 1000

        self.steps_var = tk.StringVar(value=str(steps))
        body = ttk.Frame(self, padding=8)
        self._body = body
        body.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        row = 0
        self.scheduler_var = None
        self.flow_var = None
        self.flow_check = None
        if self.scheduler_names:
            ttk.Label(body, text='Scheduler', anchor='e').grid(
                row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
            self.scheduler_var = tk.StringVar(value=scheduler_name)
            self.scheduler = ttk.Combobox(
                body, textvariable=self.scheduler_var, state='readonly',
                values=self.scheduler_names)
            self.scheduler.grid(row=row, column=1, columnspan=2, padx=_entry.ROW_XPAD, sticky='ew')
            self.scheduler.bind('<<ComboboxSelected>>', lambda _event: self._load_scheduler())
            row += 1
            self.flow_var = tk.BooleanVar(value=use_flow_sigmas)
            self.flow_check = ttk.Checkbutton(
                body, text='Use flow sigmas', variable=self.flow_var,
                command=self._load_scheduler)
            self.flow_check.grid(row=row, column=1, columnspan=2, padx=_entry.ROW_XPAD, sticky='w')
            if not _schedule.scheduler_supports_flow(scheduler_name):
                self.flow_check.grid_remove()
            row += 1

        ttk.Label(body, text='Steps', anchor='e').grid(
            row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.steps = IntSpinbox(
            body, from_=self.min_steps, to=self.max_steps, textvariable=self.steps_var)
        self.steps.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.steps.create_spin_buttons(body).grid(row=row, column=2, sticky='w')
        row += 1

        row = self._build_plot(body, row)

        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=3, pady=(6, 0), sticky='e')
        ttk.Button(buttons, text='Reset', command=self._reset).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='Cancel', command=self.destroy).pack(side='left')
        self.insert_button = ttk.Button(buttons, text='Insert', command=self._ok)
        self.insert_button.pack(side='left', padx=(8, 0))

        self.steps_var.trace_add('write', lambda *_args: self._on_steps())
        self._ready = True
        self._sync_values()
        self._show_point(0)
        self.bind('<Escape>', lambda _event: self.destroy())
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        import dgenerate.console.appearance as _appearance
        _appearance.install(self)
        self.update_idletasks()
        _util.position_toplevel(master, self, size=(self._width, self.winfo_reqheight()))
        self.minsize(int(420 * self._px), self.winfo_reqheight())
        self.grab_set()

    def _steps(self) -> int | None:
        try:
            steps = int(float(self.steps_var.get()))
        except ValueError:
            return None
        if steps < self.min_steps or steps > self.max_steps:
            return None
        return steps

    def _on_steps(self):
        if not self._ready:
            return
        steps = self._steps()
        if steps is None:
            return
        if self._using_formula:
            self._load_scheduler()
            return
        sampled = _schedule.resample([float(value) for value in self.origin], steps)
        self._forget_history()
        self.base = _schedule._integers(sampled)
        self.values = list(self.base)
        self.dirty = self.values != self.origin
        self._notice = ''
        self._sync_values()
        self._show_point(self._selected)
        self._draw()

    def _set_steps_quietly(self, steps: int):
        self._ready = False
        self.steps_var.set(str(steps))
        self._ready = True

    def _sync_flow_check(self, name: str):
        if self.flow_check is None:
            return
        shown = bool(self.flow_check.grid_info())
        if _schedule.scheduler_supports_flow(name):
            self.flow_check.grid()
        else:
            self.flow_check.grid_remove()
        # Refitting resets the window geometry, which flashes it on Windows.
        if bool(self.flow_check.grid_info()) != shown:
            self._refit()

    def _refit(self):
        _util.refit_toplevel(self, self._width, self._body)

    def _load_scheduler(self):
        if not self._ready or self.scheduler_var is None:
            return
        name = self.scheduler_var.get().strip()
        steps = self._steps()
        if not name or steps is None:
            return
        flow = bool(self.flow_var.get()) if self.flow_var is not None else False
        self._sync_flow_check(name)
        try:
            curve = _schedule.scheduler_timesteps(name, steps, flow, self.model_type)
        except _schedule.SigmaScheduleError as error:
            self._notice = str(error)
            self._show_point(None)
            return
        self._notice = ''
        self._using_formula = True
        self._forget_history()
        self.base = curve
        self.values = list(curve)
        self.dirty = self.values != self.origin
        if len(curve) != steps:
            self._set_steps_quietly(len(curve))
        self._sync_values()
        self._show_point(self._selected)
        self._draw()

    def _reset(self):
        """Undo edits and steps. The chosen scheduler stays."""
        self.dirty = False
        self._notice = ''
        self._selected = None
        self._forget_history()
        self._set_steps_quietly(len(self.origin))
        if self._using_formula:
            self._load_scheduler()
            if not self._notice:
                return
            self._using_formula = False
            self._set_steps_quietly(len(self.origin))
        self.base = list(self.origin)
        self.values = list(self.origin)
        self._sync_values()
        self._show_point(0)
        self._draw()

    def _ok(self):
        if not self._commit_typed():
            self.values_entry.focus_set()
            return
        if (self._on_scheduler is not None and self._using_formula and
                self.scheduler_var is not None and not self._notice):
            flow = bool(self.flow_var.get()) if self.flow_var is not None else False
            name = self.scheduler_var.get().strip()
            if name and (name != self._opened_name or flow != self._opened_flow):
                self._on_scheduler(name, flow)
        self._on_accept(list(self.values), self.dirty)
        self.destroy()

    def _ceiling(self) -> int | None:
        name = self.scheduler_var.get().strip() if self.scheduler_var is not None else ''
        return _schedule.timestep_ceiling(name or None, self.model_type)

    def _axis_top(self) -> float:
        if self._drag_top is not None:
            return self._drag_top
        peak = max(self.values) if self.values else 1
        ceiling = self._ceiling()
        if ceiling is not None:
            peak = max(peak, ceiling)
        return max(peak, 1) * 1.08

    def _reference(self):
        return self.base

    def _format(self, value) -> str:
        return str(value)

    def _place(self, index, proposed):
        self.values = _schedule.move_timestep(self.values, index, max(proposed, 0), self._ceiling())
        self.dirty = True
        self._notice = ''

    def _nudged(self, index, direction, large):
        return self.values[index] + direction * (10 if large else 1)

    def _snapshot(self):
        return tuple(self.values), self.dirty, tuple(self.base)

    def _restore(self, state):
        values, dirty, base = state
        self.values = list(values)
        self.dirty = dirty
        self.base = list(base)
        self._set_steps_quietly(len(self.values))

    def _parse_typed(self, text):
        values = _schedule.parse_timesteps(text)
        if not self.min_steps <= len(values) <= self.max_steps:
            raise _schedule.SigmaScheduleError(
                f'Enter between {self.min_steps} and {self.max_steps} timesteps.')
        ceiling = self._ceiling()
        if ceiling is not None and values[0] > ceiling:
            raise _schedule.SigmaScheduleError(
                f"The first timestep is above this scheduler's largest, {ceiling}.")
        return values

    def _accept_typed(self, values):
        if len(values) != len(self.values):
            try:
                self.base = _schedule._integers(
                    _schedule.resample([float(value) for value in self.base], len(values)))
            except _schedule.SigmaScheduleError:
                self.base = list(values)
            self._set_steps_quietly(len(values))
        self.values = list(values)
        self.dirty = True
        self._notice = ''


class _TimestepEntry(_entry._Entry):
    NAME = 'timesteps'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)
        self.text_var = tk.StringVar(value=self.config.get('default', '') or '')
        self.label_widget = ttk.Label(self.master, text=self.get_label('Timesteps'), anchor='e')
        self.value_entry = _t_entry.TextEntry(self.master, textvariable=self.text_var)
        buttons = ttk.Frame(self.master)
        ttk.Button(buttons, text='Build', command=self._open).pack(side='left')

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.value_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        buttons.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

    def is_empty(self) -> bool:
        return self.text_var.get().strip() == ''

    def is_valid(self) -> bool:
        text = self.text_var.get().strip()
        if not text or text.startswith('expr:'):
            return True
        try:
            _schedule.parse_timesteps(text)
        except _schedule.SigmaScheduleError:
            return False
        return True

    def invalid(self):
        _entry.invalid_colors(self.value_entry)

    def valid(self):
        _entry.valid_colors(self.value_entry)

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(self.text_var.get().strip()))

    def _inference_steps(self) -> int:
        for entry in getattr(self.form, '_entries', ()) or ():
            if getattr(entry, 'arg', None) != '--inference-steps':
                continue
            try:
                steps = int(float(entry.text_var.get()))
            except (AttributeError, TypeError, ValueError):
                break
            return max(steps, 2)
        return 30

    def _scheduler_widget(self):
        for entry in getattr(self.form, '_entries', ()) or ():
            if getattr(entry, 'arg', None) == '--scheduler' and hasattr(entry, 'plugin_name_var'):
                return entry
        return None

    def _scheduler_flow(self, entry) -> bool:
        slot = getattr(entry, 'entries', {}).get('use-flow-sigmas')
        if slot is None:
            return False
        try:
            return bool(slot[1].get())
        except (tk.TclError, ValueError, TypeError):
            return False

    def _write_scheduler(self, name: str, flow: bool):
        entry = self._scheduler_widget()
        if entry is None or not name:
            return
        with _entry.kept_in_view(self.form, self.value_entry):
            self._set_scheduler(entry, name, flow)

    @staticmethod
    def _set_scheduler(entry, name: str, flow: bool):
        if entry.plugin_name_var.get() != name:
            entry.plugin_name_var.set(name)
            entry._on_plugin_change(name)
        slot = entry.entries.get('use-flow-sigmas')
        if slot is not None and bool(slot[1].get()) != bool(flow):
            slot[1].set(bool(flow))

    def _starting_curve(self, name: str, flow: bool, steps: int,
                        model_type: str | None) -> list[int]:
        if not name:
            default = _schedule.default_scheduler_name(model_type)
            if default in _schedule.timestep_scheduler_names(model_type):
                name = default
        try:
            if name:
                return _schedule.scheduler_timesteps(name, steps, flow, model_type)
        except _schedule.SigmaScheduleError:
            pass
        return _schedule.scheduler_timesteps('EulerDiscreteScheduler', steps, False, model_type)

    def _open(self):
        text = self.text_var.get().strip()
        steps = self._inference_steps()
        scheduler = self._scheduler_widget()
        model_type = _entry.recipe_model_type(self.form)
        names = _entry.offered_schedulers(
            _schedule.timestep_scheduler_names(model_type), scheduler)
        current = ''
        flow = False
        rejected = ''
        if scheduler is not None:
            current = scheduler.plugin_name_var.get().strip()
            flow = self._scheduler_flow(scheduler)
            if current not in names:
                rejected = current
                current = names[0] if current and names else ''
                flow = _schedule.scheduler_requires_flow_sigmas(current)
        try:
            values = _schedule.parse_timesteps(text) if text and not text.startswith('expr:') else None
        except _schedule.SigmaScheduleError:
            values = None
        if values is None:
            values = self._starting_curve(current, flow, steps, model_type)
            steps = len(values)
        parent = self.master.winfo_toplevel()
        if hasattr(self.form, 'unbind_mousewheel'):
            self.form.unbind_mousewheel()

        def accept(new_values, _dirty):
            self.text_var.set(','.join(str(value) for value in new_values))
            # The form still names a scheduler that cannot take timesteps.
            if rejected and current and scheduler.plugin_name_var.get().strip() == rejected:
                self._write_scheduler(current, flow)

        dialog = _TimestepBuildDialog(
            parent,
            self.config.get('label', 'Timesteps'),
            values,
            len(values),
            values,
            names,
            current,
            flow,
            accept,
            self._write_scheduler if scheduler is not None else None,
            model_type)

        def _closed(event):
            if event.widget is dialog and hasattr(self.form, 'bind_mousewheel'):
                self.form.bind_mousewheel()

        dialog.bind('<Destroy>', _closed)
