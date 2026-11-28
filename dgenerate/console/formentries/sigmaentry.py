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

import math
import re
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.formentries.curveplot as _curveplot
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.sigmaschedule as _schedule
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
from dgenerate.console.spinbox import FloatSpinbox, IntSpinbox

_EXPRESSION = re.compile(r'expr:\s*sigmas\s*\*\s*([0-9]*\.?[0-9]+)\s*$')


class _SigmaBuildDialog(_curveplot.CurvePlot, tk.Toplevel):
    """Editor for one sigma schedule. Nothing is written until Insert."""

    def __init__(self, master, title, base, lock_length, min_steps, max_steps,
                 scale, steps, values, edited, on_accept,
                 scheduler_names=None, scheduler_name='', use_flow_sigmas=False,
                 on_scheduler=None, require_scheduler=False, model_type=None):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self._init_plot()
        self._width = int(560 * self._px)
        self.model_type = model_type
        # The Steps box matches --inference-steps. The values exclude a counted terminal 0.
        self._terminal_steps = _schedule.terminal_steps(model_type)
        self.base = list(base)
        self.recipe_base = list(base)
        self.lock_length = lock_length
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.values = list(values)
        self.edited = edited
        self.scheduler_names = list(scheduler_names or [])
        self._on_scheduler = on_scheduler
        self._require_scheduler = bool(require_scheduler) and bool(self.scheduler_names)
        self._opened_name = scheduler_name
        self._opened_flow = bool(use_flow_sigmas)
        self._using_formula = False
        self._notice = ''
        self._on_accept = on_accept
        self._ready = False

        self.scale_var = tk.StringVar(value=_schedule.format_number(scale))
        self.steps_var = tk.StringVar(value=str(steps + self._terminal_steps))

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
            if (_schedule.scheduler_requires_flow_sigmas(scheduler_name)
                    or not _schedule.scheduler_supports_flow(scheduler_name)):
                self.flow_check.grid_remove()
            if _schedule.scheduler_requires_flow_sigmas(scheduler_name):
                self.flow_var.set(True)
            row += 1

        ttk.Label(body, text='Scale', anchor='e').grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.scale = FloatSpinbox(
            body, from_=0.05, to=2.0, increment=0.05, textvariable=self.scale_var)
        self.scale.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.scale.create_spin_buttons(body).grid(row=row, column=2, sticky='w')
        row += 1

        if not lock_length:
            ttk.Label(body, text='Steps', anchor='e').grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
            self.steps = IntSpinbox(
                body, from_=min_steps + self._terminal_steps,
                to=max_steps + self._terminal_steps, textvariable=self.steps_var)
            self.steps.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
            self.steps.create_spin_buttons(body).grid(row=row, column=2, sticky='w')
            row += 1
        else:
            self.steps = None

        row = self._build_plot(body, row)

        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=3, pady=(6, 0), sticky='e')
        ttk.Button(buttons, text='Reset', command=self._reset).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='Cancel', command=self.destroy).pack(side='left')
        self.insert_button = ttk.Button(buttons, text='Insert', command=self._ok)
        self.insert_button.pack(side='left', padx=(8, 0))
        self._sync_insert()

        self.scale_var.trace_add('write', lambda *_args: self._on_scale())
        self.steps_var.trace_add('write', lambda *_args: self._on_steps())
        self._ready = True
        self._sync_values()
        self._show_point(0)
        if self._require_scheduler and not self.edited:
            self._load_scheduler()
        self._sync_scale_limit()
        self.bind('<Escape>', lambda _event: self.destroy())
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        import dgenerate.console.appearance as _appearance
        _appearance.install(self)
        self.update_idletasks()
        _util.position_toplevel(master, self, size=(self._width, self.winfo_reqheight()))
        self.minsize(int(420 * self._px), self.winfo_reqheight())
        self.grab_set()

    def _ceiling(self) -> float | None:
        name = self.scheduler_var.get().strip() if self.scheduler_var is not None else ''
        flow = bool(self.flow_var.get()) if self.flow_var is not None else False
        ceiling = _schedule.sigma_ceiling(self.model_type, name or None, flow)
        if ceiling is None and not name and not _schedule.is_noise_range(self.recipe_base):
            # A recipe curve inside (0, 1] is a flow schedule.
            return 1.0
        return ceiling

    def _drag_limit(self) -> float:
        peak = max(list(self.base) + list(self.values)) if self.values else 1.0
        return max(max(self.base, default=1.0) * 4, peak)

    def _max_scale(self) -> float:
        ceiling = self._ceiling()
        peak = max(self.base) if self.base else 0.0
        if ceiling is None or peak <= 0:
            return 2.0
        return max(0.05, min(2.0, math.floor(ceiling / peak * 100 + 1e-6) / 100))

    def _sync_scale_limit(self):
        limit = self._max_scale()
        self.scale.to = limit
        try:
            typed = float(self.scale_var.get())
        except ValueError:
            return
        if typed > limit:
            # Writing the spinbox variable from inside its own trace corrupts
            # the displayed text, so the correction waits for the next idle.
            self.after_idle(lambda: self.scale_var.set(_schedule.format_number(limit)))

    def _apply_scale(self):
        self._sync_scale_limit()
        scale = self._scale()
        if scale is None:
            return
        self._forget_history()
        self.values = [value * scale for value in self.base]
        self.edited = False
        self._sync_values()
        self._show_point(self._selected)
        self._draw()

    def _on_scale(self):
        if not self._ready:
            return
        self._apply_scale()

    def _on_steps(self):
        if not self._ready:
            return
        steps = self._steps()
        if steps is None:
            return
        if self._using_formula:
            self._load_scheduler()
            return
        self.base = _schedule.resample_sigmas(self.recipe_base, steps)
        self._apply_scale()

    def _set_steps_quietly(self, steps: int):
        if self.lock_length:
            return
        self._ready = False
        self.steps_var.set(str(steps + self._terminal_steps))
        self._ready = True

    def _sync_flow_check(self, name: str):
        if self.flow_check is None:
            return
        shown = bool(self.flow_check.grid_info())
        if _schedule.scheduler_requires_flow_sigmas(name):
            self.flow_var.set(True)
            self.flow_check.grid_remove()
        elif _schedule.scheduler_supports_flow(name):
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
            self._sync_insert()
            return
        if _schedule.scheduler_requires_flow_sigmas(name):
            flow = True
        else:
            flow = bool(self.flow_var.get()) if self.flow_var is not None else False
        self._sync_flow_check(name)
        try:
            curve = _schedule.scheduler_curve(name, steps, flow, self.model_type)
        except _schedule.SigmaScheduleError as error:
            self._notice = str(error)
            self._show_point(None)
            self._sync_insert()
            return
        self._notice = ''
        self._using_formula = True
        self.base = curve
        if len(curve) != steps:
            self._set_steps_quietly(len(curve))
        self._apply_scale()
        self._sync_insert()

    def _sync_insert(self):
        if not hasattr(self, 'insert_button'):
            return
        if not self._require_scheduler or self.scheduler_var is None:
            self.insert_button.configure(state='normal')
            return
        state = 'normal' if self.scheduler_var.get().strip() else 'disabled'
        self.insert_button.configure(state=state)

    def _reset(self):
        """Undo edits, scale, and steps. The chosen scheduler stays."""
        self.edited = False
        self._notice = ''
        self._selected = None
        self._ready = False
        self.scale_var.set('1')
        self.steps_var.set(str(len(self.recipe_base) + self._terminal_steps))
        self._ready = True
        self._forget_history()
        if self._using_formula:
            self._load_scheduler()
            if not self._notice:
                return
            self._using_formula = False
            self._set_steps_quietly(len(self.recipe_base))
        self.base = list(self.recipe_base)
        self.values = list(self.recipe_base)
        self._sync_scale_limit()
        self._sync_values()
        self._show_point(0)
        self._draw()
        self._sync_insert()

    def _scale(self) -> float | None:
        try:
            scale = float(self.scale_var.get())
        except ValueError:
            return None
        if scale < 0.05:
            return None
        return min(scale, self._max_scale())

    def _steps(self) -> int | None:
        if self.lock_length:
            return len(self.base)
        try:
            steps = int(float(self.steps_var.get())) - self._terminal_steps
        except ValueError:
            return None
        if steps < self.min_steps or steps > self.max_steps:
            return None
        return steps

    def _ok(self):
        if not self._commit_typed():
            self.values_entry.focus_set()
            return
        scale = self._scale()
        steps = self._steps()
        if scale is None or steps is None:
            return
        name = self.scheduler_var.get().strip() if self.scheduler_var is not None else ''
        if self._require_scheduler and not name:
            return
        if self._on_scheduler is not None and name and (
                self._using_formula or self._require_scheduler):
            flow = bool(self.flow_var.get()) if self.flow_var is not None else False
            # A preselected scheduler still has to be written. The form field
            # was blank, and the model default cannot take this sigma list.
            if (self._require_scheduler or name != self._opened_name
                    or flow != self._opened_flow):
                self._on_scheduler(name, flow)
        self._on_accept(self.values, self.edited, scale, steps)
        self.destroy()

    def _axis_top(self) -> float:
        if self._drag_top is not None:
            return self._drag_top
        # The scale control runs up to 2. Keep that much room above the
        # unscaled schedule so raising the scale lifts the curve. Fitting
        # the axis to the scaled peak redraws the same shape.
        reference = max(self.base) if self.base else 1.0
        room = max(reference, 1e-6) * 2
        ceiling = self._ceiling()
        if ceiling is not None:
            room = min(room, ceiling)
        peak = max(self.values) if self.values else reference
        return max(room, peak) * 1.04

    def _wants_log_axis(self) -> bool:
        # Noise and EDM schedules start well above 1 and spend most of their
        # steps below 1. On a linear axis those points sit on the floor.
        # Only the starting curve decides, so dragging a point cannot flip it.
        return _schedule.is_noise_range(self.base)

    def _axis_bottom(self) -> float:
        if self._drag_bottom is not None:
            return self._drag_bottom
        values = [value for value in list(self.base) + list(self.values) if value > 0]
        low = min(values) if values else 1e-3
        return max(low / 2, 1e-6)

    def _reference(self):
        return self.base

    def _format(self, value) -> str:
        return _schedule.format_number(value)

    def _place(self, index, proposed):
        self.values = _schedule.move_point(self.values, index, proposed, self._ceiling())
        self.edited = True

    def _nudged(self, index, direction, large):
        value = self.values[index]
        if self._log_axis():
            return value * (1.1 if large else 1.01) ** direction
        return value + direction * self._axis_top() * (0.05 if large else 0.005)

    def _snapshot(self):
        return tuple(self.values), self.edited, tuple(self.base)

    def _restore(self, state):
        values, edited, base = state
        self.values = list(values)
        self.edited = edited
        self.base = list(base)
        self._set_steps_quietly(len(self.values))

    def _parse_typed(self, text):
        values = _schedule.parse_sigmas(text)
        if self.lock_length and len(values) != len(self.base):
            raise _schedule.SigmaScheduleError(
                f'This schedule takes exactly {len(self.base)} values.')
        if not self.lock_length and not self.min_steps <= len(values) <= self.max_steps:
            raise _schedule.SigmaScheduleError(
                f'Enter between {self.min_steps} and {self.max_steps} values.')
        ceiling = self._ceiling()
        if ceiling is not None and values[0] > ceiling * (1 + 1e-6):
            if ceiling == 1.0:
                raise _schedule.SigmaScheduleError(
                    'Flow sigmas are a share of noise and cannot go above 1.')
            raise _schedule.SigmaScheduleError(
                f"The first value is above this model's largest sigma, "
                f'{_schedule.format_number(ceiling)}.')
        return values

    def _accept_typed(self, values):
        if len(values) != len(self.values):
            self.base = _schedule.resample_sigmas(self.base, len(values))
            self._set_steps_quietly(len(values))
        self.values = list(values)
        self.edited = True


class _SigmaEntry(_entry._Entry):
    NAME = 'sigmas'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)
        try:
            self.base = _schedule.parse_base(self.config.get('base'))
        except _schedule.SigmaScheduleError as error:
            raise RuntimeError(f'Sigma field {self.arg or ""}: {error}') from error

        self.lock_length = bool(self.config.get('lock-length', False))
        self.min_steps = len(self.base) if self.lock_length else int(self.config.get('min-steps', 2))
        if self.lock_length:
            self.max_steps = len(self.base)
        else:
            # The spinner needs an upper bound. A recipe can set a lower one with max-steps.
            configured_max = self.config.get('max-steps', 1000)
            self.max_steps = max(int(configured_max), len(self.base))

        self.values = list(self.base)
        self.edited = False
        self.scale = 1.0
        self.steps = len(self.base)
        self.text_var = tk.StringVar(value='')

        self.label_widget = ttk.Label(
            self.master, text=self.get_label('Sigmas'), anchor='e')
        self.value_entry = _t_entry.TextEntry(self.master, textvariable=self.text_var)
        buttons = ttk.Frame(self.master)
        ttk.Button(buttons, text='Build', command=self._open).pack(side='left')

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.value_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        buttons.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

    def set_scale(self, scale: float):
        self.scale = float(scale)
        steps = len(self.base) if self.lock_length else self.steps
        self.steps = steps
        self.values = [value * self.scale for value in _schedule.resample_sigmas(self.base, steps)]
        self.edited = False
        self._publish()

    def set_steps(self, steps: int):
        if self.lock_length:
            return
        self.steps = int(steps)
        self.values = [value * self.scale for value in _schedule.resample_sigmas(self.base, self.steps)]
        self.edited = False
        self._publish()

    def set_point(self, index: int, value: float):
        self.values[index] = _schedule.clamp_point(self.values, index, float(value))
        self.edited = True
        self._publish()

    def is_empty(self) -> bool:
        return self.text_var.get().strip() == ''

    def exclude_targets(self):
        # A concrete list is the step count. A scale expression still
        # starts from the steps field, so that field stays available.
        # Stage sigmas are a later pass and do not replace it.
        steps_arg = {
            '--sigmas': '--inference-steps',
            '--sdxl-refiner-sigmas': '--second-model-inference-steps',
        }.get(self.arg)
        value = self.text_var.get().strip()
        if steps_arg is None or not value or value.startswith('expr:'):
            return ()
        entries = getattr(self.form, '_entries', None) or ()
        return tuple(
            entry for entry in entries
            if getattr(entry, 'arg', None) == steps_arg)

    def is_valid(self) -> bool:
        return True

    def invalid(self):
        _entry.invalid_colors(self.value_entry)

    def valid(self):
        _entry.valid_colors(self.value_entry)

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(self.text_var.get().strip()))

    def _publish(self):
        text = _schedule.argument(self.values, self.base, self.scale, self.edited)
        self.text_var.set('' if text is None else text)

    def _seed_from_text(self):
        text = self.text_var.get().strip()
        if not text:
            return list(self.base), False, 1.0, len(self.base)
        expression = _EXPRESSION.fullmatch(text)
        if expression:
            scale = float(expression.group(1))
            values = [value * scale for value in self.base]
            return values, False, scale, len(self.base)
        try:
            values = _schedule.parse_base(_schedule.split_values(text))
        except _schedule.SigmaScheduleError:
            return list(self.base), False, 1.0, len(self.base)
        return values, True, 1.0, len(values)

    def _scheduler_widget(self):
        entries = getattr(self.form, '_entries', None)
        if not entries:
            return None
        wanted = '--second-model-scheduler' if self.arg == '--sdxl-refiner-sigmas' else '--scheduler'
        for entry in entries:
            if getattr(entry, 'arg', None) == wanted and hasattr(entry, 'plugin_name_var'):
                return entry
        return None

    def _model_type(self) -> str | None:
        # The refiner stage is always an SDXL refiner, also under Kolors.
        if self.arg == '--sdxl-refiner-sigmas':
            return 'sdxl'
        return _entry.recipe_model_type(self.form)

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

    def _open(self):
        values, edited, scale, steps = self._seed_from_text()
        parent = self.master.winfo_toplevel()
        if hasattr(self.form, 'unbind_mousewheel'):
            self.form.unbind_mousewheel()

        def accept(new_values, new_edited, new_scale, new_steps):
            self.values = list(new_values)
            self.scale = new_scale
            self.steps = len(new_values)
            reference = [value * new_scale for value in _schedule.resample_sigmas(self.base, len(new_values))]
            same = (len(reference) == len(new_values) and
                    all(abs(left - right) < 1e-4 for left, right in zip(reference, new_values)))
            self.edited = new_edited or not same
            self._publish()
            if self.is_empty():
                self.edited = True
                self._publish()

        scheduler = self._scheduler_widget()
        model_type = self._model_type()
        names = None if self.lock_length else _entry.offered_schedulers(
            _schedule.sigma_scheduler_names(model_type), scheduler)
        current = ''
        flow = False
        rejected = False
        if scheduler is not None and names:
            current = scheduler.plugin_name_var.get().strip()
            flow = self._scheduler_flow(scheduler)
            if current not in names:
                rejected = bool(current)
                current = ''

        require_scheduler = (
            scheduler is not None and bool(names) and not current and
            (rejected or not _schedule.default_scheduler_accepts_sigmas(model_type)))
        if require_scheduler:
            current = names[0]
            if _schedule.scheduler_requires_flow_sigmas(current):
                flow = True

        dialog = _SigmaBuildDialog(
            parent,
            self.config.get('label', 'Sigmas'),
            self.base,
            self.lock_length,
            self.min_steps,
            self.max_steps,
            scale,
            steps,
            values,
            edited,
            accept,
            names,
            current,
            flow,
            self._write_scheduler if scheduler is not None else None,
            require_scheduler,
            model_type)

        def _closed(event):
            if event.widget is dialog and hasattr(self.form, 'bind_mousewheel'):
                self.form.bind_mousewheel()

        dialog.bind('<Destroy>', _closed)
