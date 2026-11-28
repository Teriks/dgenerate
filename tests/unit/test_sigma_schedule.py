import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None

from dgenerate.console.sigmaschedule import (
    SigmaScheduleError,
    argument,
    clamp_point,
    move_point,
    parse_base,
    resample,
    scheduler_curve,
)

_STAGE = [0.909375, 0.725, 0.421875]


class SigmaScheduleTest(unittest.TestCase):
    def test_base_must_descend(self):
        self.assertEqual(parse_base(_STAGE), _STAGE)
        with self.assertRaises(SigmaScheduleError):
            parse_base([0.2, 0.9])
        with self.assertRaises(SigmaScheduleError):
            parse_base([0.9])

    def test_resample_keeps_the_ends_and_the_order(self):
        self.assertEqual(resample(_STAGE, 3), _STAGE)
        longer = resample(_STAGE, 5)
        self.assertEqual(len(longer), 5)
        self.assertAlmostEqual(longer[0], _STAGE[0])
        self.assertAlmostEqual(longer[-1], _STAGE[-1])
        self.assertTrue(all(left > right for left, right in zip(longer, longer[1:])))

    def test_a_noise_schedule_resamples_in_log_space(self):
        from dgenerate.console.sigmaschedule import resample_sigmas

        noise = [10.0, 1.0, 0.1]
        middle = resample_sigmas(noise, 5)
        self.assertAlmostEqual(middle[1], 10 ** 0.5)
        self.assertAlmostEqual(middle[0], 10.0)
        self.assertAlmostEqual(middle[-1], 0.1)
        self.assertEqual(resample_sigmas(_STAGE, 5), resample(_STAGE, 5))

    def test_typed_sigmas(self):
        from dgenerate.console.sigmaschedule import parse_sigmas, parse_timesteps

        self.assertEqual(parse_sigmas('1.0 0.5,\t0.1'), [1.0, 0.5, 0.1])
        self.assertEqual(parse_timesteps('999 500 , 1'), [999, 500, 1])
        with self.assertRaises(SigmaScheduleError):
            parse_sigmas('14.6,14.6000001,0.1')
        with self.assertRaises(SigmaScheduleError):
            parse_sigmas('1.0,abc')

    def test_axis_ticks(self):
        from dgenerate.console.formentries.curveplot import linear_ticks, log_ticks, tick_label

        self.assertEqual(linear_ticks(1.04), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(log_ticks(0.015, 15.2), [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
        self.assertEqual(log_ticks(0.015, 15.2, most=5), [0.1, 1, 10])
        self.assertEqual(tick_label(0.25), '0.25')
        self.assertEqual(tick_label(1000), '1000')

    def test_a_pipeline_built_list_is_the_curve_for_every_scheduler(self):
        import json
        import re

        from dgenerate.console.sigmaschedule import pipeline_builds_sigmas

        self.assertTrue(pipeline_builds_sigmas('flux3'))
        self.assertFalse(pipeline_builds_sigmas('sd3'))
        with open('dgenerate/console/recipes/flux3-base.recipe', encoding='utf-8') as file:
            text = file.read()
        base = json.loads(re.search(r'@sigmas\[(\{.*?\})\]', text).group(1))['base']
        curve = scheduler_curve('FlowMatchEulerDiscreteScheduler', len(base), model_type='flux3')
        self.assertEqual(len(curve), len(base))
        for left, right in zip(curve, base):
            self.assertAlmostEqual(left, right, places=6)

    def test_the_schema_matches_the_pipelines(self):
        import dgenerate.pipelinewrapper as pipelinewrapper
        from dgenerate.console.sigmaschedule import pipeline_builds_sigmas

        for model_type in pipelinewrapper.supported_model_type_strings():
            self.assertEqual(
                pipeline_builds_sigmas(model_type),
                pipelinewrapper.model_type_uses_flow_default_sigmas(model_type),
                model_type)

    def test_a_flow_match_grid_reproduces_the_default(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        from dgenerate.console.sigmaschedule import _flow_match

        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        scheduler.set_timesteps(10)
        default = [float(value) for value in scheduler.sigmas[:-1]]
        grid = _flow_match({'shift': {'default': 3.0}}, 10, shifted=False)
        scheduler.set_timesteps(sigmas=grid)
        passed = [float(value) for value in scheduler.sigmas[:-1]]
        for left, right in zip(passed, default):
            self.assertAlmostEqual(left, right, places=5)

    def test_a_shifted_checkpoint_curve_reproduces_its_default(self):
        import json
        import pathlib
        import re

        from diffusers import FlowMatchEulerDiscreteScheduler

        import dgenerate.pipelinewrapper as pipelinewrapper

        config = json.loads(pathlib.Path(
            'dgenerate/pipelinewrapper/hub_configs/models--kandinskylab--Kandinsky-6.0-Pro-5s-Diffusers/'
            'scheduler/scheduler_config.json').read_text())
        self.assertEqual(pipelinewrapper.reference_flow_shift('kandinsky6-video'), config['shift'])
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
        scheduler.set_timesteps(50)
        default = [float(value) for value in scheduler.sigmas[:-1]]
        curve = scheduler_curve('FlowMatchEulerDiscreteScheduler', 50, model_type='kandinsky6-video')
        scheduler.set_timesteps(sigmas=curve)
        for left, right in zip((float(value) for value in scheduler.sigmas[:-1]), default):
            self.assertAlmostEqual(left, right, places=5)

        for path in pathlib.Path('dgenerate/console/recipes').glob('kandinsky-6-*-pro-*.recipe'):
            if 'distilled' in path.name:
                continue
            with self.subTest(recipe=path.name):
                text = path.read_text(encoding='utf-8')
                base = json.loads(re.search(r'@sigmas\[(\{.*?\})\]', text).group(1))['base']
                self.assertEqual(base, curve)
                self.assertIn('"arg":"--timesteps"', text)

        piflow = scheduler_curve('PiflowScheduler', 10, model_type='kandinsky6-video')
        self.assertAlmostEqual(piflow[1], 0.977011, places=6)

    def test_video_recipe_bases_reproduce_the_default(self):
        import json
        import pathlib
        import re

        import numpy
        from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

        def recipe_base(name):
            text = pathlib.Path('dgenerate/console/recipes', name).read_text(encoding='utf-8')
            return json.loads(re.search(r'@sigmas\[(\{.*?\})\]', text).group(1))['base']

        for name, scheduler in (
                ('wan-t2v-1.3b-gguf.recipe', UniPCMultistepScheduler(
                    use_flow_sigmas=True, flow_shift=3.0, prediction_type='flow_prediction')),
                ('kandinsky-5-t2v.recipe', FlowMatchEulerDiscreteScheduler(shift=5.0))):
            with self.subTest(recipe=name):
                base = recipe_base(name)
                scheduler.set_timesteps(len(base))
                default = [float(value) for value in scheduler.sigmas[:-1]]
                scheduler.set_timesteps(sigmas=numpy.array(base))
                for left, right in zip((float(value) for value in scheduler.sigmas[:-1]), default):
                    self.assertAlmostEqual(left, right, places=4)

        base = recipe_base('hyvideo-15-t2v.recipe')
        self.assertEqual(base, scheduler_curve('FlowMatchEulerDiscreteScheduler', len(base),
                                               model_type='hyvideo-15'))
        self.assertAlmostEqual(base[-1], 1 / len(base), places=6)

    def test_sd3_timesteps_follow_the_checkpoint_shift(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        from dgenerate.console.sigmaschedule import scheduler_timesteps

        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        scheduler.set_timesteps(28)
        self.assertEqual(
            scheduler_timesteps('FlowMatchEulerDiscreteScheduler', 28, model_type='sd3'),
            [round(float(value)) for value in scheduler.timesteps])

    def test_schedule_fields_in_recipes(self):
        import json
        import pathlib
        import re
        from dgenerate.console.sigmaschedule import timestep_scheduler_names

        both = {'flux', 'flux2', 'flux3', 'sd3', 'z-image', 'krea2', 'kandinsky5-image',
                'kandinsky5-video', 'hyvideo-15', 'ltx-video', 'motif-video', 'wan',
                'wan-animate-2'}
        for path in pathlib.Path('dgenerate/console/recipes').glob('*.recipe'):
            text = path.read_text(encoding='utf-8')
            flag = re.search(r'^--model-type\s+([\w.-]+)', text, re.M)
            if not flag or flag.group(1) not in both:
                continue
            scheduler = re.search(r'@karrasscheduler\[(\{.*?\})\]', text)
            locked = json.loads(scheduler.group(1)).get('filter') if scheduler else None
            takes_timesteps = not locked or bool(
                set(locked) & set(timestep_scheduler_names(flag.group(1))))
            with self.subTest(recipe=path.name):
                self.assertIn('"arg":"--sigmas"', text)
                if takes_timesteps:
                    self.assertIn('"arg":"--timesteps"', text)
                else:
                    self.assertNotIn('"arg":"--timesteps"', text)

    def test_flux3_timesteps_follow_the_pipeline_grid(self):
        from dgenerate.console.sigmaschedule import scheduler_timesteps

        self.assertEqual(
            scheduler_timesteps('FlowMatchEulerDiscreteScheduler', 16, model_type='flux3'),
            [round(1000 * value) for value in scheduler_curve(
                'FlowMatchEulerDiscreteScheduler', 16, model_type='flux3')])

    def test_a_point_cannot_cross_its_neighbors(self):
        values = list(_STAGE)
        self.assertLess(clamp_point(values, 1, 0.99), values[0])
        self.assertGreater(clamp_point(values, 1, 0.01), values[2])

    def test_a_drag_can_push_its_neighbors(self):
        moved = move_point(_STAGE, 1, 0.99)
        self.assertGreater(moved[0], moved[1])
        self.assertGreater(moved[1], moved[2])
        self.assertGreater(moved[1], 0.9)

    def test_argument_text(self):
        self.assertIsNone(argument(_STAGE, _STAGE, 1, False))
        self.assertEqual(argument([value * 0.95 for value in _STAGE], _STAGE, 0.95, False),
                         '0.863906,0.68875,0.400781')
        edited = list(_STAGE)
        edited[1] = 0.6
        self.assertEqual(argument(edited, _STAGE, 1, True), '0.909375,0.6,0.421875')
        longer = resample(_STAGE, 4)
        self.assertEqual(
            argument(longer, _STAGE, 1, False).count(','), 3)

    def test_baked_curves_match_the_scheduler(self):
        from diffusers import (
            EulerDiscreteScheduler,
            FlowMatchEulerDiscreteScheduler,
            UniPCMultistepScheduler,
        )

        def live(scheduler):
            values = []
            for value in scheduler.sigmas:
                number = round(float(value), 6)
                if number <= 0:
                    break
                values.append(number)
            return values

        euler = EulerDiscreteScheduler()
        euler.set_timesteps(30)
        baked = scheduler_curve('EulerDiscreteScheduler', 30)
        self.assertEqual(len(baked), 30)
        self.assertAlmostEqual(baked[0], live(euler)[0], places=4)

        flow = UniPCMultistepScheduler(use_flow_sigmas=True)
        flow.set_timesteps(4)
        baked = scheduler_curve('UniPCMultistepScheduler', 4, True)
        self.assertEqual(baked, live(flow))
        self.assertTrue(all(0 < value <= 1 for value in baked))

        match = FlowMatchEulerDiscreteScheduler()
        match.set_timesteps(8)
        self.assertEqual(scheduler_curve('FlowMatchEulerDiscreteScheduler', 8), live(match))

        with self.assertRaises(SigmaScheduleError):
            scheduler_curve('LCMScheduler', 4)

    def test_noise_curves_use_the_model_betas(self):
        from diffusers import EulerDiscreteScheduler
        from dgenerate.console.sigmaschedule import sigma_ceiling

        for model_type, beta_end, count in (('sd', 0.012, 1000), ('kolors', 0.014, 1100)):
            euler = EulerDiscreteScheduler(
                beta_start=0.00085, beta_end=beta_end,
                beta_schedule='scaled_linear', num_train_timesteps=count)
            euler.set_timesteps(30)
            baked = scheduler_curve('EulerDiscreteScheduler', 30, model_type=model_type)
            self.assertAlmostEqual(baked[0], float(euler.sigmas[0]), places=3)
            self.assertAlmostEqual(sigma_ceiling(model_type), float(euler.sigmas[0]), places=3)

        self.assertIsNone(sigma_ceiling('flux'))
        self.assertIsNone(sigma_ceiling('sd', 'EDMEulerScheduler'))
        self.assertEqual(sigma_ceiling('sd', 'UniPCMultistepScheduler', True), 1.0)
        self.assertEqual(sigma_ceiling('flux', 'FlowMatchEulerDiscreteScheduler'), 1.0)
        self.assertEqual(sigma_ceiling('minimax-h3', 'MiniMaxH3Scheduler'), 1.0)

    def test_timesteps_stop_at_the_training_range(self):
        from dgenerate.console.sigmaschedule import move_timestep, timestep_ceiling

        self.assertEqual(timestep_ceiling('EulerDiscreteScheduler', 'sd'), 999)
        self.assertEqual(timestep_ceiling('EulerDiscreteScheduler', 'kolors'), 1099)
        self.assertEqual(timestep_ceiling('FlowMatchEulerDiscreteScheduler', 'flux'), 1000)
        self.assertEqual(timestep_ceiling(None, 'sdxl'), 999)
        moved = move_timestep([900, 500, 100], 1, 5000, ceiling=999)
        self.assertEqual(moved[0], 999)
        self.assertGreater(moved[0], moved[1])

    def test_a_drag_stops_at_the_ceiling(self):
        curve = [10.0, 5.0, 1.0]
        moved = move_point(curve, 1, 50.0, ceiling=12.0)
        self.assertLessEqual(moved[0], 12.0)
        self.assertGreater(moved[0], moved[1])

    def test_every_recipe_names_a_known_model_type(self):
        import importlib.resources
        import json
        import re
        from dgenerate.console.formentries.entry import recipe_model_type

        with importlib.resources.open_text(
                'dgenerate.console.schemas', 'compatibleschedulers.json') as file:
            known = set(json.load(file))

        class _Form:
            _entries = ()

        recipes = importlib.resources.files('dgenerate.console.recipes')
        checked = 0
        for path in recipes.iterdir():
            if not path.name.endswith('.recipe'):
                continue
            form = _Form()
            form._content = path.read_text(encoding='utf-8')
            model_type = recipe_model_type(form)
            with self.subTest(recipe=path.name):
                self.assertIn(model_type, known)
                flag = re.search(r'--model-type\s+([A-Za-z0-9._-]+)', form._content)
                if flag:
                    self.assertEqual(model_type, flag.group(1))
            checked += 1
        self.assertGreater(checked, 0)


class SigmaEntryTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def _entry(self, **extra):
        from dgenerate.console.formentries.sigmaentry import _SigmaEntry

        frame = tk.Frame(_ROOT)
        self.addCleanup(frame.destroy)
        config = {
            'label': 'Stage Sigmas',
            'arg': '--ltx-stage-sigmas',
            'base': _STAGE,
        }
        config.update(extra)
        return _SigmaEntry(
            form=object(),
            master=frame,
            row=0,
            config=config,
            placeholder='@sigmas')

    def test_the_published_schedule_is_omitted(self):
        entry = self._entry()
        self.assertEqual(entry.template('before @sigmas after'), 'before  after')

    def test_a_scale_is_written_as_numbers(self):
        entry = self._entry()
        entry.set_scale(0.95)
        text = entry.template('@sigmas')
        self.assertIn('--ltx-stage-sigmas', text)
        self.assertIn('0.863906,0.68875,0.400781', text)
        self.assertNotIn('expr:', text)

    def test_a_dragged_point_is_written_as_numbers(self):
        entry = self._entry()
        entry.set_point(1, 0.6)
        text = entry.template('@sigmas')
        self.assertIn('0.909375,0.6,0.421875', text)
        self.assertNotIn('expr:', text)

    def test_a_new_length_resamples_the_published_curve(self):
        entry = self._entry()
        entry.set_steps(5)
        text = entry.template('@sigmas')
        numbers = text.split(' ', 1)[1].strip('"').split(',')
        self.assertEqual(len(numbers), 5)
        values = [float(number) for number in numbers]
        self.assertAlmostEqual(values[0], _STAGE[0])
        self.assertAlmostEqual(values[-1], _STAGE[-1])
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))

    def test_locked_length_ignores_a_step_change(self):
        entry = self._entry(**{'lock-length': True})
        entry.set_steps(6)
        self.assertEqual(entry.template('keep @sigmas here'), 'keep  here')

    def test_build_window_writes_the_field(self):
        import tkinter as tk

        entry = self._entry()
        self.assertEqual(entry.widget_rows, 1)
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        self.assertEqual(str(dialog.insert_button.cget('text')), 'Insert')
        self.assertIn('0.909375', dialog.values_var.get())
        self.assertIn('Point 1 of 3', dialog.point_var.get())
        dialog.scale_var.set('0.95')
        dialog._ok()
        self.assertIn('0.863906,0.68875,0.400781', entry.template('@sigmas'))
        self.assertFalse(dialogs[-1].winfo_exists())

    def test_a_noise_schedule_is_drawn_on_a_log_axis(self):
        from dgenerate.console.formentries.sigmaentry import _SigmaBuildDialog
        from dgenerate.console.sigmaschedule import scheduler_curve

        base = scheduler_curve('EulerDiscreteScheduler', 30, model_type='sdxl')
        dialog = _SigmaBuildDialog(
            _ROOT, 'Sigmas', base, False, 2, 1000, 1.0, 30, base, False,
            lambda *args: None, model_type='sdxl')
        self.addCleanup(dialog.destroy)
        dialog.update()
        self.assertTrue(dialog._log_axis())
        coords, top, pad_y, span_y = dialog._points()
        # On a linear axis the last ten points share a few pixels.
        self.assertGreater(abs(coords[-1][1] - coords[-10][1]), span_y / 4)
        for (_x, y), value in zip(coords, dialog.values):
            self.assertAlmostEqual(dialog._y_to_value(y, top, pad_y, span_y) / value, 1.0, places=6)

        flow = scheduler_curve('UniPCMultistepScheduler', 8, True)
        linear = _SigmaBuildDialog(
            _ROOT, 'Sigmas', flow, False, 2, 1000, 1.0, 8, flow, False, lambda *args: None)
        self.addCleanup(linear.destroy)
        self.assertFalse(linear._log_axis())

    def _dialog(self, base, **kwargs):
        from dgenerate.console.formentries.sigmaentry import _SigmaBuildDialog

        accepted = []
        dialog = _SigmaBuildDialog(
            _ROOT, 'Sigmas', base, False, 2, 1000, 1.0, len(base), base, False,
            lambda *args: accepted.append(args), **kwargs)
        self.addCleanup(lambda: dialog.winfo_exists() and dialog.destroy())
        dialog.update()
        return dialog, accepted

    def test_arrow_keys_nudge_and_ctrl_z_undoes(self):
        dialog, _accepted = self._dialog(list(_STAGE))
        dialog._select_at(1)
        expected = _STAGE[1] + dialog._axis_top() * 0.005
        dialog._nudge(1, False)
        self.assertAlmostEqual(dialog.values[1], expected)
        self.assertTrue(dialog.edited)
        self.assertIn('started at 0.725', dialog.point_var.get())
        dialog.undo()
        self.assertEqual(dialog.values, _STAGE)
        self.assertFalse(dialog.edited)
        dialog.redo()
        self.assertAlmostEqual(dialog.values[1], expected)
        dialog.scale_var.set('0.5')
        self.assertEqual(dialog._undo, [])
        self.assertEqual(dialog._redo, [])

    def test_the_plot_draws_axes_and_the_starting_curve(self):
        dialog, _accepted = self._dialog(list(_STAGE))
        canvas = dialog.canvas

        def texts():
            return {canvas.itemcget(item, 'text') for item in canvas.find_all()
                    if canvas.type(item) == 'text'}

        def dashed_lines():
            return [item for item in canvas.find_all()
                    if canvas.type(item) == 'line' and canvas.itemcget(item, 'dash') == '5 3']

        dialog._draw()
        self.assertTrue({'0', '0.5', '1', '3'} <= texts())
        self.assertEqual(dashed_lines(), [])
        dialog._select_at(1)
        dialog._nudge(-1, True)
        self.assertEqual(len(dashed_lines()), 1)

    def test_dragging_across_1_keeps_the_axis_still(self):
        import types

        dialog, _accepted = self._dialog(list(_STAGE))
        dialog._draw()
        coords, top, pad_y, span_y = dialog._points()
        x, y = coords[0]
        dialog._press(types.SimpleNamespace(x=x, y=y))
        bottom = pad_y + span_y
        ceiling_y = dialog._value_y(1.0, dialog._frame())
        path = [pad_y + span_y * fraction for fraction in (0.0, 0.1, 0.3, 0.5, 0.7, 0.4, 0.05, 0.6)]
        for target in path:
            dialog._drag(types.SimpleNamespace(x=x, y=target))
            placed = dialog._points()[0][0][1]
            # The point lands under the pointer unless the ceiling, a neighbor,
            # or the floor stops it.
            if target < bottom - span_y * 0.25:
                self.assertAlmostEqual(placed, max(target, ceiling_y), delta=1.5)
            self.assertLessEqual(dialog.values[0], 1.0)
        dialog._release()

    def test_a_flow_point_stops_at_1(self):
        import types

        dialog, _accepted = self._dialog(list(_STAGE))
        self.assertEqual(dialog._ceiling(), 1.0)
        coords, _top, pad_y, _span_y = dialog._points()
        x, y = coords[0]
        dialog._press(types.SimpleNamespace(x=x, y=y))
        for _ in range(50):
            dialog._drag(types.SimpleNamespace(x=x, y=pad_y - 200))
        self.assertLessEqual(dialog.values[0], 1.0)
        self.assertLess(dialog._axis_top(), 1.1)
        dialog._release()
        dialog.values_var.set('1.5,0.5,0.1')
        self.assertFalse(dialog._commit_typed())
        self.assertIn('cannot go above 1', dialog.point_var.get())

    def test_a_drag_without_a_ceiling_grows_the_axis_a_little(self):
        import types

        from dgenerate.console.sigmaschedule import scheduler_curve as curve

        base = curve('EDMEulerScheduler', 8)
        dialog, _accepted = self._dialog(base)
        self.assertIsNone(dialog._ceiling())
        coords, _top, pad_y, _span_y = dialog._points()
        x, y = coords[0]
        dialog._press(types.SimpleNamespace(x=x, y=y))
        for _ in range(100):
            dialog._drag(types.SimpleNamespace(x=x, y=pad_y - 200))
        self.assertLessEqual(dialog.values[0], max(base) * 4 + 1e-6)
        dialog._release()

    def test_reset_keeps_the_chosen_scheduler(self):
        from dgenerate.console.sigmaschedule import scheduler_curve as curve

        base = curve('EulerDiscreteScheduler', 8, model_type='sdxl')
        names = ['EulerDiscreteScheduler', 'DPMSolverMultistepScheduler']
        dialog, _accepted = self._dialog(
            base, scheduler_names=names, scheduler_name='EulerDiscreteScheduler', model_type='sdxl')
        dialog.scheduler_var.set('DPMSolverMultistepScheduler')
        dialog._load_scheduler()
        dpm = list(dialog.values)
        dialog.steps_var.set('12')
        dialog.scale_var.set('0.5')
        dialog._select_at(2)
        dialog._nudge(-1, True)
        dialog._reset()
        self.assertEqual(dialog.scheduler_var.get(), 'DPMSolverMultistepScheduler')
        self.assertEqual(dialog.steps_var.get(), '8')
        self.assertEqual(float(dialog.scale_var.get()), 1.0)
        self.assertEqual(dialog.values, dpm)
        self.assertFalse(dialog.edited)

    def test_minimax_video_steps_count_the_terminal_zero(self):
        from dgenerate.console.sigmaschedule import scheduler_curve as curve

        base = curve('MiniMaxH3Scheduler', 29, model_type='minimax-h3-video')
        dialog, accepted = self._dialog(
            base, scheduler_names=['MiniMaxH3Scheduler'], scheduler_name='MiniMaxH3Scheduler',
            model_type='minimax-h3-video')
        self.assertEqual(len(dialog.values), 29)
        self.assertEqual(dialog.steps_var.get(), '30')
        dialog.steps_var.set('20')
        self.assertEqual(len(dialog.values), 19)
        dialog._reset()
        self.assertEqual(dialog.steps_var.get(), '30')
        self.assertEqual(len(dialog.values), 29)

    def test_reset_without_a_new_scheduler_restores_the_recipe(self):
        dialog, _accepted = self._dialog(list(_STAGE))
        dialog._select_at(1)
        dialog._nudge(1, True)
        dialog._reset()
        self.assertEqual(dialog.values, _STAGE)

    def test_typed_values_replace_the_curve(self):
        dialog, accepted = self._dialog(list(_STAGE))
        dialog.values_var.set('0.9 0.8 0.5 0.2')
        dialog._ok()
        values, edited, _scale, steps = accepted[0]
        self.assertEqual(values, [0.9, 0.8, 0.5, 0.2])
        self.assertTrue(edited)
        self.assertEqual(steps, 4)

    def test_invalid_typed_values_keep_the_dialog_open(self):
        dialog, accepted = self._dialog(list(_STAGE))
        dialog.values_var.set('0.5,0.9')
        dialog._ok()
        self.assertEqual(accepted, [])
        self.assertTrue(dialog.winfo_exists())
        self.assertIn('descend', dialog.point_var.get())
        dialog._revert_typed(None)
        self.assertEqual(dialog.values_var.get(), '0.909375, 0.725, 0.421875')
        self.assertNotIn('descend', dialog.point_var.get())

    def test_typed_sigmas_above_the_model_ceiling_are_refused(self):
        base = scheduler_curve('EulerDiscreteScheduler', 4, model_type='sdxl')
        dialog, _accepted = self._dialog(base, model_type='sdxl')
        dialog.values_var.set('20,1,0.1')
        self.assertFalse(dialog._commit_typed())
        self.assertIn('largest sigma', dialog.point_var.get())
        dialog.values_var.set('14,1,0.1')
        self.assertTrue(dialog._commit_typed())
        self.assertEqual(dialog.values, [14.0, 1.0, 0.1])

    def test_rows_added_above_keep_the_field_in_view(self):
        import types

        from dgenerate.console.recipesform import _RecipesForm

        window = tk.Toplevel(_ROOT)
        self.addCleanup(window.destroy)
        window.geometry('300x200+40+40')
        canvas = tk.Canvas(window, height=200)
        canvas.pack(fill='both', expand=True)
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        for index in range(30):
            tk.Label(frame, text=f'row {index}').grid(row=index * 10)
        field = tk.Entry(frame)
        field.grid(row=400)
        window.update()
        canvas.configure(scrollregion=canvas.bbox('all'))
        canvas.yview_moveto(0.8)
        window.update()
        form = types.SimpleNamespace(canvas=canvas, update_idletasks=window.update_idletasks)
        before = field.winfo_rooty()
        with _RecipesForm.keep_in_view(form, field):
            for index in range(6):
                tk.Label(frame, text=f'argument {index}').grid(row=5 + index)
        window.update()
        self.assertAlmostEqual(field.winfo_rooty(), before, delta=2)

    def test_requiring_a_scheduler_keeps_its_label(self):
        import types

        from dgenerate.console.formentries.karrasschedulerentry import _KarrasSchedulerEntry

        frame = tk.Frame(_ROOT)
        self.addCleanup(frame.destroy)
        entry = _KarrasSchedulerEntry(
            form=types.SimpleNamespace(_content='', _entries=[]), master=frame, row=0,
            config={'label': 'Scheduler', 'arg': '--scheduler'}, placeholder='@scheduler')
        self.assertEqual(entry.dropdown_label.cget('text'), '(Optional) Scheduler')
        self.assertFalse(entry.is_empty())
        self.assertFalse(entry.is_active())

        entry.set_name_required(True)
        self.assertEqual(entry.dropdown_label.cget('text'), 'Scheduler')
        self.assertNotIn('', entry.plugin_dropdown['values'])
        self.assertTrue(entry.is_empty())

        entry.set_name_required(False)
        self.assertEqual(entry.dropdown_label.cget('text'), '(Optional) Scheduler')
        self.assertIn('', entry.plugin_dropdown['values'])
        self.assertFalse(entry.is_empty())
        self.assertFalse(entry.optional)

    def test_a_sigma_list_disables_inference_steps(self):
        from dgenerate.console.formentries.intentry import _IntEntry
        from dgenerate.console.recipesform import _RecipesForm

        frame = tk.Frame(_ROOT)
        self.addCleanup(frame.destroy)
        steps = _IntEntry(
            form=object(), master=frame, row=1,
            config={'label': 'Inference Steps', 'arg': '--inference-steps', 'default': '30'},
            placeholder='@int')
        sigmas = self._entry(**{'arg': '--sigmas', 'label': 'Sigmas'})
        stage = self._entry()

        class _Form:
            _entries = [steps, sigmas, stage]

            def _matches_condition(self, spec):
                return True

            def _sync_image_seed_requirement(self):
                pass

            refresh_exclusions = _RecipesForm.refresh_exclusions
            _sync_scheduler_requirement = _RecipesForm._sync_scheduler_requirement

        form = _Form()
        sigmas.form = form
        stage.form = form
        form.refresh_exclusions()
        self.assertFalse(steps._row_inactive)

        sigmas.text_var.set('expr: sigmas * 0.95')
        form.refresh_exclusions()
        self.assertFalse(steps._row_inactive)

        sigmas.text_var.set('1.0,0.5,0.1')
        form.refresh_exclusions()
        self.assertTrue(steps._row_inactive)

        sigmas.text_var.set('')
        stage.text_var.set('0.9,0.7,0.4')
        form.refresh_exclusions()
        self.assertFalse(steps._row_inactive)

    def test_the_scheduler_list_rebuilds_the_curve(self):
        import tkinter as tk

        entry = self._entry()
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        self.addCleanup(dialog.destroy)
        self.assertIsNotNone(dialog.scheduler_var)
        listed = set(dialog.scheduler['values'])
        self.assertIn('EulerDiscreteScheduler', listed)
        self.assertIn('UniPCMultistepScheduler', listed)
        self.assertNotIn('FlowMatchEulerDiscreteScheduler', listed)
        self.assertNotIn('LCMScheduler', listed)
        self.assertNotIn('DDIMScheduler', listed)
        dialog.scheduler_var.set('UniPCMultistepScheduler')
        dialog.flow_var.set(False)
        dialog._load_scheduler()
        self.assertLessEqual(max(dialog.values), 1.0)
        self.assertGreater(dialog.values[0], dialog.values[-1])
        self.assertEqual(dialog.flow_check.grid_info(), {})
        self.assertTrue(dialog.flow_var.get())
        dialog._ok()
        text = entry.template('@sigmas')
        self.assertIn('--ltx-stage-sigmas', text)
        self.assertNotIn('expr:', text)

    def test_a_blank_scheduler_starts_on_the_first_that_accepts_sigmas(self):
        import tkinter as tk

        from dgenerate.console.sigmaschedule import sigma_scheduler_names

        entry = self._entry(**{'arg': '--sigmas', 'label': 'Sigmas'})
        chosen = tk.StringVar(value='')

        class _Scheduler:
            arg = '--scheduler'
            plugin_name_var = chosen
            entries = {}

            def _on_plugin_change(self, _name):
                pass

        class _Form:
            _content = ''
            _entries = None

        scheduler = _Scheduler()
        form = _Form()
        form._entries = [scheduler, entry]
        entry.form = form
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        self.addCleanup(dialog.destroy)
        first = sigma_scheduler_names('sd')[0]
        self.assertEqual(dialog.scheduler_var.get(), first)
        self.assertNotIn('', dialog.scheduler['values'])
        dialog._ok()
        self.assertEqual(chosen.get(), first)


class TimestepEntryTest(unittest.TestCase):
    def setUp(self):
        if _ROOT is None:
            self.skipTest('tkinter is not available')

    def _entry(self):
        from dgenerate.console.formentries.timestepentry import _TimestepEntry

        frame = tk.Frame(_ROOT)
        self.addCleanup(frame.destroy)
        return _TimestepEntry(
            form=object(),
            master=frame,
            row=0,
            config={
                'label': 'Timesteps (override)',
                'arg': '--timesteps',
                'id': 'ts',
                'default': '',
            },
            placeholder='@timesteps')

    def test_baked_timesteps_match_the_scheduler(self):
        from diffusers import EulerDiscreteScheduler
        from dgenerate.console.sigmaschedule import scheduler_timesteps

        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(8)
        live = []
        for value in scheduler.timesteps:
            number = int(round(float(value)))
            if live and number >= live[-1]:
                continue
            live.append(number)
        self.assertEqual(scheduler_timesteps('EulerDiscreteScheduler', 8), live)

    def test_the_lists_leave_out_schedulers_that_cannot_take_one(self):
        from dgenerate.console.sigmaschedule import sigma_scheduler_names, timestep_scheduler_names

        sigmas = set(sigma_scheduler_names())
        timesteps = set(timestep_scheduler_names())
        sd_sigmas = set(sigma_scheduler_names('sd'))
        sd_timesteps = set(timestep_scheduler_names('sd'))
        self.assertIn('EulerDiscreteScheduler', sigmas)
        self.assertIn('UniPCMultistepScheduler', sigmas)
        self.assertIn('FlowMatchEulerDiscreteScheduler', sigmas)
        self.assertNotIn('FlowMatchEulerDiscreteScheduler', sd_sigmas)
        self.assertNotIn('LCMScheduler', sigmas)
        self.assertNotIn('DDIMScheduler', sigmas)
        self.assertNotIn('HeunDiscreteScheduler', sigmas)
        self.assertIn('HeunDiscreteScheduler', timesteps)
        self.assertIn('EulerDiscreteScheduler', timesteps)
        self.assertIn('FlowMatchEulerDiscreteScheduler', timesteps)
        self.assertNotIn('FlowMatchEulerDiscreteScheduler', sd_timesteps)
        self.assertNotIn('UniPCMultistepScheduler', timesteps)
        self.assertNotIn('LCMScheduler', timesteps)

    def test_editor_lists_follow_the_recipe_scheduler_dropdown(self):
        from dgenerate.console.formentries.entry import offered_schedulers
        from dgenerate.console.sigmaschedule import sigma_scheduler_names, timestep_scheduler_names

        class Dropdown:
            def __init__(self, choices):
                self.choices = choices

            def _choice_names(self):
                return list(self.choices)

        unipc_only = Dropdown(['UniPCMultistepScheduler'])
        self.assertEqual(
            offered_schedulers(sigma_scheduler_names('wan'), unipc_only),
            ['UniPCMultistepScheduler'])
        self.assertEqual(offered_schedulers(timestep_scheduler_names('wan'), unipc_only), [])
        self.assertEqual(
            offered_schedulers(sigma_scheduler_names('wan'), None),
            sigma_scheduler_names('wan'))
        self.assertIsNone(offered_schedulers(None, unipc_only))

    def test_a_model_default_that_cannot_take_sigmas_is_in_the_schema(self):
        from dgenerate.console.sigmaschedule import (
            default_scheduler_accepts_sigmas,
            default_scheduler_name,
            sigma_scheduler_names,
        )

        self.assertEqual(default_scheduler_name('sd'), 'PNDMScheduler')
        self.assertFalse(default_scheduler_accepts_sigmas('sd'))
        self.assertIn('EulerDiscreteScheduler', sigma_scheduler_names('sd'))
        self.assertNotIn('PiflowScheduler', sigma_scheduler_names())
        self.assertEqual(default_scheduler_name('flux'), 'FlowMatchEulerDiscreteScheduler')
        self.assertTrue(default_scheduler_accepts_sigmas('flux'))
        self.assertTrue(default_scheduler_accepts_sigmas('sdxl'))

    def test_insert_writes_the_schedule(self):
        entry = self._entry()
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        self.addCleanup(dialog.destroy)
        shown = dialog.values_var.get().replace(' ', '')
        dialog._ok()
        text = entry.template('@timesteps')
        self.assertIn('--timesteps', text)
        self.assertIn(shown.split(',')[0], text)

    def test_a_blank_scheduler_starts_from_the_model_default(self):
        from dgenerate.console.sigmaschedule import scheduler_timesteps

        entry = self._entry()
        self.assertEqual(
            entry._starting_curve('', False, 16, 'flux3'),
            scheduler_timesteps('FlowMatchEulerDiscreteScheduler', 16, model_type='flux3'))
        self.assertEqual(
            entry._starting_curve('', False, 8, 'sd'),
            scheduler_timesteps('EulerDiscreteScheduler', 8, model_type='sd'))

    def test_typed_timesteps_and_nudges(self):
        entry = self._entry()
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        self.addCleanup(dialog.destroy)
        dialog.scheduler_var.set('EulerDiscreteScheduler')
        dialog.values_var.set('5000,10')
        self.assertFalse(dialog._commit_typed())
        self.assertIn('999', dialog.point_var.get())
        dialog.values_var.set('900 500 100')
        self.assertTrue(dialog._commit_typed())
        self.assertEqual(dialog.steps_var.get(), '3')
        dialog._select_at(2)
        dialog._nudge(-1, True)
        self.assertEqual(dialog.values, [900, 500, 90])
        dialog.undo()
        self.assertEqual(dialog.values, [900, 500, 100])

    def test_a_step_change_is_written(self):
        entry = self._entry()
        entry._open()
        dialogs = [
            widget for widget in entry.master.winfo_toplevel().winfo_children()
            if isinstance(widget, tk.Toplevel)]
        dialog = dialogs[-1]
        dialog.steps_var.set('4')
        dialog._ok()
        text = entry.template('@timesteps')
        self.assertIn('--timesteps', text)
        numbers = [int(part) for part in text.split(' ', 1)[1].strip('"').split(',')]
        self.assertEqual(len(numbers), 4)
        self.assertTrue(all(left > right for left, right in zip(numbers, numbers[1:])))
