import unittest

from dgenerate.transpile import transpile


class TestTranspileRelaxed(unittest.TestCase):

    def test_invocation_calls_render_loop(self):
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts "a cat"\n'
        )
        output = transpile(source, relaxed=True)
        self.assertIn('from dgenerate.renderloop import RenderLoop', output)
        self.assertIn('from dgenerate.renderloopconfig import RenderLoopConfig', output)
        self.assertIn('from dgenerate.prompt import Prompt', output)
        self.assertNotIn('transpile.runtime import', output)
        self.assertIn('render_loop: RenderLoop = RenderLoop()', output)
        self.assertIn('config: RenderLoopConfig = RenderLoopConfig()', output)
        self.assertIn("Prompt.parse('a cat')", output)
        self.assertIn('render_loop.config = config', output)
        self.assertIn('render_loop.run()', output)
        self.assertNotIn('script.run', output)
        self.assertNotIn('script.invoke', output)
        self.assertNotIn('Script()', output)

    def test_directive_shares_the_scripts_render_loop(self):
        source = (
            '\\set name cat\n'
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts "{{ name }}"\n'
        )
        output = transpile(source, relaxed=True)
        self.assertIn('from dgenerate.transpile.runtime import Script', output)
        self.assertIn('from dgenerate.renderloop import RenderLoop', output)
        self.assertNotIn('runtime import Script, RenderLoop', output)
        self.assertNotIn('runtime import RenderLoop', output)
        self.assertIn('script = Script()', output)
        self.assertIn('render_loop: RenderLoop = script.render_loop', output)
        self.assertIn('render_loop.run()', output)
        self.assertIn('script.sync()', output)
        self.assertNotIn('render_loop: RenderLoop = RenderLoop()', output)

    def test_builtins_come_from_their_module(self):
        source = (
            '{% for n in [1] %}\n'
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts "{{ quote(first([n])) }}"\n'
            '{% endfor %}\n'
        )
        for relaxed in (False, True):
            output = transpile(source, relaxed=relaxed)
            self.assertIn(
                'from dgenerate.batchprocess.configrunnerbuiltins import first, quote',
                output,
            )
            self.assertNotIn('runtime import first', output)
            self.assertNotIn('runtime import quote', output)
            self.assertNotRegex(output, r'runtime import [^\\n]*\b(first|quote)\b')

    def test_strict_still_uses_script_run(self):
        source = 'sd2-community/stable-diffusion-2-1 --prompts "a cat"\n'
        output = transpile(source)
        self.assertIn('from dgenerate.arguments import DgenerateArguments', output)
        self.assertIn('from dgenerate.prompt import Prompt', output)
        self.assertIn('script.run(config)', output)
        self.assertIn('DgenerateArguments', output)
        self.assertNotIn('render_loop.run()', output)

    def test_relaxed_invocation_context(self):
        source = (
            'sd2-community/stable-diffusion-2-1 --prompts "a cat" '
            '--plugin-modules plugins/extra.py -v\n'
        )
        output = transpile(source, relaxed=True)
        self.assertIn(
            "with script.options(plugin_module_paths=['plugins/extra.py'], verbose=True):",
            output,
        )
        self.assertIn('render_loop.run()', output)
        self.assertNotIn('config.verbose', output)
        self.assertNotIn('config.plugin_module_paths', output)
        plain = transpile('sd2-community/stable-diffusion-2-1 --prompts "a cat"\n', relaxed=True)
        self.assertNotIn('script.options', plain)

    def test_strict_keeps_plugin_and_verbose_on_the_arguments(self):
        source = (
            'sd2-community/stable-diffusion-2-1 --prompts "a cat" '
            '--plugin-modules plugins/extra.py -v\n'
        )
        output = transpile(source)
        self.assertIn("config.plugin_module_paths = ['plugins/extra.py']", output)
        self.assertIn('config.verbose = True', output)
        self.assertIn('script.run(config)', output)
        self.assertNotIn('script.options', output)

    def test_invocation_restores_log_level_and_plugin_paths(self):
        import dgenerate.messages as messages
        import dgenerate.plugin as plugin
        from dgenerate.transpile.runtime import Script

        previous_level = messages.LEVEL
        previous_paths = set(plugin.PLUGIN_PATHS)
        script = Script()
        try:
            with script.options(verbose=True, plugin_module_paths=['plugins/extra.py']):
                self.assertEqual(messages.LEVEL, messages.DEBUG)
                self.assertIn('plugins/extra.py', plugin.PLUGIN_PATHS)
            self.assertEqual(messages.LEVEL, previous_level)
            self.assertEqual(set(plugin.PLUGIN_PATHS), previous_paths)
        finally:
            messages.LEVEL = previous_level
            plugin.PLUGIN_PATHS.clear()
            plugin.PLUGIN_PATHS.update(previous_paths)


if __name__ == '__main__':
    unittest.main()
