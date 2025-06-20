import sys
import typing
import unittest

import dgenerate.plugin as _plugin


class PluginUnNamed(_plugin.Plugin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PluginNamed(_plugin.Plugin):
    NAMES = 'named'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PluginMultiNamed(_plugin.Plugin):
    NAMES = ['hello', 'hi']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PluginHidden(_plugin.Plugin):
    NAMES = ['hidden']
    HIDDEN = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TestPluginLoading(unittest.TestCase):

    def test_no_name(self):
        loader = _plugin.PluginLoader()

        loader.add_class(PluginUnNamed)

        instance = loader.load('unit.test_plugin_loading.PluginUnNamed')

        self.assertEqual(instance.loaded_by_name,
                         'unit.test_plugin_loading.PluginUnNamed')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module(sys.modules[__name__])

        instance = loader.load('unit.test_plugin_loading.PluginUnNamed')

        self.assertEqual(instance.loaded_by_name,
                         'unit.test_plugin_loading.PluginUnNamed')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module_string(__name__)

        instance = loader.load('unit.test_plugin_loading.PluginUnNamed')

        self.assertEqual(instance.loaded_by_name,
                         'unit.test_plugin_loading.PluginUnNamed')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

    def test_string_name(self):
        loader = _plugin.PluginLoader()

        loader.add_class(PluginNamed)

        instance = loader.load('named')

        self.assertEqual(instance.loaded_by_name, 'named')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module(sys.modules[__name__])

        instance = loader.load('named')

        self.assertEqual(instance.loaded_by_name, 'named')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module_string(__name__)

        instance = loader.load('named')

        self.assertEqual(instance.loaded_by_name, 'named')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

    def test_multi_name(self):
        loader = _plugin.PluginLoader()

        loader.add_class(PluginMultiNamed)

        instance = loader.load('hello')

        self.assertEqual(instance.loaded_by_name, 'hello')

        instance = loader.load('hi')

        self.assertEqual(instance.loaded_by_name, 'hi')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module(sys.modules[__name__])

        instance = loader.load('hello')

        self.assertEqual(instance.loaded_by_name, 'hello')

        instance = loader.load('hi')

        self.assertEqual(instance.loaded_by_name, 'hi')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module_string(__name__)

        instance = loader.load('hello')

        self.assertEqual(instance.loaded_by_name, 'hello')

        instance = loader.load('hi')

        self.assertEqual(instance.loaded_by_name, 'hi')

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('Exception')

    def test_hidden(self):
        loader = _plugin.PluginLoader()

        loader.add_class(PluginHidden)

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('hidden')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module(sys.modules[__name__])

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('hidden')

        # ==============

        loader = _plugin.PluginLoader()

        loader.add_search_module_string(__name__)

        with self.assertRaises(_plugin.PluginNotFoundError):
            loader.load('hidden')

    def test_typed_arguments(self):
        runner = self

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['typed-arguments']

            def __init__(self,
                         arg1,
                         arg2: int,
                         arg3: float,
                         arg4: dict,
                         arg5: set,
                         arg6: list,
                         arg7: str, **kwargs):
                super().__init__(**kwargs)

                runner.assertIsInstance(arg1, PluginTypedArguments.ARG1.__class__)
                runner.assertIsInstance(arg2, int)
                runner.assertIsInstance(arg3, float)
                runner.assertIsInstance(arg4, dict)
                runner.assertIsInstance(arg5, set)
                runner.assertIsInstance(arg6, list)
                runner.assertIsInstance(arg7, str)

                runner.assertEqual(PluginTypedArguments.ARG1, arg1)
                runner.assertEqual(PluginTypedArguments.ARG2, arg2)
                runner.assertEqual(PluginTypedArguments.ARG3, arg3)
                runner.assertDictEqual(PluginTypedArguments.ARG4, arg4)
                runner.assertSetEqual(PluginTypedArguments.ARG5, arg5)
                runner.assertListEqual(PluginTypedArguments.ARG6, arg6)
                runner.assertEqual(PluginTypedArguments.ARG7, arg7)

        loader = _plugin.PluginLoader()

        loader.add_class(PluginTypedArguments)

        PluginTypedArguments.ARG1 = 'any'
        PluginTypedArguments.ARG2 = 1
        PluginTypedArguments.ARG3 = 4.5
        PluginTypedArguments.ARG4 = {'hello': 1}
        PluginTypedArguments.ARG5 = {'hello'}
        PluginTypedArguments.ARG6 = [1, 2, 3]
        PluginTypedArguments.ARG7 = 'string'

        loader.load(
            'typed-arguments;arg1="any";arg2="1";arg3=\'4.5\';arg4={"hello":1};arg5={"hello"};arg6=[1,2,3];arg7=string')

        PluginTypedArguments.ARG1 = ['any']

        loader.load(
            'typed-arguments;arg1=["any"];arg2="1";arg3=\'4.5\';arg4={"hello":1};arg5={"hello"};arg6=[1,2,3];arg7=string')

        with self.assertRaises(_plugin.PluginArgumentError) as e:
            loader.load(
                'typed-arguments;arg1=["any"];arg2=error;arg3=\'4.5\';arg4={"hello":1};arg5={"hello"};arg6=[1,2,3];arg7=string')

        self.assertIn('arg2', str(e.exception))

        with self.assertRaises(_plugin.PluginArgumentError) as e:
            loader.load(
                'typed-arguments;arg1=["any"];arg2=1;arg3=error;arg4={"hello":1};arg5={"hello"};arg6=[1,2,3];arg7=string')

        self.assertIn('arg3', str(e.exception))

        with self.assertRaises(_plugin.PluginArgumentError) as e:
            loader.load(
                'typed-arguments;arg1=["any"];arg2=1;arg3=4.6;arg4=error;arg5={"hello"};arg6=[1,2,3];arg7=string')

        self.assertIn('arg4', str(e.exception))

        with self.assertRaises(_plugin.PluginArgumentError) as e:
            loader.load(
                'typed-arguments;arg1=["any"];arg2=1;arg3=4.6;arg4={"hello":1};arg5=error;arg6=[1,2,3];arg7=string')

        self.assertIn('arg5', str(e.exception))

        with self.assertRaises(_plugin.PluginArgumentError) as e:
            loader.load(
                'typed-arguments;arg1=["any"];arg2=1;arg3=4.6;arg4={"hello":1};arg5={"hello"};arg6=error;arg7=string')

        self.assertIn('arg6', str(e.exception))

    def test_typed_arguments_defaults(self):
        runner = self

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['typed-arguments']

            def __init__(self,
                         arg1='Any',
                         arg2: int = 3,
                         arg3: float = 2.0,
                         arg4: dict = {'dict': 1},
                         arg5: set = {'set'},
                         arg6: list = ['list'],
                         arg7: str = 'str', **kwargs):
                super().__init__(**kwargs)

                runner.assertIsInstance(arg1, PluginTypedArguments.ARG1.__class__)
                runner.assertIsInstance(arg2, int)
                runner.assertIsInstance(arg3, float)
                runner.assertIsInstance(arg4, dict)
                runner.assertIsInstance(arg5, set)
                runner.assertIsInstance(arg6, list)
                runner.assertIsInstance(arg7, str)

                runner.assertEqual(PluginTypedArguments.ARG1, arg1)
                runner.assertEqual(PluginTypedArguments.ARG2, arg2)
                runner.assertEqual(PluginTypedArguments.ARG3, arg3)
                runner.assertDictEqual(PluginTypedArguments.ARG4, arg4)
                runner.assertSetEqual(PluginTypedArguments.ARG5, arg5)
                runner.assertListEqual(PluginTypedArguments.ARG6, arg6)
                runner.assertEqual(PluginTypedArguments.ARG7, arg7)

        loader = _plugin.PluginLoader()

        loader.add_class(PluginTypedArguments)

        def set_defaults():
            PluginTypedArguments.ARG1 = 'Any'
            PluginTypedArguments.ARG2 = 3
            PluginTypedArguments.ARG3 = 2.0
            PluginTypedArguments.ARG4 = {'dict': 1}
            PluginTypedArguments.ARG5 = {'set'}
            PluginTypedArguments.ARG6 = ['list']
            PluginTypedArguments.ARG7 = 'str'

        set_defaults()
        loader.load('typed-arguments')

        set_defaults()
        PluginTypedArguments.ARG1 = 'any'
        loader.load('typed-arguments;arg1="any"')

        set_defaults()
        PluginTypedArguments.ARG2 = 1
        loader.load('typed-arguments;arg2=1')

        set_defaults()
        PluginTypedArguments.ARG3 = 4.5
        loader.load('typed-arguments;arg3=4.5')

        set_defaults()
        PluginTypedArguments.ARG4 = {'hello': 1}
        loader.load('typed-arguments;arg4={"hello": 1}')

        set_defaults()
        PluginTypedArguments.ARG5 = {'hello'}
        loader.load('typed-arguments;arg5={"hello"}')

        set_defaults()
        PluginTypedArguments.ARG6 = [1, 2, 3]
        loader.load('typed-arguments;arg6=[1,2,3]')

        set_defaults()
        PluginTypedArguments.ARG7 = 'string'
        loader.load('typed-arguments;arg7=string')

    def test_union_arguments(self):
        runner = self

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['union-arguments']

            def __init__(self,
                         arg1: str | int,
                         arg2: typing.Union[str, float, None] = None,
                         **kwargs):
                super().__init__(**kwargs)

                runner.assertIsInstance(arg1, int)
                runner.assertIsInstance(arg2, str)

        loader = _plugin.PluginLoader()

        loader.add_class(PluginTypedArguments)

        loader.load('union-arguments;arg1=5;arg2=test')

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['union-arguments']

            def __init__(self,
                         arg1: str | int,
                         arg2: typing.Union[str, int, None] = None,
                         **kwargs):
                super().__init__(**kwargs)

                runner.assertNotIsInstance(arg1, str)
                runner.assertNotIsInstance(arg2, str)

        loader = _plugin.PluginLoader()

        loader.add_class(PluginTypedArguments)

        with self.assertRaises(_plugin.PluginArgumentError):
            loader.load('union-arguments;arg1=5.5')

        with self.assertRaises(_plugin.PluginArgumentError):
            loader.load('union-arguments;arg2=1.5')

        loader.load('union-arguments;arg1=5')

        loader.load('union-arguments;arg1=2;arg2=1')

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['union-arguments']

            def __init__(self,
                         arg1: str | set,
                         arg2: typing.Union[str, set, None],
                         **kwargs):
                super().__init__(**kwargs)

                runner.assertNotIsInstance(arg1, str)
                runner.assertNotIsInstance(arg2, str)

        loader = _plugin.PluginLoader()

        loader.add_class(PluginTypedArguments)

        with self.assertRaises(_plugin.PluginArgumentError):
            loader.load('union-arguments;arg1=4;arg2=None')

        with self.assertRaises(_plugin.PluginArgumentError):
            loader.load('union-arguments;arg1=5;arg2={1,1}')

        loader.load('union-arguments;arg1={1,1};arg2=None')

        loader.load('union-arguments;arg1={1,1};arg2={1,1}')

    def test_typed_arguments_defaults_positionals(self):
        runner = self

        class PluginTypedArguments(_plugin.Plugin):
            NAMES = ['typed-arguments']

            def __init__(self,
                         pos1,
                         arg1='Any',
                         arg2: int = 3,
                         arg3: float = 2.0,
                         arg4: dict = {'dict': 1},
                         arg5: set = {'set'},
                         arg6: list = ['list'],
                         arg7: str = 'str', **kwargs):
                super().__init__(**kwargs)

                pos2 = kwargs.get('pos2')

                runner.assertIsInstance(arg1, PluginTypedArguments.ARG1.__class__)
                runner.assertIsInstance(arg2, int)
                runner.assertIsInstance(arg3, float)
                runner.assertIsInstance(arg4, dict)
                runner.assertIsInstance(arg5, set)
                runner.assertIsInstance(arg6, list)
                runner.assertIsInstance(arg7, str)

                runner.assertEqual(PluginTypedArguments.POS1, pos1)
                runner.assertEqual(PluginTypedArguments.POS2, pos2)
                runner.assertEqual(PluginTypedArguments.ARG1, arg1)
                runner.assertEqual(PluginTypedArguments.ARG1, arg1)
                runner.assertEqual(PluginTypedArguments.ARG2, arg2)
                runner.assertEqual(PluginTypedArguments.ARG3, arg3)
                runner.assertDictEqual(PluginTypedArguments.ARG4, arg4)
                runner.assertSetEqual(PluginTypedArguments.ARG5, arg5)
                runner.assertListEqual(PluginTypedArguments.ARG6, arg6)
                runner.assertEqual(PluginTypedArguments.ARG7, arg7)

        loader = _plugin.PluginLoader(reserved_args=[_plugin.PluginArg('pos2', default='reserved')])

        loader.add_class(PluginTypedArguments)

        def set_defaults():
            PluginTypedArguments.POS1 = 'pos1'
            PluginTypedArguments.POS2 = 'reserved'
            PluginTypedArguments.ARG1 = 'Any'
            PluginTypedArguments.ARG2 = 3
            PluginTypedArguments.ARG3 = 2.0
            PluginTypedArguments.ARG4 = {'dict': 1}
            PluginTypedArguments.ARG5 = {'set'}
            PluginTypedArguments.ARG6 = ['list']
            PluginTypedArguments.ARG7 = 'str'

        # ====

        set_defaults()
        loader.load('typed-arguments;pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG1 = 'any'
        loader.load('typed-arguments;arg1="any";pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG2 = 1
        loader.load('typed-arguments;arg2=1;pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG3 = 4.5
        loader.load('typed-arguments;arg3=4.5;pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG4 = {'hello': 1}
        loader.load('typed-arguments;arg4={"hello": 1};pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG5 = {'hello'}
        loader.load('typed-arguments;arg5={"hello"};pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG6 = [1, 2, 3]
        loader.load('typed-arguments;arg6=[1,2,3];pos1=pos1;pos2="reserved"')

        set_defaults()
        PluginTypedArguments.ARG7 = 'string'
        loader.load('typed-arguments;arg7=string;pos1=pos1;pos2="reserved"')

        # ====

        set_defaults()
        loader.load('typed-arguments', pos1='pos1')
        loader.load('typed-arguments', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;pos2=reserved', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG1 = 'any'
        loader.load('typed-arguments', arg1='any', pos1='pos1')
        loader.load('typed-arguments', arg1='any', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg1="any";pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg1="any"', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg1="any";pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG2 = 1
        loader.load('typed-arguments', arg2=1, pos1='pos1')
        loader.load('typed-arguments', arg2=1, pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg2=1;pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg2=1', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg2=1;pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG3 = 4.5
        loader.load('typed-arguments', arg3=4.5, pos1='pos1')
        loader.load('typed-arguments', arg3=4.5, pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg3=4.5;pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg3=4.5', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg3=4.5;pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG4 = {'hello': 1}
        loader.load('typed-arguments', arg4={"hello": 1}, pos1='pos1')
        loader.load('typed-arguments', arg4={"hello": 1}, pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg4={"hello": 1};pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg4={"hello": 1}', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg4={"hello": 1};pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG5 = {'hello'}
        loader.load('typed-arguments', arg5={'hello'}, pos1='pos1')
        loader.load('typed-arguments', arg5={'hello'}, pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg5={"hello"};pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg5={"hello"}', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg5={"hello"};pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG6 = [1, 2, 3]
        loader.load('typed-arguments', arg6=[1, 2, 3], pos1='pos1')
        loader.load('typed-arguments', arg6=[1, 2, 3], pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg6=[1, 2, 3];pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg6=[1, 2, 3]', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg6=[1, 2, 3];pos2="reserved"', pos1='pos1', pos2='overriden')

        set_defaults()
        PluginTypedArguments.ARG7 = 'string'
        loader.load('typed-arguments', arg7='string', pos1='pos1')
        loader.load('typed-arguments', arg7='string', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg7="string";pos1=pos1;pos2="reserved"')
        loader.load('typed-arguments;arg7=string', pos1='pos1', pos2='reserved')
        loader.load('typed-arguments;arg7=\'string\';pos2="reserved"', pos1='pos1', pos2='overriden')


if __name__ == '__main__':
    unittest.main()
