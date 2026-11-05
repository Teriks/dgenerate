import unittest
from unittest.mock import patch

import dgenerate.devicecache as devicecache
from dgenerate.batchprocess.batchprocessor import BatchProcessError
from dgenerate.batchprocess.configrunner import ConfigRunner
from dgenerate.console.assistantform import release_shell_device_cache


class _Process:
    def __init__(self, alive=True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 1


class _Pipe:
    def __init__(self):
        self.data = b''

    def write(self, data):
        self.data += data


class _Monitor:
    def __init__(self, alive=True):
        self.process = _Process(alive)
        self.stdin_pipe = _Pipe()
        self.shown = []
        self.stdout_callback = self.shown.append


class TestClearDeviceCache(unittest.TestCase):
    def test_no_device_clears_every_device(self):
        seen = []
        methods = vars(devicecache)['__eviction_methods']
        devicecache.register_eviction_method(seen.append)
        try:
            devicecache.clear_device_cache()
        finally:
            methods.remove(seen.append)
        self.assertIn(None, seen)

    def test_directive_with_no_arguments_clears_every_device(self):
        with patch('dgenerate.batchprocess.configrunner._devicecache.clear_device_cache') as clear:
            code = ConfigRunner._clear_device_cache(())
        self.assertEqual(code, 0)
        clear.assert_called_once_with()

    def test_directive_still_rejects_an_unknown_device(self):
        with self.assertRaises(BatchProcessError):
            ConfigRunner._clear_device_cache(('not-a-device',))

    def test_shell_is_told_to_clear_before_the_assistant_starts(self):
        monitor = _Monitor()
        scheduled = []
        done = []

        def schedule(_delay, callback):
            scheduled.append(callback)

        release_shell_device_cache(monitor, schedule, lambda: done.append(True))
        text = monitor.stdin_pipe.data.decode('utf-8')
        self.assertTrue(text.startswith(
            '\\clear_device_cache\n\\clear_object_cache\n\\print dgenerate-device-cache-cleared-'))
        self.assertEqual(done, [])
        marker = text.split('\\print ', 1)[1].splitlines()[0]
        monitor.stdout_callback(b'a model line\n')
        monitor.stdout_callback((marker + '\n').encode('utf-8'))
        self.assertEqual(done, [True])
        self.assertEqual(monitor.shown, [b'a model line\n'])

    def test_a_dead_shell_does_not_block_the_assistant(self):
        monitor = _Monitor(alive=False)
        done = []
        release_shell_device_cache(monitor, lambda *_args: None, lambda: done.append(True))
        self.assertEqual(done, [True])
        self.assertEqual(monitor.stdin_pipe.data, b'')

    def test_a_shell_that_exits_while_waiting_still_starts_the_assistant(self):
        monitor = _Monitor()
        scheduled = []
        done = []
        release_shell_device_cache(
            monitor, lambda _delay, callback: scheduled.append(callback), lambda: done.append(True))
        self.assertEqual(done, [])
        monitor.process.alive = False
        scheduled[0]()
        self.assertEqual(done, [True])
