import os
import unittest

import dgenerate.miopenenv as miopenenv


class TestMiopenEnv(unittest.TestCase):

    def setUp(self):
        self._find = os.environ.get('MIOPEN_FIND_MODE')
        self._log = os.environ.get('MIOPEN_LOG_LEVEL')
        os.environ.pop('MIOPEN_FIND_MODE', None)
        os.environ.pop('MIOPEN_LOG_LEVEL', None)

    def tearDown(self):
        self._restore('MIOPEN_FIND_MODE', self._find)
        self._restore('MIOPEN_LOG_LEVEL', self._log)

    @staticmethod
    def _restore(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_fast_find_and_errors_only(self):
        with miopenenv.applied(False):
            self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '2')
            self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '3')
        self.assertNotIn('MIOPEN_FIND_MODE', os.environ)
        self.assertNotIn('MIOPEN_LOG_LEVEL', os.environ)

    def test_verbose_keeps_warnings(self):
        with miopenenv.applied(True):
            self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '2')
            self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '4')

    def test_existing_values_stay(self):
        os.environ['MIOPEN_FIND_MODE'] = '1'
        os.environ['MIOPEN_LOG_LEVEL'] = '6'
        with miopenenv.applied(False):
            self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '1')
            self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '6')
        self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '1')
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '6')

    def test_restores_after_an_error(self):
        with self.assertRaises(RuntimeError):
            with miopenenv.applied(False):
                raise RuntimeError('stop')
        self.assertNotIn('MIOPEN_FIND_MODE', os.environ)
        self.assertNotIn('MIOPEN_LOG_LEVEL', os.environ)
