import os
import unittest

import dgenerate.miopenenv as miopenenv


class TestMiopenEnv(unittest.TestCase):

    def setUp(self):
        self._find = os.environ.get('MIOPEN_FIND_MODE')
        self._log = os.environ.get('MIOPEN_LOG_LEVEL')
        self._owned = miopenenv._log_level_owned
        os.environ.pop('MIOPEN_FIND_MODE', None)
        os.environ.pop('MIOPEN_LOG_LEVEL', None)
        miopenenv._log_level_owned = False

    def tearDown(self):
        self._restore('MIOPEN_FIND_MODE', self._find)
        self._restore('MIOPEN_LOG_LEVEL', self._log)
        miopenenv._log_level_owned = self._owned

    @staticmethod
    def _restore(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_fast_find_and_errors_only(self):
        miopenenv.configure(False)
        self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '2')
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '3')

    def test_verbose_keeps_warnings(self):
        miopenenv.configure(True)
        self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '2')
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '4')

    def test_existing_values_stay(self):
        os.environ['MIOPEN_FIND_MODE'] = '1'
        os.environ['MIOPEN_LOG_LEVEL'] = '6'
        miopenenv.configure(False)
        self.assertEqual(os.environ['MIOPEN_FIND_MODE'], '1')
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '6')

    def test_owned_log_level_follows_verbose(self):
        miopenenv.configure(False)
        miopenenv.configure(True)
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '4')
        miopenenv.configure(False)
        self.assertEqual(os.environ['MIOPEN_LOG_LEVEL'], '3')

    def test_argv_verbose(self):
        self.assertFalse(miopenenv.argv_is_verbose(['--file', 'a.dgen']))
        self.assertTrue(miopenenv.argv_is_verbose(['-v', '--file', 'a.dgen']))
        self.assertTrue(miopenenv.argv_is_verbose(['--verbose']))
