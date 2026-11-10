import os
import subprocess
import unittest
from unittest.mock import patch

import dgenerate.msvcenv as msvcenv


class TestMsvcEnv(unittest.TestCase):

    def test_skipped_off_windows(self):
        with patch('dgenerate.msvcenv.os.name', 'posix'), \
                patch('dgenerate.msvcenv.subprocess.run') as run:
            msvcenv.activate_msvc_environment()
        run.assert_not_called()

    def test_skipped_when_cl_is_already_available(self):
        with patch('dgenerate.msvcenv.os.name', 'nt'), \
                patch('dgenerate.msvcenv.shutil.which', return_value=r'C:\cl.exe'), \
                patch('dgenerate.msvcenv.subprocess.run') as run:
            msvcenv.activate_msvc_environment()
        run.assert_not_called()

    def test_vcvars_environment_is_applied(self):
        install = r'C:\VS'
        dumped = 'INCLUDE=C:\\inc\r\nLIB=C:\\lib\r\nPath=C:\\cldir;old\r\n'
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if isinstance(args, list) and str(args[0]).endswith('vswhere.exe'):
                return subprocess.CompletedProcess(args, 0, stdout=install.encode('utf-8'))
            return subprocess.CompletedProcess(
                args, 0, stdout=dumped.encode('utf-16le'))

        saved_include = os.environ.get('INCLUDE')
        saved_lib = os.environ.get('LIB')
        saved_path = os.environ.get('PATH')
        try:
            with patch('dgenerate.msvcenv.os.name', 'nt'), \
                    patch.dict(os.environ, {}, clear=False), \
                    patch('dgenerate.msvcenv.shutil.which', return_value=None), \
                    patch('dgenerate.msvcenv._vswhere_path', return_value=r'C:\vswhere.exe'), \
                    patch('dgenerate.msvcenv.os.path.isfile', return_value=True), \
                    patch('dgenerate.msvcenv.subprocess.run', side_effect=run):
                os.environ.pop('CXX', None)
                msvcenv.activate_msvc_environment()
                self.assertEqual(os.environ['INCLUDE'], r'C:\inc')
                self.assertEqual(os.environ['LIB'], r'C:\lib')
                self.assertTrue(os.environ['PATH'].startswith(r'C:\cldir'))
                self.assertIn('>nul', calls[1])
        finally:
            self._restore('INCLUDE', saved_include)
            self._restore('LIB', saved_lib)
            self._restore('PATH', saved_path)

    @staticmethod
    def _restore(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
