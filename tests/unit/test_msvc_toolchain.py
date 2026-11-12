import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_INSTALLER_DIR = Path(__file__).resolve().parents[2] / 'installer'
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

from network_installer.msvc_toolchain import (  # noqa: E402
    VC_TOOLS_WORKLOAD,
    bootstrapper_arguments,
    install_vc_tools,
    supports_msvc_toolchain,
    vc_tools_installed,
)


class TestMsvcToolchain(unittest.TestCase):

    def test_arguments_hide_the_installer_window(self):
        arguments = bootstrapper_arguments()
        self.assertIn('--quiet', arguments)
        self.assertNotIn('--passive', arguments)
        self.assertIn('--wait', arguments)
        self.assertIn(VC_TOOLS_WORKLOAD, arguments)
        self.assertIn('--includeRecommended', arguments)

    def test_toolchain_option_starts_at_6_0_0(self):
        self.assertFalse(supports_msvc_toolchain(None))
        self.assertFalse(supports_msvc_toolchain(''))
        self.assertFalse(supports_msvc_toolchain('5.9.0'))
        self.assertFalse(supports_msvc_toolchain('v5.0.0'))
        self.assertFalse(supports_msvc_toolchain('not-a-version'))
        self.assertTrue(supports_msvc_toolchain('6.0.0'))
        self.assertTrue(supports_msvc_toolchain('v6.0.0'))
        self.assertTrue(supports_msvc_toolchain('6.1.0'))

    def test_xpu_checks_the_compiler_unless_it_is_already_installed(self):
        from network_installer.msvc_toolchain import msvc_option_checked
        self.assertTrue(msvc_option_checked(preset=False, installed=False, intel_xpu=True))
        self.assertFalse(msvc_option_checked(preset=False, installed=False, intel_xpu=False))
        self.assertFalse(msvc_option_checked(preset=True, installed=True, intel_xpu=True))
        self.assertTrue(msvc_option_checked(preset=True, installed=False, intel_xpu=False))

    def test_missing_vswhere_means_the_compiler_is_absent(self):
        with patch('network_installer.msvc_toolchain.os.name', 'nt'), \
                patch('network_installer.msvc_toolchain.vswhere_path', return_value=None):
            self.assertFalse(vc_tools_installed())

    def test_vswhere_install_path_means_the_compiler_is_present(self):
        completed = subprocess.CompletedProcess([], 0, stdout=b'C:\\BuildTools\r\n')
        with patch('network_installer.msvc_toolchain.os.name', 'nt'), \
                patch('network_installer.msvc_toolchain.vswhere_path', return_value=r'C:\vswhere.exe'), \
                patch('network_installer.msvc_toolchain.run_silent', return_value=completed):
            self.assertTrue(vc_tools_installed())

    def test_install_is_windows_only(self):
        with patch('network_installer.msvc_toolchain.os.name', 'posix'):
            ok, error = install_vc_tools(lambda message: None)
        self.assertFalse(ok)
        self.assertIn('Windows', error)

    def test_install_skips_a_machine_that_already_has_the_toolset(self):
        with patch('network_installer.msvc_toolchain.os.name', 'nt'), \
                patch('network_installer.msvc_toolchain.vc_tools_installed', return_value=True), \
                patch('network_installer.msvc_toolchain._download_bootstrapper') as download:
            ok, error = install_vc_tools(lambda message: None)
        self.assertTrue(ok)
        self.assertIsNone(error)
        download.assert_not_called()

    def test_install_accepts_a_reboot_request_when_the_toolset_is_present(self):
        messages = []
        with patch('network_installer.msvc_toolchain.os.name', 'nt'), \
                patch('network_installer.msvc_toolchain.vc_tools_installed', side_effect=[False, True]), \
                patch('network_installer.msvc_toolchain._download_bootstrapper', return_value=None), \
                patch('network_installer.msvc_toolchain.tempfile.mkstemp', return_value=(3, 'vs.exe')), \
                patch('network_installer.msvc_toolchain.os.close'), \
                patch('network_installer.msvc_toolchain.os.remove'), \
                patch('network_installer.msvc_toolchain.run_silent',
                      return_value=subprocess.CompletedProcess([], 3010)) as run:
            ok, error = install_vc_tools(messages.append)
        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertIn('--quiet', run.call_args.args[0])
        self.assertTrue(any('Reboot' in message for message in messages))

    def test_install_reports_a_cancelled_bootstrapper(self):
        with patch('network_installer.msvc_toolchain.os.name', 'nt'), \
                patch('network_installer.msvc_toolchain.vc_tools_installed', return_value=False), \
                patch('network_installer.msvc_toolchain._download_bootstrapper', return_value=None), \
                patch('network_installer.msvc_toolchain.tempfile.mkstemp', return_value=(3, 'vs.exe')), \
                patch('network_installer.msvc_toolchain.os.close'), \
                patch('network_installer.msvc_toolchain.os.remove'), \
                patch('network_installer.msvc_toolchain.run_silent',
                      return_value=subprocess.CompletedProcess([], 1602)):
            ok, error = install_vc_tools(lambda message: None)
        self.assertFalse(ok)
        self.assertIn('1602', error)
