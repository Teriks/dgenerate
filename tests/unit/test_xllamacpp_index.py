import sys
import unittest
from pathlib import Path

_INSTALLER_DIR = Path(__file__).resolve().parents[2] / 'installer'
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

from network_installer.xllamacppinstall import xllamacpp_index_url  # noqa: E402


class TestXllamaCppIndex(unittest.TestCase):

    def test_macos_keeps_pypi_metal_wheel(self):
        self.assertIsNone(xllamacpp_index_url('Darwin', cuda=(13, 2), has_nvidia=True))

    def test_cuda_version_selects_index(self):
        self.assertTrue(
            xllamacpp_index_url('Windows', cuda=(13, 2), has_nvidia=True).endswith('/cu132')
        )
        self.assertTrue(
            xllamacpp_index_url('Linux', cuda=(12, 8), has_nvidia=True).endswith('/cu128')
        )
        self.assertTrue(
            xllamacpp_index_url('Linux', cuda=(12, 9), has_nvidia=True).endswith('/cu128')
        )
        self.assertTrue(
            xllamacpp_index_url('Windows', cuda=(11, 8), has_nvidia=True).endswith('/vulkan')
        )

    def test_linux_rocm_prefers_rocm_wheels(self):
        self.assertTrue(
            xllamacpp_index_url('Linux', rocm=(7, 2), has_amd=True).endswith('/rocm-7.2.4')
        )
        self.assertTrue(
            xllamacpp_index_url('Linux', rocm=(6, 4), has_amd=True).endswith('/rocm-6.4.1')
        )

    def test_windows_amd_and_intel_use_vulkan(self):
        self.assertTrue(
            xllamacpp_index_url('Windows', rocm=(7, 2), has_amd=True).endswith('/vulkan')
        )
        self.assertTrue(
            xllamacpp_index_url('Windows', has_intel=True).endswith('/vulkan')
        )

    def test_nvidia_without_driver_version_uses_cu128(self):
        self.assertTrue(
            xllamacpp_index_url('Windows', has_nvidia=True).endswith('/cu128')
        )

    def test_legacy_nvidia_uses_vulkan(self):
        self.assertTrue(
            xllamacpp_index_url('Windows', cuda=(12, 9), has_nvidia=True, nvidia_legacy=True).endswith('/vulkan')
        )

    def test_cpu_only_keeps_pypi_wheel(self):
        self.assertIsNone(xllamacpp_index_url('Linux'))
        self.assertIsNone(xllamacpp_index_url('Windows'))


if __name__ == '__main__':
    unittest.main()
