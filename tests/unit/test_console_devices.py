import subprocess
import unittest
from unittest.mock import patch

import dgenerate.console.resources as resources


def _completed(stdout: str, returncode: int = 0, stderr: str = '') -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout.encode(),
        stderr=stderr.encode())


class TestTorchDevices(unittest.TestCase):
    def test_nvidia_indexes_keep_windows_line_endings_out_of_the_name(self):
        def which(name):
            return 'nvidia-smi' if name == 'nvidia-smi' else None

        output = _completed('0\r\n1\r\n')
        with patch.object(resources.platform, 'system', return_value='Windows'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=output), \
                patch.dict(resources.os.environ, {}, clear=True):
            self.assertEqual(
                resources.get_torch_devices(),
                ['cuda:0', 'cuda:1', 'cpu'])

    def test_cuda_visible_devices_renumbers_from_zero(self):
        def which(name):
            return 'nvidia-smi' if name == 'nvidia-smi' else None

        output = _completed('0\n1\n2\n')
        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=output), \
                patch.dict(resources.os.environ, {'CUDA_VISIBLE_DEVICES': '0,2'}, clear=True):
            self.assertEqual(
                resources.get_torch_devices(),
                ['cuda:0', 'cuda:1', 'cpu'])

    def test_cuda_visible_devices_none_hides_gpus(self):
        def which(name):
            return 'nvidia-smi' if name == 'nvidia-smi' else None

        output = _completed('0\n')
        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=output), \
                patch.dict(resources.os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}, clear=True):
            self.assertEqual(resources.get_torch_devices(), ['cpu'])

    def test_intel_mac_does_not_offer_mps(self):
        with patch.object(resources.platform, 'system', return_value='Darwin'), \
                patch.object(resources.platform, 'machine', return_value='x86_64'), \
                patch.object(resources.subprocess, 'run', return_value=_completed('0\n', returncode=1)):
            self.assertEqual(resources.get_torch_devices(), ['cpu'])

    def test_apple_silicon_offers_mps(self):
        with patch.object(resources.platform, 'system', return_value='Darwin'), \
                patch.object(resources.platform, 'machine', return_value='arm64'):
            self.assertEqual(resources.get_torch_devices(), ['mps', 'cpu'])

    def test_hipinfo_device_numbers_are_the_ordinals(self):
        text = """
device#                           0
Name:                             AMD Radeon RX 7900 XTX
gcnArchName:                      gfx1100
device#                           1
Name:                             AMD Radeon Graphics
gcnArchName:                      gfx1150
"""

        def which(name):
            return 'hipinfo' if name == 'hipinfo' else None

        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(text)), \
                patch.dict(resources.os.environ, {}, clear=True):
            self.assertEqual(
                resources.get_torch_devices(),
                ['cuda:0', 'cuda:1', 'cpu'])

    def test_hipinfo_ordinals_are_not_masked_again(self):
        # hipinfo already applied HIP_VISIBLE_DEVICES=0,2 and renumbered.
        text = """
device#                           0
gcnArchName:                      gfx1100
device#                           1
gcnArchName:                      gfx1101
non-peers:                        device#0
"""

        def which(name):
            return 'hipinfo' if name == 'hipinfo' else None

        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(text)), \
                patch.dict(resources.os.environ, {'HIP_VISIBLE_DEVICES': '0,2'}, clear=True):
            self.assertEqual(
                resources.get_torch_devices(),
                ['cuda:0', 'cuda:1', 'cpu'])

    def test_rocminfo_counts_gcn_arch_agents(self):
        text = """
  Name:                    AMD Ryzen
  Device Type:             CPU
  Name:                    gfx1100
  gcnArchName:             gfx1100
  Device Type:             GPU
  Name:                    gfx1101
  gcnArchName:             gfx1101
  Device Type:             GPU
"""

        def which(name):
            return 'rocminfo' if name == 'rocminfo' else None

        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(text)), \
                patch.dict(resources.os.environ, {}, clear=True):
            self.assertEqual(
                resources.get_torch_devices(),
                ['cuda:0', 'cuda:1', 'cpu'])

    def test_current_rocminfo_counts_device_type_gpu_agents(self):
        # Shape from the rocminfo manual's MI300X sample: no gcnArchName field.
        text = """
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx941
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx941:sramecc+:xnack-
"""

        def which(name):
            return 'rocminfo' if name == 'rocminfo' else None

        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(text)), \
                patch.dict(resources.os.environ, {}, clear=True):
            self.assertEqual(resources.get_torch_devices(), ['cuda:0', 'cpu'])

    def test_rocm_smi_product_lines_collapse_to_one_ordinal_per_gpu(self):
        text = """
GPU[0]		: Card series:		AMD Radeon RX 7900 XTX
GPU[0]		: Card model:		0x744c
GPU[0]		: Card vendor:		Advanced Micro Devices, Inc.
GPU[1]		: Card series:		AMD Radeon Graphics
GPU[1]		: Card model:		0x150e
"""

        def which(name):
            return 'rocm-smi' if name == 'rocm-smi' else None

        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(text)), \
                patch.dict(resources.os.environ, {'HIP_VISIBLE_DEVICES': '1'}, clear=True):
            self.assertEqual(resources.get_torch_devices(), ['cuda:0', 'cpu'])

    def _xpu(self, devices: str, mask: str | None):
        def which(name):
            return 'xpu-smi' if name == 'xpu-smi' else None

        env = {} if mask is None else {'ZE_AFFINITY_MASK': mask}
        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', side_effect=which), \
                patch.object(resources.subprocess, 'run', return_value=_completed(devices)), \
                patch.dict(resources.os.environ, env, clear=True):
            return resources.get_torch_devices()

    def test_xpu_empty_and_default_masks_keep_every_device(self):
        devices = """
{"device_list": [
  {"device_id": 0, "device_name": "Intel(R) Arc(TM) B580 Graphics", "pci_bdf_address": "0000:09:00.0"},
  {"device_id": 1, "device_name": "Intel(R) Arc(TM) A770 Graphics", "pci_bdf_address": "0000:0a:00.0"}
]}
"""
        self.assertEqual(self._xpu(devices, ''), ['xpu:0', 'xpu:1', 'cpu'])
        self.assertEqual(self._xpu(devices, 'default'), ['xpu:0', 'xpu:1', 'cpu'])

    def test_xpu_drops_integrated_gpu_when_a_discrete_gpu_is_visible(self):
        devices = """
{"device_list": [
  {"device_id": 0, "device_name": "Intel(R) UHD Graphics 770", "pci_bdf_address": "0000:00:02.0"},
  {"device_id": 1, "device_name": "Intel(R) Arc(TM) B580 Graphics", "pci_bdf_address": "0000:09:00.0"}
]}
"""
        self.assertEqual(self._xpu(devices, None), ['xpu:0', 'cpu'])
        self.assertEqual(self._xpu(devices, '0'), ['xpu:0', 'cpu'])
        self.assertEqual(self._xpu(devices, '0,1'), ['xpu:0', 'cpu'])

    def test_xpu_keeps_an_integrated_gpu_when_it_is_the_only_one(self):
        devices = """
{"device_list": [
  {"device_id": 0, "device_name": "Intel(R) Iris(R) Xe Graphics", "pci_bdf_address": "0000:00:02.0"}
]}
"""
        self.assertEqual(self._xpu(devices, None), ['xpu:0', 'cpu'])

    def test_xpu_affinity_mask_renumbers_in_mask_order(self):
        devices = """
{"device_list": [
  {"device_id": 0, "device_name": "Intel(R) Arc(TM) A770 Graphics", "pci_bdf_address": "0000:09:00.0"},
  {"device_id": 1, "device_name": "Intel(R) Arc(TM) B580 Graphics", "pci_bdf_address": "0000:0a:00.0"},
  {"device_id": 2, "device_name": "Intel(R) Arc(TM) A750 Graphics", "pci_bdf_address": "0000:0b:00.0"}
]}
"""
        self.assertEqual(self._xpu(devices, '2,0'), ['xpu:0', 'xpu:1', 'cpu'])
        self.assertEqual(self._xpu(devices, '-1'), ['cpu'])

    def test_no_vendor_tool_offers_cpu_only(self):
        with patch.object(resources.platform, 'system', return_value='Linux'), \
                patch.object(resources.shutil, 'which', return_value=None):
            self.assertEqual(resources.get_torch_devices(), ['cpu'])


if __name__ == '__main__':
    unittest.main()
