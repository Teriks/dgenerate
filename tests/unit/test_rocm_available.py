import unittest
from unittest import mock

import torch

import dgenerate.torchutil as torchutil


class TestRocmAvailable(unittest.TestCase):

    def test_hip_is_rocm(self):
        with mock.patch.object(torchutil, 'is_cuda_available', return_value=True), \
                mock.patch.object(torch.version, 'hip', '6.2.0', create=True):
            self.assertTrue(torchutil.is_rocm_available())

    def test_nvidia_cuda_is_not_rocm(self):
        with mock.patch.object(torchutil, 'is_cuda_available', return_value=True), \
                mock.patch.object(torch.version, 'hip', None, create=True):
            self.assertFalse(torchutil.is_rocm_available())

    def test_cpu_is_not_rocm(self):
        with mock.patch.object(torchutil, 'is_cuda_available', return_value=False), \
                mock.patch.object(torch.version, 'hip', '6.2.0', create=True):
            self.assertFalse(torchutil.is_rocm_available())
