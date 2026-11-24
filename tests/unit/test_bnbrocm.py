import sys
import types
import unittest
from unittest import mock

import dgenerate.pipelinewrapper.bnbrocm as bnbrocm


class MatmulLtState:
    def __init__(self):
        self.threshold = 0.0


class MatMul8bitFp:
    seen = None

    @staticmethod
    def apply(A, B, out, bias, state):
        MatMul8bitFp.seen = (A, B, out, bias, state)
        return 'dequantized'


def _native(A, B, out=None, state=None, threshold=0.0, bias=None):
    return 'native'


class TestGfx950Name(unittest.TestCase):

    def test_arch_token_is_gfx950(self):
        self.assertTrue(bnbrocm._arch_is_gfx950('gfx950'))
        self.assertTrue(bnbrocm._arch_is_gfx950('gfx950:sramecc+:xnack-'))
        self.assertFalse(bnbrocm._arch_is_gfx950('gfx942:sramecc+:xnack-'))
        self.assertFalse(bnbrocm._arch_is_gfx950('gfx1150'))
        self.assertFalse(bnbrocm._arch_is_gfx950(''))


class TestBnbRocm(unittest.TestCase):

    def setUp(self):
        bnbrocm._installed = False
        MatMul8bitFp.seen = None
        self._saved = {
            name: sys.modules.get(name)
            for name in (
                'bitsandbytes',
                'bitsandbytes.autograd',
                'bitsandbytes.autograd._functions',
            )
        }

    def tearDown(self):
        bnbrocm._installed = False
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_fake_bitsandbytes(self):
        package = types.ModuleType('bitsandbytes')
        package.matmul = _native
        autograd = types.ModuleType('bitsandbytes.autograd')
        functions = types.ModuleType('bitsandbytes.autograd._functions')
        functions.matmul = _native
        functions.MatMul8bitFp = MatMul8bitFp
        functions.MatmulLtState = MatmulLtState
        package.autograd = autograd
        sys.modules['bitsandbytes'] = package
        sys.modules['bitsandbytes.autograd'] = autograd
        sys.modules['bitsandbytes.autograd._functions'] = functions
        return package, functions

    def test_cuda_keeps_the_native_matmul(self):
        package, functions = self._install_fake_bitsandbytes()
        with mock.patch('dgenerate.torchutil.is_rocm_available', return_value=False):
            bnbrocm.use_dequantized_int8_matmul()
        self.assertIs(package.matmul, _native)
        self.assertIs(functions.matmul, _native)

    def test_other_rocm_keeps_the_native_matmul(self):
        package, functions = self._install_fake_bitsandbytes()
        with mock.patch('dgenerate.torchutil.is_rocm_available', return_value=True), \
                mock.patch.object(bnbrocm, '_has_gfx950', return_value=False):
            bnbrocm.use_dequantized_int8_matmul()
        self.assertIs(package.matmul, _native)
        self.assertIs(functions.matmul, _native)

    def test_gfx950_uses_the_dequantized_matmul(self):
        package, functions = self._install_fake_bitsandbytes()
        with mock.patch('dgenerate.torchutil.is_rocm_available', return_value=True), \
                mock.patch.object(bnbrocm, '_has_gfx950', return_value=True):
            bnbrocm.use_dequantized_int8_matmul()
            bnbrocm.use_dequantized_int8_matmul()
        state = MatmulLtState()
        self.assertEqual(package.matmul('a', 'b', state=state, threshold=6.0), 'dequantized')
        self.assertEqual(MatMul8bitFp.seen[0], 'a')
        self.assertEqual(MatMul8bitFp.seen[1], 'b')
        self.assertIs(MatMul8bitFp.seen[4], state)
        self.assertEqual(state.threshold, 6.0)
        self.assertIs(functions.matmul, package.matmul)
