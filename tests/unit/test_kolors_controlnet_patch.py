import unittest

import dgenerate.extras.kolors.controlnet_patch as _patch


class _States(str):
    def to(self, device):
        return self


class _Weight:
    device = 'cpu'


class _Projection:
    def __init__(self, name):
        self.name = name
        self.weight = _Weight()

    def __call__(self, states):
        return f'{self.name}:{states}'


class _Config:
    encoder_hid_dim_type = 'text_proj'


class _ControlNet:
    def __init__(self, name):
        self.name = name
        self.encoder_hid_proj = _Projection(name)
        self.config = _Config()

    def forward(self, *args, **kwargs):
        return f'{self.name}/{kwargs["encoder_hidden_states"]}/cond={kwargs["controlnet_cond"]}'


class TestKolorsControlNetPatch(unittest.TestCase):
    def test_each_controlnet_keeps_its_own_forward(self):
        pose = _ControlNet('pose')
        canny = _ControlNet('canny')
        originals = [
            _patch.patch_controlnet_text_projection(pose),
            _patch.patch_controlnet_text_projection(canny),
        ]

        self.assertEqual(
            pose.forward(encoder_hidden_states=_States('text'), controlnet_cond='skeleton'),
            'pose/pose:text/cond=skeleton')
        self.assertEqual(
            canny.forward(encoder_hidden_states=_States('text'), controlnet_cond='edges'),
            'canny/canny:text/cond=edges')

        pose.forward = originals[0]
        canny.forward = originals[1]
        self.assertEqual(
            pose.forward(encoder_hidden_states='text', controlnet_cond='skeleton'),
            'pose/text/cond=skeleton')
