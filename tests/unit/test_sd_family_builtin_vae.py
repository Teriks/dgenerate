import unittest

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat


class TestSdFamilyBuiltinVae(unittest.TestCase):

    def test_checkpoint_has_embedded_vae(self):
        self.assertTrue(_singlefilecompat._sd_family_checkpoint_has_embedded_vae(
            ['first_stage_model.decoder.conv_in.weight']))
        self.assertTrue(_singlefilecompat._sd_family_checkpoint_has_embedded_vae(
            ['vae.encoder.conv_in.weight']))
        self.assertFalse(_singlefilecompat._sd_family_checkpoint_has_embedded_vae(
            ['model.diffusion_model.input_blocks.0.weight']))

    def test_checkpoint_shapes_are_sd2(self):
        key = _singlefilecompat._SD_FAMILY_SD2_CROSS_ATTENTION_KEY
        self.assertTrue(_singlefilecompat._sd_family_checkpoint_shapes_are_sd2(
            {key: (320, 1024)}))
        self.assertFalse(_singlefilecompat._sd_family_checkpoint_shapes_are_sd2(
            {key: (320, 768)}))
        self.assertFalse(_singlefilecompat._sd_family_checkpoint_shapes_are_sd2({}))

    def test_builtin_vae_source(self):
        source = _singlefilecompat._sd_family_builtin_vae_source(_enums.ModelType.SD)
        self.assertEqual(source, ('stabilityai/sd-vae-ft-mse', None))
        source = _singlefilecompat._sd_family_builtin_vae_source(
            _enums.ModelType.UPSCALER_X4)
        self.assertEqual(
            source,
            ('sd2-community/stable-diffusion-2-1', 'vae'))
        source = _singlefilecompat._sd_family_builtin_vae_source(
            _enums.ModelType.SDXL)
        self.assertEqual(source, ('madebyollin/sdxl-vae-fp16-fix', None))
        self.assertIsNone(
            _singlefilecompat._sd_family_builtin_vae_source(_enums.ModelType.FLUX))

    def test_sd_checkpoint_selects_sd2_builtin_vae(self):
        import json
        import os
        import struct
        import tempfile

        key = _singlefilecompat._SD_FAMILY_SD2_CROSS_ATTENTION_KEY
        header = json.dumps({
            key: {'dtype': 'F32', 'shape': [320, 1024], 'data_offsets': [0, 4]},
        }).encode()
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as handle:
            handle.write(struct.pack('<Q', len(header)))
            handle.write(header)
            handle.write(b'\0\0\0\0')
            path = handle.name
        try:
            source = _singlefilecompat._sd_family_builtin_vae_source(
                _enums.ModelType.SD, path, local_files_only=True)
            self.assertEqual(
                source,
                ('sd2-community/stable-diffusion-2-1', 'vae'))
            uri = _singlefilecompat._sd_family_builtin_vae_uri(
                _enums.ModelType.SD, _enums.DataType.FLOAT16,
                path, local_files_only=True)
            self.assertEqual(uri.subfolder, 'vae')
        finally:
            os.remove(path)

    def test_should_load_builtin_vae(self):
        self.assertTrue(_singlefilecompat._sd_family_should_load_builtin_vae(
            None, _enums.ModelType.SDXL, local_files_only=False))
        self.assertFalse(_singlefilecompat._sd_family_should_load_builtin_vae(
            None, _enums.ModelType.SDXL, local_files_only=True))
        self.assertFalse(_singlefilecompat._sd_family_should_load_builtin_vae(
            'AutoencoderKL;model=foo', _enums.ModelType.SD, local_files_only=False))
        self.assertFalse(_singlefilecompat._sd_family_should_load_builtin_vae(
            None, _enums.ModelType.FLUX, local_files_only=False))

    def test_builtin_vae_uri(self):
        uri = _singlefilecompat._sd_family_builtin_vae_uri(
            _enums.ModelType.SD, _enums.DataType.FLOAT16)
        self.assertEqual(uri.encoder, 'AutoencoderKL')
        self.assertEqual(uri.model, 'stabilityai/sd-vae-ft-mse')
        self.assertEqual(uri.dtype, _enums.DataType.FLOAT16)
