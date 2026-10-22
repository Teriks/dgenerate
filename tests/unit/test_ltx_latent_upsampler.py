import unittest
import unittest.mock

import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines.ltx as _ltx
import dgenerate.pipelinewrapper.videopipelines.ltx.components as _ltx_components
import dgenerate.pipelinewrapper.uris.ltxlatentupsampleruri as _uri
from dgenerate.pipelinewrapper.arguments import DiffusionArguments


class TestLTXLatentUpsampler(unittest.TestCase):
    def test_uri_parsing(self):
        parsed = _uri.LTXLatentUpsamplerUri.parse(
            'org/upsampler;revision=v1;variant=fp16;subfolder=latent_upsampler')

        self.assertEqual(parsed.model, 'org/upsampler')
        self.assertEqual(parsed.revision, 'v1')
        self.assertEqual(parsed.variant, 'fp16')
        self.assertEqual(parsed.subfolder, 'latent_upsampler')

    def test_component_weight_detection_accepts_complete_shards(self):
        filenames = {
            'config.json',
            'diffusion_pytorch_model.safetensors.index.json',
            'diffusion_pytorch_model-00001-of-00002.safetensors',
            'diffusion_pytorch_model-00002-of-00002.safetensors',
        }

        self.assertTrue(_ltx_components._ltx_component_has_weights(filenames, None))
        self.assertFalse(_ltx_components._ltx_component_has_weights(
            filenames - {'diffusion_pytorch_model-00002-of-00002.safetensors'}, None))

    def test_older_ltx_rejects_upscale_before_component_probe(self):
        args = DiffusionArguments()
        args.width = 512
        args.height = 512
        args.ltx_latent_upscale = True

        class Wrapper:
            model_path = 'org/ltx-video'
            _subfolder = None
            _revision = None
            _auth_token = None
            _local_files_only = True

        with unittest.mock.patch.object(
                _ltx._putil, 'fetch_model_index_dict',
                return_value={'_class_name': 'LTXPipeline'}), \
                unittest.mock.patch.object(
                    _ltx, '_check_ltx_latent_upsampler_available') as check_component:
            with self.assertRaisesRegex(
                    _pipelines.UnsupportedPipelineConfigError,
                    'earlier LTX pipeline does not support --ltx-latent-upscale'):
                _ltx._preflight_ltx_components(Wrapper(), args)

        check_component.assert_not_called()

    def test_temporal_only_component_is_rejected_before_generation(self):
        args = DiffusionArguments()
        args.ltx_latent_upscale = 'org/ltx;subfolder=temporal_latent_upsampler'

        class Wrapper:
            model_path = 'org/ltx'
            _local_files_only = True
            _auth_token = None

        with unittest.mock.patch.object(_ltx_components, '_check_ltx_component_available'), \
                unittest.mock.patch.object(
                    _ltx_components, '_ltx_latent_upsampler_config',
                    return_value={'spatial_upsample': False, 'temporal_upsample': True}):
            with self.assertRaisesRegex(
                    _pipelines.UnsupportedPipelineConfigError, 'temporal-only'):
                _ltx_components._check_ltx_latent_upsampler_available(Wrapper(), args)


if __name__ == '__main__':
    unittest.main()
