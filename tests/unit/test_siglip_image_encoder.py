import unittest
import unittest.mock as mock

import torch

from dgenerate.pipelinewrapper.models import SiglipImageEncoder


class TestSiglipImageEncoder(unittest.TestCase):

    def test_from_pretrained_accepts_dtype(self):
        fake_encoder = mock.Mock()
        fake_processor = mock.Mock()

        with mock.patch(
                'transformers.SiglipVisionModel.from_pretrained',
                return_value=fake_encoder) as vision_load, \
             mock.patch(
                'transformers.SiglipImageProcessor.from_pretrained',
                return_value=fake_processor):
            result = SiglipImageEncoder.from_pretrained(
                'google/siglip-so400m-patch14-384',
                dtype=torch.float16)

        self.assertIsInstance(result, SiglipImageEncoder)
        self.assertIs(result.image_encoder, fake_encoder)
        self.assertIs(result.feature_extractor, fake_processor)
        self.assertEqual(vision_load.call_args.kwargs['dtype'], torch.float16)

    def test_from_pretrained_torch_dtype_alias(self):
        fake_encoder = mock.Mock()
        fake_processor = mock.Mock()

        with mock.patch(
                'transformers.SiglipVisionModel.from_pretrained',
                return_value=fake_encoder) as vision_load, \
             mock.patch(
                'transformers.SiglipImageProcessor.from_pretrained',
                return_value=fake_processor):
            SiglipImageEncoder.from_pretrained(
                'google/siglip-so400m-patch14-384',
                torch_dtype=torch.bfloat16)

        self.assertEqual(vision_load.call_args.kwargs['dtype'], torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
