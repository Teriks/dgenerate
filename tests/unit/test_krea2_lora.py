import unittest

import torch

from dgenerate.pipelinewrapper.krea2_lora import convert_krea2_lora_state_dict


def _kohya(module, rank=4, alpha=8.0, out_features=8, in_features=6):
    return {
        f'{module}.lora_down.weight': torch.ones(rank, in_features),
        f'{module}.lora_up.weight': torch.full((out_features, rank), 2.0),
        f'{module}.alpha': torch.tensor(alpha),
    }


class TestKrea2Lora(unittest.TestCase):
    def test_musubi_keys_map_onto_diffusers_modules(self):
        state = {}
        modules = (
            'lora_unet_blocks_0_attn_wq',
            'lora_unet_blocks_0_attn_wk',
            'lora_unet_blocks_0_attn_wv',
            'lora_unet_blocks_0_attn_wo',
            'lora_unet_blocks_0_attn_gate',
            'lora_unet_blocks_0_mlp_gate',
            'lora_unet_blocks_0_mlp_up',
            'lora_unet_blocks_0_mlp_down',
            'lora_unet_first',
            'lora_unet_last_linear',
            'lora_unet_tmlp_0',
            'lora_unet_tmlp_2',
            'lora_unet_tproj_1',
            'lora_unet_txtmlp_1',
            'lora_unet_txtmlp_3',
            'lora_unet_txtfusion_projector',
            'lora_unet_txtfusion_layerwise_blocks_0_attn_gate',
            'lora_unet_txtfusion_layerwise_blocks_1_mlp_down',
            'lora_unet_txtfusion_refiner_blocks_0_attn_wo',
            'lora_unet_txtfusion_refiner_blocks_1_mlp_up',
        )
        for module in modules:
            state.update(_kohya(module))

        converted = convert_krea2_lora_state_dict(state)

        expected = (
            'transformer.transformer_blocks.0.attn.to_q',
            'transformer.transformer_blocks.0.attn.to_k',
            'transformer.transformer_blocks.0.attn.to_v',
            'transformer.transformer_blocks.0.attn.to_out.0',
            'transformer.transformer_blocks.0.attn.to_gate',
            'transformer.transformer_blocks.0.ff.gate',
            'transformer.transformer_blocks.0.ff.up',
            'transformer.transformer_blocks.0.ff.down',
            'transformer.img_in',
            'transformer.final_layer.linear',
            'transformer.time_embed.linear_1',
            'transformer.time_embed.linear_2',
            'transformer.time_mod_proj',
            'transformer.txt_in.linear_1',
            'transformer.txt_in.linear_2',
            'transformer.text_fusion.projector',
            'transformer.text_fusion.layerwise_blocks.0.attn.to_gate',
            'transformer.text_fusion.layerwise_blocks.1.ff.down',
            'transformer.text_fusion.refiner_blocks.0.attn.to_out.0',
            'transformer.text_fusion.refiner_blocks.1.ff.up',
        )
        self.assertEqual(set(converted), {
            f'{name}.lora_A.weight' for name in expected
        } | {
            f'{name}.lora_B.weight' for name in expected
        })
        # alpha 8 / rank 4 folds into lora_B, which was filled with 2.
        up = converted['transformer.transformer_blocks.0.attn.to_q.lora_B.weight']
        self.assertTrue(torch.equal(up, torch.full((8, 4), 4.0)))
        down = converted['transformer.img_in.lora_A.weight']
        self.assertTrue(torch.equal(down, torch.ones(4, 6)))

    def test_reference_and_ai_toolkit_layouts_still_convert(self):
        reference = {
            'diffusion_model.blocks.3.attn.wq.lora_A.weight': torch.ones(2, 4),
            'diffusion_model.blocks.3.attn.wq.lora_B.weight': torch.ones(4, 2),
        }
        toolkit = {
            'base_model.model.blocks.3.mlp.down.lora_A.weight': torch.ones(2, 4),
            'base_model.model.blocks.3.mlp.down.lora_B.weight': torch.ones(4, 2),
        }
        self.assertIn(
            'transformer.transformer_blocks.3.attn.to_q.lora_A.weight',
            convert_krea2_lora_state_dict(reference))
        self.assertIn(
            'transformer.transformer_blocks.3.ff.down.lora_B.weight',
            convert_krea2_lora_state_dict(toolkit))

    def test_dotted_kohya_keys_fold_alpha(self):
        state = _kohya('diffusion_model.blocks.1.attn.wo', rank=4, alpha=4.0)
        converted = convert_krea2_lora_state_dict(state)
        up = converted['transformer.transformer_blocks.1.attn.to_out.0.lora_B.weight']
        self.assertTrue(torch.equal(up, torch.full((8, 4), 2.0)))

    def test_diffusers_keys_pass_through(self):
        state = {
            'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(2, 4),
            'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(4, 2),
        }
        self.assertIs(convert_krea2_lora_state_dict(state), state)


if __name__ == '__main__':
    unittest.main()
