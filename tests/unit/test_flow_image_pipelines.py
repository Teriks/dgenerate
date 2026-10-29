import inspect
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import diffusers

import dgenerate.arguments as _arguments
import dgenerate.mediainput as _mediainput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.sdnqload as _sdnqload
import dgenerate.renderloopconfig as _renderloopconfig


# Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32 transformer/quantization_config.json
# from the SDNQ release that issue 84 depends on. sdnq.load_sdnq_model reads this file.
_Z_IMAGE_TURBO_SDNQ_UINT4 = {
    "add_skip_keys": False,
    "dequantize_fp32": False,
    "group_size": 0,
    "is_integer": True,
    "is_training": False,
    "modules_dtype_dict": {},
    "modules_to_not_convert": [
        "all_x_embedder",
        "cap_embedder",
        "all_final_layer",
        "t_embedder",
        "layers.0.adaLN_modulation.0.weight",
    ],
    "non_blocking": False,
    "quant_conv": False,
    "quant_method": "sdnq",
    "quantization_device": None,
    "quantized_matmul_dtype": None,
    "return_device": None,
    "svd_rank": 32,
    "svd_steps": 8,
    "use_grad_ckpt": True,
    "use_quantized_matmul": False,
    "use_quantized_matmul_conv": False,
    "use_static_quantization": True,
    "use_stochastic_rounding": False,
    "use_svd": True,
    "weights_dtype": "uint4",
}


def _config(**values):
    config = _renderloopconfig.RenderLoopConfig()
    for name, value in values.items():
        setattr(config, name, value)
    return config


class TestFlowImagePipelines(unittest.TestCase):
    def test_lora_no_disk_fuse_skips_the_lora_component_cache(self):
        class Pipeline:
            _lora_loadable_modules = [
                'unet', 'text_encoder', 'text_encoder_2', 'lora_only']

        repo, skip, fused, fuse_first = _pipelines._components_cached_before_quantization(
            Pipeline, _pipelinewrapper.ModelType.SDXL, 'org/sdxl',
            ['org/lora'], ['org/control'], True, set(), None)
        self.assertEqual(repo, [])
        self.assertTrue(skip)
        self.assertTrue(fused)
        self.assertFalse(fuse_first)

        single, skip, _, fuse_first = _pipelines._components_cached_before_quantization(
            Pipeline, _pipelinewrapper.ModelType.SDXL, 'model.safetensors',
            ['org/lora'], ['org/control'], True, set(), None)
        self.assertFalse(fuse_first)
        self.assertFalse(skip)
        self.assertNotIn('lora_only', single)
        self.assertEqual(single, _pipelines._get_quantizable_components(
            _pipelinewrapper.ModelType.SDXL))

        fused_components, _, _, fuse_first = _pipelines._components_cached_before_quantization(
            Pipeline, _pipelinewrapper.ModelType.SDXL, 'model.safetensors',
            ['org/lora'], ['org/control'], False, set(), None)
        self.assertTrue(fuse_first)
        self.assertIn('lora_only', fused_components)
        self.assertIn('unet', fused_components)

    def test_new_image_models_plan_transformer_lora_quantization_preflight(self):
        for pipeline_class, model_type in (
                (diffusers.QwenImage21Pipeline, _pipelinewrapper.ModelType.QWEN_IMAGE_21),
                (diffusers.Krea2Pipeline, _pipelinewrapper.ModelType.KREA2),
                (diffusers.Kandinsky5T2IPipeline, _pipelinewrapper.ModelType.KANDINSKY5_IMAGE)):
            components = _pipelines._get_affected_components_for_lora(pipeline_class)
            self.assertIn('transformer', components, pipeline_class.__name__)
            self.assertIn('transformer', _pipelines._get_quantizable_components(
                model_type))

    def test_cache_dit_rejected_without_block_adapter(self):
        for model_type in (_pipelinewrapper.ModelType.QWEN_IMAGE_21,
                           _pipelinewrapper.ModelType.MOTIF_VIDEO):
            with self.subTest(model_type=model_type):
                with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
                    _config(model_path='org/model', model_type=model_type,
                            cache_dit=True).check()
        for model_type in (_pipelinewrapper.ModelType.KREA2,
                           _pipelinewrapper.ModelType.KANDINSKY5_IMAGE):
            with self.subTest(model_type=model_type):
                _config(model_path='org/model', model_type=model_type,
                        cache_dit=True).check()

    def test_enum_parsing(self):
        self.assertEqual(_pipelinewrapper.get_model_type_enum('flux2'), _pipelinewrapper.ModelType.FLUX2)
        self.assertEqual(_pipelinewrapper.get_model_type_enum('z-image'), _pipelinewrapper.ModelType.Z_IMAGE)
        self.assertEqual(_pipelinewrapper.get_model_type_enum('qwen-image'), _pipelinewrapper.ModelType.QWEN_IMAGE)
        self.assertEqual(_pipelinewrapper.get_model_type_string(_pipelinewrapper.ModelType.FLUX2), 'flux2')
        self.assertIn('flux2', _pipelinewrapper.supported_model_type_strings())
        self.assertIn('z-image', _pipelinewrapper.supported_model_type_strings())
        self.assertIn('qwen-image', _pipelinewrapper.supported_model_type_strings())
        self.assertTrue(_pipelinewrapper.model_type_is_flux('flux'))
        self.assertTrue(_pipelinewrapper.model_type_is_flux('flux-fill'))
        self.assertFalse(_pipelinewrapper.model_type_is_flux('flux2'))
        self.assertTrue(_pipelinewrapper.model_type_is_flux2('flux2'))
        self.assertTrue(_pipelinewrapper.model_type_is_flow_image('qwen-image'))
        for model_type in ('qwen-image-21', 'krea2', 'kandinsky5-image'):
            self.assertTrue(_pipelinewrapper.model_type_is_flow_image(model_type))
        self.assertTrue(_pipelinewrapper.model_type_is_qwen_image_21('qwen-image-21'))
        self.assertTrue(_pipelinewrapper.model_type_is_qwen_image_21(
            _pipelinewrapper.ModelType.QWEN_IMAGE_21))
        self.assertTrue(_pipelinewrapper.model_type_is_krea2('krea2'))
        self.assertTrue(_pipelinewrapper.model_type_is_krea2(
            _pipelinewrapper.ModelType.KREA2))
        self.assertTrue(_pipelinewrapper.model_type_is_kandinsky5_image('kandinsky5-image'))
        self.assertTrue(_pipelinewrapper.model_type_is_kandinsky5_image(
            _pipelinewrapper.ModelType.KANDINSKY5_IMAGE))
        self.assertFalse(_pipelinewrapper.model_type_is_flow_image('flux'))

        parsed = _arguments.parse_args([
            'org/qwen-image-2.1', '--model-type', 'qwen-image-21',
            '--prompts', 'a fox', '--qwen21-output-resolution', '768',
            '--qwen21-no-kv-cache'], throw=True, log_error=False)
        self.assertEqual(parsed.qwen21_output_resolution, 768)
        self.assertFalse(parsed.qwen21_use_kv_cache)

        import shlex

        import dgenerate.pipelinewrapper.argreconstruct as _argreconstruct

        wrapper = _pipelinewrapper.DiffusionPipelineWrapper(
            model_path=parsed.model_path,
            model_type=parsed.model_type,
            device='cpu')
        command = _argreconstruct.gen_dgenerate_command(
            wrapper, next(parsed.iterate_diffusion_args()), omit_device=True)
        again = _arguments.parse_args(
            shlex.split(command)[1:], throw=True, log_error=False)
        self.assertEqual(again.qwen21_output_resolution, 768)
        self.assertFalse(again.qwen21_use_kv_cache)

    def test_new_flow_image_models_allow_latent_output_and_denoising_splits(self):
        for model_type, model_path in (
                (_pipelinewrapper.ModelType.QWEN_IMAGE_21, 'Qwen/Qwen-Image-2.1'),
                (_pipelinewrapper.ModelType.KREA2, 'krea/Krea-2-Raw'),
                (_pipelinewrapper.ModelType.KANDINSKY5_IMAGE,
                 'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers')):
            with self.subTest(model_type=model_type):
                config = _config(
                    model_path=model_path,
                    model_type=model_type,
                    image_format='pt',
                    denoising_end=0.8)
                config.check()

    def test_model_index_class_names(self):
        flux2 = _pipelinewrapper.ModelType.FLUX2
        z_image = _pipelinewrapper.ModelType.Z_IMAGE
        qwen = _pipelinewrapper.ModelType.QWEN_IMAGE
        flux = _pipelinewrapper.ModelType.FLUX

        _pipelines.validate_model_index_class(flux2, 'Flux2Pipeline', 'org/flux2')
        _pipelines.validate_model_index_class(flux2, 'Flux2KleinPipeline', 'org/klein')
        _pipelines.validate_model_index_class(flux2, 'Flux2KleinInpaintPipeline', 'org/klein')
        _pipelines.validate_model_index_class(z_image, 'ZImagePipeline', 'org/z')
        _pipelines.validate_model_index_class(z_image, 'ZImageImg2ImgPipeline', 'org/z')
        _pipelines.validate_model_index_class(qwen, 'QwenImagePipeline', 'org/qwen')

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.validate_model_index_class(flux, 'Flux2Pipeline', 'org/flux2')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.validate_model_index_class(flux2, 'FluxPipeline', 'org/flux')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.validate_model_index_class(z_image, 'Flux2Pipeline', 'org/flux2')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as klein_kv:
            _pipelines.validate_model_index_class(flux2, 'Flux2KleinKVPipeline', 'org/kv')
        self.assertIn('flux2-klein-kv', str(klein_kv.exception))
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.FLUX2_KLEIN_KV, 'Flux2KleinKVPipeline', 'org/kv')
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.FLUX2_KLEIN_KV, 'Flux2KleinPipeline',
            'black-forest-labs/FLUX.2-klein-9b-kv')
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.FLUX2, 'Flux2KleinPipeline', 'org/klein')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as edit:
            _pipelines.validate_model_index_class(qwen, 'QwenImageEditPipeline', 'org/edit')
        self.assertIn('qwen-image-edit', str(edit.exception))
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, 'QwenImageEditPlusPipeline', 'org/edit')
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED, 'QwenImageLayeredPipeline', 'org/layered')
        _pipelines.validate_model_index_class(qwen, 'QwenImageControlNetPipeline', 'org/cn')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.validate_model_index_class(z_image, 'ZImageOmniPipeline', 'org/omni')
        _pipelines.validate_model_index_class(
            _pipelinewrapper.ModelType.Z_IMAGE_OMNI, 'ZImageOmniPipeline', 'org/omni')

    def test_class_choice_for_each_mode(self):
        txt2img = _pipelinewrapper.PipelineType.TXT2IMG
        img2img = _pipelinewrapper.PipelineType.IMG2IMG
        inpaint = _pipelinewrapper.PipelineType.INPAINT

        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, txt2img, model_class_name='Flux2Pipeline'),
            diffusers.Flux2Pipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, txt2img, model_class_name='Flux2KleinPipeline'),
            diffusers.Flux2KleinPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, inpaint, model_class_name='Flux2KleinPipeline'),
            diffusers.Flux2KleinInpaintPipeline)

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, img2img, model_class_name='Flux2Pipeline')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, inpaint, model_class_name='Flux2Pipeline')
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2, txt2img,
                model_class_name='Flux2Pipeline', controlnet_uris=['org/control'])

        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.Z_IMAGE, txt2img),
            diffusers.ZImagePipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.Z_IMAGE, img2img),
            diffusers.ZImageImg2ImgPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.Z_IMAGE, inpaint),
            diffusers.ZImageInpaintPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.Z_IMAGE, txt2img, controlnet_uris=['org/union']),
            diffusers.ZImageControlNetPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.Z_IMAGE, inpaint, controlnet_uris=['org/union']),
            diffusers.ZImageControlNetInpaintPipeline)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.Z_IMAGE, img2img, controlnet_uris=['org/union'])

        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.QWEN_IMAGE, txt2img),
            diffusers.QwenImagePipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.QWEN_IMAGE, img2img),
            diffusers.QwenImageImg2ImgPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.QWEN_IMAGE, inpaint),
            diffusers.QwenImageInpaintPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE, txt2img, controlnet_uris=['org/control']),
            diffusers.QwenImageControlNetPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE, inpaint, controlnet_uris=['org/control']),
            diffusers.QwenImageControlNetInpaintPipeline)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE, img2img, controlnet_uris=['org/control'])

        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.FLUX2_KLEIN_KV, txt2img,
                model_class_name='Flux2KleinKVPipeline'),
            diffusers.Flux2KleinKVPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.Z_IMAGE_OMNI, txt2img),
            diffusers.ZImageOmniPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, txt2img,
                model_class_name='QwenImageEditPipeline'),
            diffusers.QwenImageEditPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, txt2img,
                model_class_name='QwenImageEditPlusPipeline'),
            diffusers.QwenImageEditPlusPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(
                _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, inpaint,
                model_class_name='QwenImageEditPipeline'),
            diffusers.QwenImageEditInpaintPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED, txt2img),
            diffusers.QwenImageLayeredPipeline)

        qwen21 = _pipelinewrapper.ModelType.QWEN_IMAGE_21
        krea2 = _pipelinewrapper.ModelType.KREA2
        kandinsky5 = _pipelinewrapper.ModelType.KANDINSKY5_IMAGE
        self.assertIs(_pipelines.get_pipeline_class(qwen21, txt2img), diffusers.QwenImage21Pipeline)
        self.assertIs(_pipelines.get_pipeline_class(krea2, txt2img), diffusers.Krea2Pipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(kandinsky5, txt2img), diffusers.Kandinsky5T2IPipeline)
        self.assertIs(
            _pipelines.get_pipeline_class(kandinsky5, img2img), diffusers.Kandinsky5I2IPipeline)
        for model_type in (qwen21, krea2):
            with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
                _pipelines.get_pipeline_class(model_type, img2img)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(kandinsky5, inpaint)
        for model_type, expected in (
                (qwen21, diffusers.QwenImage21Transformer2DModel),
                (krea2, diffusers.Krea2Transformer2DModel),
                (kandinsky5, diffusers.Kandinsky5Transformer3DModel)):
            self.assertIs(_pipelines.diffusion_transformer_class(model_type), expected)

        krea_text_encoder_args = [
            name for name in inspect.getfullargspec(diffusers.Krea2Pipeline.__init__).args
            if _pipelines._is_text_encoder_component(name)
        ]
        self.assertEqual(krea_text_encoder_args, ['text_encoder'])
        self.assertFalse(_pipelines._is_text_encoder_component('text_encoder_select_layers'))
        self.assertTrue(_pipelines._is_text_encoder_component('text_encoder_2'))

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.Z_IMAGE_OMNI, inpaint)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(_pipelinewrapper.ModelType.FLUX2_KLEIN_KV, inpaint)

    def test_rejected_options(self):
        z_image = _pipelinewrapper.ModelType.Z_IMAGE
        for model_type in (
            _pipelinewrapper.ModelType.FLUX2,
            z_image,
            _pipelinewrapper.ModelType.QWEN_IMAGE,
        ):
            for kwargs in (
                {'t2i_adapter_uris': ['org/adapter']},
                {'ip_adapter_uris': ['org/ip']},
                {'pag': True},
                {'unet_uri': 'org/unet'},
            ):
                with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
                    _pipelines.get_pipeline_class(model_type, **kwargs)

        config = _config(
            model_path='Tongyi-MAI/Z-Image-Turbo',
            model_type=z_image,
            prompt_weighter_uri='sd-embed',
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            config.check()
        self.assertIn('prompt_weighter', str(raised.exception))

        import dgenerate.promptweighters as _promptweighters
        for model_type in (
            _pipelinewrapper.ModelType.FLUX2,
            _pipelinewrapper.ModelType.FLUX2_KLEIN_KV,
            _pipelinewrapper.ModelType.Z_IMAGE,
            _pipelinewrapper.ModelType.Z_IMAGE_OMNI,
            _pipelinewrapper.ModelType.QWEN_IMAGE,
            _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT,
            _pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED,
        ):
            for uri in ('sd-embed', 'compel', 'llm4gen;encoder=base-all'):
                with self.assertRaises(_promptweighters.PromptWeightingUnsupported):
                    _promptweighters.create_prompt_weighter(
                        uri, model_type, _pipelinewrapper.DataType.AUTO, device='cpu')

        clipped = _config(
            model_path='Tongyi-MAI/Z-Image-Turbo',
            model_type=z_image,
            clip_skips=[1],
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            clipped.check()
        self.assertIn('clip_skips', str(raised.exception))

        refiner = _config(
            model_path='Qwen/Qwen-Image',
            model_type=_pipelinewrapper.ModelType.QWEN_IMAGE,
            sdxl_refiner_uri='stabilityai/stable-diffusion-xl-refiner-1.0',
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            refiner.check()

    def test_qwen_true_cfg_scale_mapping(self):
        mapped = _pipelines.qwen_image_call_guidance(4.0, None)
        self.assertEqual(mapped['true_cfg_scale'], 4.0)
        self.assertEqual(mapped['negative_prompt'], ' ')
        self.assertNotIn('guidance_scale', mapped)

        kept = _pipelines.qwen_image_call_guidance(4.0, 'blurry')
        self.assertEqual(kept['negative_prompt'], 'blurry')

        no_cfg = _pipelines.qwen_image_call_guidance(1.0, None)
        self.assertEqual(no_cfg['true_cfg_scale'], 1.0)
        self.assertNotIn('negative_prompt', no_cfg)

    def test_flux2_references_are_text_to_image(self):
        beach = 'examples/media/beach.jpg'
        parsed = _mediainput.parse_image_seed_uri(beach)
        self.assertIsNone(parsed.reference_images)
        config = _config(
            model_path='black-forest-labs/FLUX.2-dev',
            model_type=_pipelinewrapper.ModelType.FLUX2,
            image_seeds=[beach],
            prompts=['a wave'])
        config.check()

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            _config(
                model_path='black-forest-labs/FLUX.2-dev',
                model_type=_pipelinewrapper.ModelType.FLUX2,
                image_seeds=[beach],
                image_seed_strengths=[0.6],
                prompts=['a wave']).check()
        self.assertIn('strength', str(raised.exception))

        masked = _mediainput.parse_image_seed_uri(
            f'{beach};mask=examples/media/horse1-mask.jpg;reference=examples/media/earth.jpg')
        self.assertEqual(masked.reference_images, ['examples/media/earth.jpg'])

        several = _mediainput.parse_image_seed_uri(
            f'{beach};mask=examples/media/horse1-mask.jpg;'
            'reference=examples/media/earth.jpg, examples/media/mountain.png')
        self.assertEqual(len(several.reference_images), 2)

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='Tongyi-MAI/Z-Image-Turbo',
                model_type=_pipelinewrapper.ModelType.Z_IMAGE,
                image_seeds=[f'{beach};reference=examples/media/earth.jpg'],
                prompts=['a wave']).check()

    def test_prefixed_options_stay_on_their_model(self):
        flux2 = _pipelinewrapper.ModelType.FLUX2
        z_image = _pipelinewrapper.ModelType.Z_IMAGE
        qwen = _pipelinewrapper.ModelType.QWEN_IMAGE

        kept = _config(
            model_path='black-forest-labs/FLUX.2-dev',
            model_type=flux2,
            flux2_caption_upsample_temperature=0.5,
            flux2_text_encoder_out_layers=(10, 20, 30))
        kept.check()
        generated = next(kept.iterate_diffusion_args())
        self.assertEqual(generated.flux2_caption_upsample_temperature, 0.5)
        self.assertEqual(generated.flux2_text_encoder_out_layers, (10, 20, 30))

        dropped = _config(
            model_path='Tongyi-MAI/Z-Image-Turbo',
            model_type=z_image,
            flux2_caption_upsample_temperature=0.5)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            dropped.check()
        self.assertIn('flux2', str(raised.exception))

        z_kept = _config(
            model_path='Tongyi-MAI/Z-Image-Turbo',
            model_type=z_image,
            z_image_cfg_normalization=True,
            z_image_cfg_truncation=0.25)
        z_kept.check()
        z_args = next(z_kept.iterate_diffusion_args())
        self.assertTrue(z_args.z_image_cfg_normalization)
        self.assertEqual(z_args.z_image_cfg_truncation, 0.25)

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            _config(
                model_path='black-forest-labs/FLUX.2-dev',
                model_type=flux2,
                qwen_guidance_scale=3.5).check()
        self.assertIn('qwen-image', str(raised.exception))

        q_kept = _config(
            model_path='Qwen/Qwen-Image',
            model_type=qwen,
            qwen_guidance_scale=3.5)
        q_kept.check()
        self.assertEqual(next(q_kept.iterate_diffusion_args()).qwen_guidance_scale, 3.5)

        qwen21 = _config(
            model_path='org/Qwen-Image-2.1',
            model_type=_pipelinewrapper.ModelType.QWEN_IMAGE_21,
            qwen21_output_resolution=768,
            qwen21_use_kv_cache=False)
        qwen21.check()
        qwen21_args = next(qwen21.iterate_diffusion_args())
        self.assertEqual(qwen21_args.qwen21_output_resolution, 768)
        self.assertFalse(qwen21_args.qwen21_use_kv_cache)

        older_qwen_models = (
            (_pipelinewrapper.ModelType.QWEN_IMAGE, 'Qwen/Qwen-Image', {}),
            (_pipelinewrapper.ModelType.QWEN_IMAGE_EDIT, 'Qwen/Qwen-Image-Edit',
             {'image_seeds': ['examples/media/beach.jpg']}),
            (_pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED, 'Qwen/Qwen-Image-Layered',
             {'image_seeds': ['examples/media/beach.jpg']}),
        )
        for model_type, model_path, extra in older_qwen_models:
            for option in (
                    {'qwen21_output_resolution': 768},
                    {'qwen21_use_kv_cache': False}):
                with self.subTest(model_type=model_type, option=option):
                    with self.assertRaisesRegex(
                            _renderloopconfig.RenderLoopConfigError,
                            'only supported for --model-type qwen-image-21'):
                        _config(
                            model_path=model_path,
                            model_type=model_type,
                            **extra,
                            **option).check()

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='org/Z-Image',
                model_type=z_image,
                qwen21_output_resolution=768).check()

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            _config(
                model_path='Tongyi-MAI/Z-Image-Turbo',
                model_type=z_image,
                controlnet_uris=['org/union;start=0.2'],
                image_seeds=['examples/media/beach.jpg']).check()
        self.assertIn('start and end', str(raised.exception))

    def test_call_argument_and_padding_delegation(self):
        class Pipe:
            def __call__(self, padding_mask_crop=None, caption_upsample_temperature=None):
                return None

        class NoPadding:
            def __call__(self, prompt=None):
                return None

        wrapper = _pipelinewrapper.DiffusionPipelineWrapper.__new__(
            _pipelinewrapper.DiffusionPipelineWrapper)
        wrapper._pipeline = Pipe()
        args = _pipelinewrapper.DiffusionArguments()
        args.inpaint_crop = True
        args.inpaint_crop_padding = 48
        self.assertEqual(wrapper._delegated_padding_mask_crop(args), 48)

        args.inpaint_crop_feather = 8
        self.assertIsNone(wrapper._delegated_padding_mask_crop(args))

        args.inpaint_crop_feather = None
        args.inpaint_crop_padding = (10, 20)
        self.assertIsNone(wrapper._delegated_padding_mask_crop(args))

        wrapper._pipeline = NoPadding()
        args.inpaint_crop_padding = 32
        self.assertIsNone(wrapper._delegated_padding_mask_crop(args))

        pipeline_args = {}
        wrapper._pipeline = Pipe()
        wrapper._set_accepted_call_argument(
            pipeline_args, 'caption_upsample_temperature', 0.4, 'missing')
        self.assertEqual(pipeline_args['caption_upsample_temperature'], 0.4)

        wrapper._pipeline = NoPadding()
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            wrapper._set_accepted_call_argument(
                {}, 'caption_upsample_temperature', 0.4, 'Klein rejects caption upsampling.')

    def test_qwen_layered_reference_images_are_rgba(self):
        from PIL import Image

        import dgenerate.image as _image

        rgb = Image.new('RGB', (8, 8), (10, 20, 30))
        rgb.filename = 'seed.png'
        rgba = Image.new('RGBA', (8, 8), (10, 20, 30, 128))

        converted = _image.to_rgba(rgb)
        self.assertEqual(converted.mode, 'RGBA')
        self.assertEqual(converted.getpixel((0, 0))[3], 255)
        self.assertEqual(converted.filename, 'seed.png')
        self.assertIs(_image.to_rgba(rgba), rgba)

        out = _pipelinewrapper.DiffusionPipelineWrapper._qwen_layered_reference_images(
            [rgb, rgba])
        self.assertEqual(out[0].mode, 'RGBA')
        self.assertIs(out[1], rgba)

    def test_condition_model_types(self):
        beach = 'examples/media/beach.jpg'
        edit = _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT
        layered = _pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED
        omni = _pipelinewrapper.ModelType.Z_IMAGE_OMNI
        kandinsky5 = _pipelinewrapper.ModelType.KANDINSKY5_IMAGE

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(model_path='Qwen/Qwen-Image-Edit', model_type=edit).check()

        kept = _config(
            model_path='Qwen/Qwen-Image-Edit',
            model_type=edit,
            image_seeds=[beach])
        kept.check()

        kandinsky_edit = _config(
            model_path='org/Kandinsky-5-I2I',
            model_type=kandinsky5,
            image_seeds=[beach])
        kandinsky_edit.check()
        self.assertTrue(_pipelinewrapper.model_type_image_is_condition(kandinsky5))
        self.assertEqual(
            kandinsky_edit.parsed_image_seeds[0].images[0], beach)
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='org/Kandinsky-5-I2I',
                model_type=kandinsky5,
                image_seeds=[beach],
                image_seed_strengths=[0.5]).check()

        layers = _config(
            model_path='Qwen/Qwen-Image-Layered',
            model_type=layered,
            image_seeds=[beach],
            qwen_layered_layers=6,
            qwen_layered_use_en_prompt=True)
        layers.check()
        generated = next(layers.iterate_diffusion_args())
        self.assertEqual(generated.qwen_layered_layers, 6)
        self.assertTrue(generated.qwen_layered_use_en_prompt)

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='Tongyi-MAI/Z-Image-Turbo',
                model_type=_pipelinewrapper.ModelType.Z_IMAGE,
                qwen_layered_layers=4).check()

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='org/omni',
                model_type=omni,
                image_seeds=[beach],
                image_seed_strengths=[0.5]).check()

        _config(
            model_path='Qwen/Qwen-Image',
            model_type=_pipelinewrapper.ModelType.QWEN_IMAGE,
            controlnet_uris=['InstantX/Qwen-Image-ControlNet-Inpainting'],
            image_seeds=[f'{beach};mask=examples/media/horse1-mask.jpg']).check()

    def test_qwen_image_controlnet_condition_channels(self):
        inpaint = _pipelinewrapper.PipelineType.INPAINT
        txt2img = _pipelinewrapper.PipelineType.TXT2IMG
        union = types.SimpleNamespace(
            config=types.SimpleNamespace(extra_condition_channels=0))
        inpaint_cn = types.SimpleNamespace(
            config=types.SimpleNamespace(extra_condition_channels=4))

        _pipelines._validate_qwen_image_controlnet_condition_channels(
            union, txt2img, 'InstantX/Qwen-Image-ControlNet-Union')
        _pipelines._validate_qwen_image_controlnet_condition_channels(
            inpaint_cn, inpaint, 'InstantX/Qwen-Image-ControlNet-Inpainting')

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as raised:
            _pipelines._validate_qwen_image_controlnet_condition_channels(
                union, inpaint, 'InstantX/Qwen-Image-ControlNet-Union')
        self.assertIn('ControlNet-Inpainting', str(raised.exception))
        self.assertIn('extra_condition_channels=0', str(raised.exception))

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as raised:
            _pipelines._validate_qwen_image_controlnet_condition_channels(
                inpaint_cn, txt2img, 'InstantX/Qwen-Image-ControlNet-Inpainting')
        self.assertIn('extra_condition_channels=4', str(raised.exception))
        self.assertIn('ControlNet-Union', str(raised.exception))

    def test_qwen_edit_plus_allows_mismatched_image_sizes(self):
        import dgenerate.mediainput as _mediainput

        uri = ('images: examples/media/beach.jpg, '
               'examples/media/horse1.jpg')
        with self.assertRaises(_mediainput.ImageSeedSizeMismatchError):
            with next(_mediainput.iterate_image_seed(
                    uri, resize_resolution=(1024, 1024),
                    check_dimensions_match=True)) as seed:
                pass

        with next(_mediainput.iterate_image_seed(
                uri, resize_resolution=(1024, 1024),
                check_dimensions_match=False)) as seed:
            self.assertEqual(len(seed.images), 2)
            self.assertNotEqual(seed.images[0].size, seed.images[1].size)

    def test_flow_option_argument_reconstruction(self):
        import shlex

        import dgenerate.arguments as _arguments
        import dgenerate.pipelinewrapper.argreconstruct as _argreconstruct

        cases = (
            [
                '--model-type', 'flux2', 'black-forest-labs/FLUX.2-dev',
                '--flux2-caption-upsample-temperature', '0.5',
                '--flux2-text-encoder-out-layers', '10,20,30',
                '--max-sequence-length', '256',
                '--inference-steps', '20', '--guidance-scales', '4',
                '--seeds', '1', '--prompts', 'a wave',
            ],
            [
                '--model-type', 'z-image', 'Tongyi-MAI/Z-Image-Turbo',
                '--z-image-cfg-normalization',
                '--z-image-cfg-truncation', '0',
                '--inference-steps', '8', '--guidance-scales', '0',
                '--seeds', '1', '--prompts', 'a horse',
            ],
            [
                '--model-type', 'qwen-image-layered', 'Qwen/Qwen-Image-Layered',
                '--qwen-guidance-scale', '2.5',
                '--qwen-layered-layers', '6',
                '--qwen-layered-resolution', '1024',
                '--qwen-layered-cfg-normalize',
                '--qwen-layered-use-en-prompt',
                '--inference-steps', '20', '--guidance-scales', '4',
                '--seeds', '1', '--prompts', 'a beach',
                '--image-seeds', 'examples/media/beach.jpg',
            ],
        )
        compared = (
            'model_type',
            'flux2_caption_upsample_temperature',
            'flux2_text_encoder_out_layers',
            'max_sequence_length',
            'z_image_cfg_normalization',
            'z_image_cfg_truncation',
            'qwen_guidance_scale',
            'qwen_layered_layers',
            'qwen_layered_resolution',
            'qwen_layered_cfg_normalize',
            'qwen_layered_use_en_prompt',
        )
        for argv in cases:
            config = _arguments.parse_args(argv, throw=True, log_error=False)
            generated = next(config.iterate_diffusion_args())
            wrapper = _pipelinewrapper.DiffusionPipelineWrapper(
                model_path=config.model_path,
                model_type=config.model_type,
                device='cpu')
            extra = None
            if '--image-seeds' in argv:
                extra = [('--image-seeds', argv[argv.index('--image-seeds') + 1])]
            command = _argreconstruct.gen_dgenerate_command(
                wrapper, generated, extra_opts=extra, omit_device=True)
            parsed = _arguments.parse_args(
                shlex.split(command)[1:], throw=True, log_error=False)
            for name in compared:
                self.assertEqual(
                    getattr(parsed, name), getattr(config, name),
                    f'{name} did not round-trip for {argv[1]}')

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='Qwen/Qwen-Image-Layered',
                model_type=_pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED,
                image_seeds=['examples/media/beach.jpg'],
                qwen_layered_resolution=512).check()

    def test_pipeline_cache_counts_bundled_modules(self):
        index = {
            '_class_name': 'ZImageOmniPipeline',
            'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
            'text_encoder': ['transformers', 'Qwen3Model'],
            'tokenizer': ['transformers', 'AutoTokenizer'],
            'transformer': ['diffusers', 'ZImageTransformer2DModel'],
            'vae': ['diffusers', 'AutoencoderKL'],
            'siglip': ['transformers', 'Siglip2VisionModel'],
            'siglip_processor': ['transformers', 'Siglip2ImageProcessorFast'],
        }
        self.assertEqual(_pipelines.extra_pipeline_weight_directories(index), ['siglip'])
        self.assertEqual(_pipelines.extra_pipeline_weight_directories(None), [])
        self.assertTrue(_pipelines.controlnet_is_owned_by_pipeline(
            _pipelinewrapper.ModelType.Z_IMAGE, False, False))
        self.assertFalse(_pipelines.controlnet_is_owned_by_pipeline(
            _pipelinewrapper.ModelType.QWEN_IMAGE, False, False))
        self.assertTrue(_pipelines.controlnet_is_owned_by_pipeline(
            _pipelinewrapper.ModelType.QWEN_IMAGE, True, False))
        self.assertTrue(_pipelines.modules_live_on_pipeline(False, False, True))
        self.assertTrue(_pipelines.controlnet_is_owned_by_pipeline(
            _pipelinewrapper.ModelType.QWEN_IMAGE, False, False, True))

    def test_group_offload_skips_quant_and_stays_cached(self):
        import shlex

        import torch

        import dgenerate.arguments as _arguments
        import dgenerate.pipelinewrapper.argreconstruct as _argreconstruct

        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            _config(
                model_path='org/model',
                model_type=_pipelinewrapper.ModelType.SD,
                model_cpu_offload=True,
                model_group_offload=True).check()

        parsed = _arguments.parse_args([
            '--model-type', 'sd', 'runwayml/stable-diffusion-v1-5',
            '--model-group-offload',
            '--prompts', 'a cat',
        ], throw=True, log_error=False)
        self.assertTrue(parsed.model_group_offload)
        generated = next(parsed.iterate_diffusion_args())
        wrapper = _pipelinewrapper.DiffusionPipelineWrapper(
            model_path=parsed.model_path,
            model_type=parsed.model_type,
            device='cpu',
            model_group_offload=True)
        command = _argreconstruct.gen_dgenerate_command(
            wrapper, generated, omit_device=True)
        self.assertIn('--model-group-offload', command)
        again = _arguments.parse_args(
            shlex.split(command)[1:], throw=True, log_error=False)
        self.assertTrue(again.model_group_offload)

        ordinary = torch.nn.Linear(4, 4)
        self.assertFalse(_pipelines.module_skips_group_offload(ordinary))

        import warnings

        class _WarnOnDirectConfig:
            def __init__(self):
                self.config = {'quantization_config': {'quant_method': 'bitsandbytes'}}

            def __getattr__(self, name):
                if name == 'quantization_config':
                    warnings.warn(
                        "Accessing config attribute `quantization_config` directly",
                        FutureWarning, stacklevel=2)
                    return self.config['quantization_config']
                raise AttributeError(name)

        warned = _WarnOnDirectConfig()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', FutureWarning)
            self.assertFalse(_sdnqload.module_is_sdnq(warned))
            self.assertEqual(
                _sdnqload.quantization_config_of(warned)['quant_method'], 'bitsandbytes')
        self.assertFalse(any('quantization_config' in str(item.message) for item in caught))
        warned.__dict__['quantization_config'] = {'quant_method': 'sdnq'}
        self.assertTrue(_sdnqload.module_is_sdnq(warned))
        quantized = torch.nn.Linear(4, 4)
        quantized.quantization_config = {'quant_method': 'sdnq'}
        parent = torch.nn.Sequential(quantized)
        self.assertTrue(_pipelines.module_skips_group_offload(parent))

        class Pipe:
            def __init__(self):
                shared = torch.nn.Linear(4, 4)
                self.transformer = torch.nn.Sequential(shared, torch.nn.Linear(4, 4))
                self.controlnet = torch.nn.Module()
                self.controlnet._modules['shared'] = shared
                self.controlnet._modules['extra'] = torch.nn.Linear(4, 4)
                self.vae = torch.nn.Linear(4, 4)
                self.vae.quantization_config = {'quant_method': 'sdnq'}
                self._exclude_from_cpu_offload = []

            @property
            def components(self):
                return {
                    'transformer': self.transformer,
                    'controlnet': self.controlnet,
                    'vae': self.vae,
                }

            def remove_all_hooks(self):
                return None

        pipe = Pipe()
        _pipelines.enable_group_offload(pipe, 'cpu')
        self.assertTrue(_pipelines.is_group_offload_enabled(pipe))
        self.assertTrue(_pipelines.is_group_offload_enabled(pipe.transformer))
        self.assertTrue(_pipelines.is_group_offload_enabled(pipe.controlnet))
        self.assertFalse(_pipelines.is_group_offload_enabled(pipe.vae))
        sample = torch.randn(2, 4)
        self.assertEqual(tuple(pipe.transformer(sample).shape), (2, 4))
        self.assertEqual(tuple(pipe.controlnet.extra(sample).shape), (2, 4))
        pipe.enable_group_offload()
        self.assertEqual(tuple(pipe.transformer(sample).shape), (2, 4))
        _pipelines.pipeline_to(pipe, 'cpu')
        self.assertTrue(_pipelines.is_group_offload_enabled(pipe.transformer))

    def test_sdnq_checkpoint_detection_and_era_config(self):
        with tempfile.TemporaryDirectory() as directory:
            quant_dir = os.path.join(directory, 'transformer')
            os.makedirs(quant_dir)
            with open(os.path.join(quant_dir, 'quantization_config.json'), 'w', encoding='utf-8') as file:
                json.dump(_Z_IMAGE_TURBO_SDNQ_UINT4, file)
            found = _sdnqload.sdnq_config_in_directory(quant_dir)
            self.assertEqual(found['weights_dtype'], 'uint4')
            self.assertTrue(found['use_svd'])
            self.assertIsNotNone(_sdnqload.sdnq_requantize_error('sdnq;type=uint4', found))
            self.assertIsNone(_sdnqload.sdnq_requantize_error(None, found))

            nested = os.path.join(directory, 'nested')
            os.makedirs(nested)
            with open(os.path.join(nested, 'config.json'), 'w', encoding='utf-8') as file:
                json.dump({
                    '_class_name': 'ZImageTransformer2DModel',
                    'quantization_config': _Z_IMAGE_TURBO_SDNQ_UINT4,
                }, file)
            nested_found = _sdnqload.sdnq_config_in_directory(nested)
            self.assertEqual(nested_found['quant_method'], 'sdnq')

        import sdnq
        loaded = sdnq.SDNQConfig.from_dict(dict(_Z_IMAGE_TURBO_SDNQ_UINT4))
        self.assertEqual(loaded.weights_dtype, 'uint4')
        self.assertEqual(loaded.svd_rank, 32)
        self.assertTrue(loaded.use_svd)

    def test_zimage_kohya_lora_drops_empty_unet_prefix(self):
        import torch
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_z_image_lora_to_diffusers)
        from dgenerate.pipelinewrapper.uris.lorauri import _collapse_zimage_lora_prefix

        rank = 2
        down = torch.zeros(rank, 4)
        up = torch.zeros(4, rank)
        alpha = torch.tensor(float(rank))
        raw = {}
        for name in (
            'lora_unet__layers_0_attention_to_q',
            'lora_unet__context_refiner_0_feed_forward_w1',
            'lora_unet__noise_refiner_1_attention_to_k',
        ):
            raw[name + '.lora_down.weight'] = down
            raw[name + '.lora_up.weight'] = up
            raw[name + '.alpha'] = alpha

        converted = _convert_non_diffusers_z_image_lora_to_diffusers(raw)
        fixed = _collapse_zimage_lora_prefix(converted)
        self.assertIn('transformer.layers.0.attention.to_q.lora_A.weight', fixed)
        self.assertIn('transformer.context_refiner.0.feed_forward.w1.lora_A.weight', fixed)
        self.assertIn('transformer.noise_refiner.1.attention.to_k.lora_A.weight', fixed)
        self.assertFalse(any('..' in key for key in fixed))

    def test_flow_image_sigma_expressions(self):
        import numpy as np

        from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper

        class _RejectingScheduler:
            def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
                raise AssertionError('flow image sigma expressions must not call set_timesteps')

        class _NoSigmasScheduler:
            def set_timesteps(self, num_inference_steps=None):
                pass

        def pipe(class_name, scheduler):
            obj = type(class_name, (), {})()
            obj.scheduler = scheduler
            return obj

        steps = 8
        expected = (np.linspace(1.0, 1.0 / steps, steps) * 0.95).tolist()
        for name in (
                'FluxPipeline',
                'FluxImg2ImgPipeline',
                'FluxInpaintPipeline',
                'FluxFillPipeline',
                'FluxKontextPipeline',
                'FluxKontextInpaintPipeline',
                'FluxControlNetPipeline',
                'Flux2Pipeline',
                'Flux2KleinPipeline',
                'Flux2KleinInpaintPipeline',
                'Flux2KleinKVPipeline',
                'ZImagePipeline',
                'ZImageImg2ImgPipeline',
                'ZImageInpaintPipeline',
                'ZImageControlNetPipeline',
                'ZImageControlNetInpaintPipeline',
                'ZImageOmniPipeline',
                'QwenImagePipeline',
                'QwenImageImg2ImgPipeline',
                'QwenImageInpaintPipeline',
                'QwenImageEditPipeline',
                'QwenImageEditPlusPipeline',
                'QwenImageEditInpaintPipeline',
                'QwenImageLayeredPipeline',
                'QwenImageControlNetPipeline',
                'QwenImageControlNetInpaintPipeline',
        ):
            result = DiffusionPipelineWrapper._sigmas_eval(
                'primary', pipe(name, _RejectingScheduler()), steps, 'sigmas * 0.95')
            self.assertEqual(len(result), steps, name)
            np.testing.assert_allclose(result, expected)

        # Qwen-Image layered writes this form. It is the same sequence.
        np.testing.assert_allclose(
            np.linspace(1.0, 1.0 / steps, steps),
            np.linspace(1.0, 0.0, steps + 1)[:-1])

        csv = DiffusionPipelineWrapper._sigmas_eval(
            'primary', pipe('ZImagePipeline', _RejectingScheduler()), steps, [0.2, 0.1])
        self.assertEqual(csv, [0.2, 0.1])

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            DiffusionPipelineWrapper._sigmas_eval(
                'primary', pipe('QwenImagePipeline', _NoSigmasScheduler()), steps, 'sigmas * 0.95')

        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as bad_expr:
            DiffusionPipelineWrapper._sigmas_eval(
                'primary', pipe('Flux2Pipeline', _RejectingScheduler()), steps, 'sigmas +')
        self.assertIn('Error interpreting sigmas expression', str(bad_expr.exception))

        class _SdScheduler:
            def __init__(self):
                self.called = False
                self.sigmas = None

            def set_timesteps(self, steps=None, sigmas=None):
                self.called = True
                self.sigmas = np.array([0.4, 0.2])

        sdxl = type('StableDiffusionXLPipeline', (), {})()
        sdxl.scheduler = _SdScheduler()
        scaled = DiffusionPipelineWrapper._sigmas_eval('primary', sdxl, 2, 'sigmas * 2')
        self.assertTrue(sdxl.scheduler.called)
        np.testing.assert_allclose(scaled, [0.8, 0.4])

    def test_prompt_length_warning_uses_pipeline_limit_for_processors(self):
        class _Inner:
            model_max_length = 100000

            def tokenize(self, text):
                return text.split()

        class _Processor:
            def __init__(self):
                self.tokenizer = _Inner()

        class Flux2Pipeline:
            def __init__(self):
                self.tokenizer = _Processor()
                self.tokenizer_max_length = 4

        with patch('dgenerate.pipelinewrapper.pipelines._messages.warning') as warning:
            _pipelines._warn_prompt_lengths(
                Flux2Pipeline(), prompt='one two three four five')
        warning.assert_called_once()
        self.assertIn('of 4', warning.call_args.args[0])

        with patch('dgenerate.pipelinewrapper.pipelines._messages.warning') as warning:
            _pipelines._warn_prompt_lengths(Flux2Pipeline(), prompt='one two')
        warning.assert_not_called()

        class _Clip:
            model_max_length = 3

            def tokenize(self, text):
                return text.split()

        class StableDiffusionPipeline:
            def __init__(self):
                self.tokenizer = _Clip()
                self.tokenizer_max_length = 77

        with patch('dgenerate.pipelinewrapper.pipelines._messages.warning') as warning:
            _pipelines._warn_prompt_lengths(
                StableDiffusionPipeline(), prompt='one two three four')
        warning.assert_called_once()
        self.assertIn('of 3', warning.call_args.args[0])

    def test_adetailer_inpaint_where_it_exists(self):
        import PIL.Image

        from dgenerate.extras.asdff.base import AdPipelineBase, adetailer_flow_inpaint_spec

        self.assertEqual(
            _pipelinewrapper.ADETAILER_INPAINT_MODEL_TYPES,
            ('sd', 'sdxl', 'kolors', 'sd3', 'flux', 'flux-fill', 'flux2',
             'z-image', 'qwen-image', 'qwen-image-edit'))
        klein = adetailer_flow_inpaint_spec('Flux2KleinPipeline')
        self.assertIs(klein.pipeline_class, diffusers.Flux2KleinInpaintPipeline)
        self.assertFalse(klein.negative_prompt)
        self.assertTrue(klein.distilled_from_config)
        self.assertIs(
            adetailer_flow_inpaint_spec('ZImageControlNetPipeline').pipeline_class,
            diffusers.ZImageInpaintPipeline)
        qwen = adetailer_flow_inpaint_spec('QwenImagePipeline')
        self.assertIs(qwen.pipeline_class, diffusers.QwenImageInpaintPipeline)
        self.assertTrue(qwen.qwen_guidance)
        edit = adetailer_flow_inpaint_spec('QwenImageEditPipeline')
        self.assertIs(edit.pipeline_class, diffusers.QwenImageEditInpaintPipeline)
        self.assertTrue(edit.needs_processor)
        self.assertIsNone(adetailer_flow_inpaint_spec('FluxPipeline'))
        self.assertTrue(_pipelinewrapper.model_type_supports_adetailer('z-image'))
        self.assertTrue(_pipelinewrapper.model_type_supports_adetailer('flux2'))
        self.assertFalse(_pipelinewrapper.model_type_supports_adetailer('z-image-omni'))
        self.assertFalse(_pipelinewrapper.model_type_supports_adetailer('qwen-image-layered'))
        self.assertFalse(_pipelinewrapper.model_type_supports_adetailer('flux2-klein-kv'))

        for name, needle in (
                ('Flux2Pipeline', 'Klein'),
                ('Flux2KleinKVPipeline', 'Klein KV'),
                ('ZImageOmniPipeline', 'Omni'),
                ('QwenImageLayeredPipeline', 'Layered'),
                ('QwenImageEditPlusPipeline', 'edit-plus'),
        ):
            with self.assertRaises(_pipelines.UnsupportedPipelineConfigError) as raised:
                adetailer_flow_inpaint_spec(name)
            self.assertIn(needle, str(raised.exception))

        beach = 'examples/media/beach.jpg'
        detector = ['Bingsu/adetailer;weight-name=face_yolov8n.pt']
        for model_path, model_type in (
                ('black-forest-labs/FLUX.2-klein-4B', _pipelinewrapper.ModelType.FLUX2),
                ('Tongyi-MAI/Z-Image-Turbo', _pipelinewrapper.ModelType.Z_IMAGE),
                ('Qwen/Qwen-Image', _pipelinewrapper.ModelType.QWEN_IMAGE),
                ('Qwen/Qwen-Image-Edit', _pipelinewrapper.ModelType.QWEN_IMAGE_EDIT),
        ):
            _config(
                model_path=model_path,
                model_type=model_type,
                image_seeds=[beach],
                image_seed_strengths=[0.4],
                adetailer_detector_uris=detector,
                prompts=['a face']).check()

        for model_path, model_type in (
                ('Tongyi-MAI/Z-Image-Omni-Base', _pipelinewrapper.ModelType.Z_IMAGE_OMNI),
                ('Qwen/Qwen-Image-Layered', _pipelinewrapper.ModelType.QWEN_IMAGE_LAYERED),
                ('black-forest-labs/FLUX.2-klein-9b-kv', _pipelinewrapper.ModelType.FLUX2_KLEIN_KV),
        ):
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
                _config(
                    model_path=model_path,
                    model_type=model_type,
                    image_seeds=[beach],
                    adetailer_detector_uris=detector,
                    prompts=['a face']).check()
            self.assertIn('no inpaint', str(raised.exception))

        class Flux2KleinInpaintPipeline:
            def __call__(
                    self, image=None, mask_image=None, prompt=None,
                    strength=None, guidance_scale=None, height=None, width=None):
                return None

        pipe = AdPipelineBase(Flux2KleinInpaintPipeline())
        pipe.auto_detect_pipe = False
        prepared = pipe._get_inpaint_args({
            'prompt': 'a face',
            'negative_prompt': 'blur',
            'width': 1024,
            'height': 1024,
            'strength': 0.4,
            'guidance_scale': 1.0,
        })
        self.assertNotIn('negative_prompt', prepared)
        self.assertNotIn('width', prepared)
        self.assertNotIn('height', prepared)
        self.assertEqual(prepared['strength'], 0.4)

        image = PIL.Image.new('RGB', (30, 20), 'red')
        mask = PIL.Image.new('L', (30, 20), 255)
        captured = {}

        def fake_call(pipeline, device, prompt_weighter=None, **kwargs):
            captured['size'] = kwargs['image'].size
            captured['mask'] = kwargs['mask_image'].size
            captured['width'] = kwargs['width']
            captured['height'] = kwargs['height']
            return [[kwargs['image']]]

        with patch('dgenerate.extras.asdff.base._pipelinewrapper.call_pipeline', fake_call):
            result = pipe.process_inpainting(
                {'prompt': 'a face', 'width': 64, 'height': 64},
                image, None, mask, (0, 0, 30, 20), 'cpu')
        self.assertEqual(captured['size'], (32, 32))
        self.assertEqual(captured['mask'], captured['size'])
        self.assertEqual(captured['width'], captured['size'][0])
        self.assertEqual(captured['height'], captured['size'][1])
        self.assertEqual(captured['width'] % 16, 0)
        self.assertEqual(captured['height'] % 16, 0)
        self.assertEqual(result.size, (30, 20))

    def test_qwen_quant_keeps_modulation_full_precision(self):
        import diffusers

        from dgenerate.pipelinewrapper.quant_skips import apply_architecture_quant_skips

        bnb = diffusers.BitsAndBytesConfig(load_in_4bit=True)
        apply_architecture_quant_skips(bnb, diffusers.QwenImageTransformer2DModel)
        skipped = bnb.llm_int8_skip_modules
        self.assertIn('norm_out', skipped)
        self.assertIn('proj_out', skipped)
        self.assertIn('time_text_embed', skipped)
        self.assertIn('transformer_blocks.0.img_mod.1', skipped)
        self.assertIn('transformer_blocks.0.txt_mod.1', skipped)

        class Config:
            modules_to_not_convert = []

        sdnq = Config()
        apply_architecture_quant_skips(sdnq, 'QwenImageTransformer2DModel')
        self.assertIn('transformer_blocks.0.txt_mod.1.weight', sdnq.modules_to_not_convert)

        # ControlNet reuses the block stack; skips apply when it is quantized.
        cn = diffusers.BitsAndBytesConfig(load_in_4bit=True)
        apply_architecture_quant_skips(cn, diffusers.QwenImageControlNetModel)
        cn_skipped = cn.llm_int8_skip_modules
        self.assertIn('time_text_embed', cn_skipped)
        self.assertIn('transformer_blocks.0.img_mod.1', cn_skipped)
        self.assertIn('transformer_blocks.0.txt_mod.1', cn_skipped)
        self.assertNotIn('norm_out', cn_skipped)
        self.assertNotIn('proj_out', cn_skipped)

        cn_sdnq = Config()
        apply_architecture_quant_skips(cn_sdnq, 'QwenImageControlNetModel')
        self.assertIn(
            'transformer_blocks.0.txt_mod.1.weight',
            cn_sdnq.modules_to_not_convert)

        # Wan-Animate aliases reuse WanTransformer3DModel skips for BnB/SDNQ.
        animate = diffusers.BitsAndBytesConfig(load_in_4bit=True)
        apply_architecture_quant_skips(animate, 'WanAnimateTransformer3DModel')
        animate_skipped = animate.llm_int8_skip_modules
        self.assertIn('scale_shift_table', animate_skipped)
        self.assertIn('patch_embedding', animate_skipped)

        animate2_sdnq = Config()
        apply_architecture_quant_skips(
            animate2_sdnq, 'WanAnimate2Transformer3DModel')
        self.assertIn('scale_shift_table', animate2_sdnq.modules_to_not_convert)
        self.assertIn('condition_embedder', animate2_sdnq.modules_to_not_convert)

        # LTX-Video (1.x) is absent from SDNQ's table; keep embeds/projections.
        ltx1 = diffusers.BitsAndBytesConfig(load_in_4bit=True)
        apply_architecture_quant_skips(ltx1, 'LTXVideoTransformer3DModel')
        ltx1_skipped = ltx1.llm_int8_skip_modules
        self.assertIn('time_embed', ltx1_skipped)
        self.assertIn('caption_projection', ltx1_skipped)
        self.assertIn('proj_in', ltx1_skipped)
        self.assertIn('proj_out', ltx1_skipped)
        self.assertIn('scale_shift_table', ltx1_skipped)
        self.assertIn('norm_out', ltx1_skipped)

        ltx1_sdnq = Config()
        apply_architecture_quant_skips(ltx1_sdnq, 'LTXVideoTransformer3DModel')
        self.assertIn('time_embed', ltx1_sdnq.modules_to_not_convert)
        self.assertIn('scale_shift_table', ltx1_sdnq.modules_to_not_convert)


class TestFlowImageTimesteps(unittest.TestCase):
    def test_timesteps_eval(self):
        import numpy as np
        from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper

        class _Scheduler:
            def set_timesteps(self, num_inference_steps=None, timesteps=None):
                self.timesteps = np.array([1000, 500, 0])[:num_inference_steps or 3]

        class _Pipeline:
            pass

        pipe = _Pipeline()
        pipe.scheduler = _Scheduler()

        self.assertEqual(
            DiffusionPipelineWrapper._timesteps_eval('primary', pipe, 2, [100, 50]),
            [100, 50])
        self.assertEqual(
            DiffusionPipelineWrapper._timesteps_eval('primary', pipe, 2, 'timesteps[::2]'),
            [1000])

        class _NoTimestepsScheduler:
            def set_timesteps(self, num_inference_steps=None):
                pass

        pipe2 = _Pipeline()
        pipe2.scheduler = _NoTimestepsScheduler()
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            DiffusionPipelineWrapper._timesteps_eval('primary', pipe2, 2, 'timesteps[::2]')

        # Reading the default schedule for an expression leaves the scheduler as it was.
        pipe.scheduler.timesteps = np.array([7, 3])
        DiffusionPipelineWrapper._timesteps_eval('primary', pipe, 3, 'timesteps[1:]')
        self.assertEqual(pipe.scheduler.timesteps.tolist(), [7, 3])

        for invalid in ([], [10.5], [-1], 'timesteps + 0.5', '5'):
            with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
                DiffusionPipelineWrapper._timesteps_eval('primary', pipe, 3, invalid)


if __name__ == '__main__':
    unittest.main()
