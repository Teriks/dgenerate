import inspect

from typing import Any, Callable, Iterable, List, Mapping, Optional

from PIL import Image

from functools import cached_property

import diffusers

import dgenerate.extras.kolors
import dgenerate.image as _image
from dgenerate.extras.asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from dgenerate.extras.asdff.yolo import yolo_detector

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper

DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class AdPipelineBase:
    def __init__(self, pipe, force_pag=False):
        self.pipe = pipe
        self.force_pag = force_pag
        self.auto_detect_pipe = True
        self.crop_control_image = False

    @cached_property
    def inpaint_pipeline(self):
        if not self.auto_detect_pipe:
            return self.pipe

        is_xl = self.pipe.__class__.__name__.startswith('StableDiffusionXL')
        is_flux = self.pipe.__class__.__name__.startswith('Flux')
        is_sd3 = self.pipe.__class__.__name__.startswith('StableDiffusion3')
        is_kolors = self.pipe.__class__.__name__.startswith('Kolors')
        is_pag = "PAG" in self.pipe.__class__.__name__ or self.force_pag

        if is_xl:
            pipe_class = \
                diffusers.StableDiffusionXLPAGInpaintPipeline if \
                    is_pag else diffusers.StableDiffusionXLInpaintPipeline

            pipe = pipe_class(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                feature_extractor=self.pipe.feature_extractor,
            )
        elif is_flux:
            pipe = diffusers.FluxInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler
            )
        elif is_sd3:
            pipe = diffusers.StableDiffusion3InpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                text_encoder_3=self.pipe.text_encoder_3,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                tokenizer_3=self.pipe.tokenizer_3,
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
            )
        elif is_kolors:
            pipe = dgenerate.extras.kolors.KolorsInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                feature_extractor=self.pipe.feature_extractor,
            )
        else:
            pipe_class = \
                diffusers.StableDiffusionPAGInpaintPipeline if \
                    is_pag else diffusers.StableDiffusionInpaintPipeline

            pipe = pipe_class(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker if hasattr(self.pipe, 'safety_checker') else None,
                feature_extractor=self.pipe.feature_extractor if hasattr(self.pipe, 'feature_extractor') else None,
            )

        _messages.debug_log(
            f'AdPipelineBase (adetailer pipeline) initialized pipeline class: {pipe.__class__.__name__}')

        return pipe

    def __call__(  # noqa: C901
            self,
            pipeline_args: Mapping[str, Any] | None = None,
            images: Image.Image | Iterable[Image.Image] | None = None,
            detector: DetectorType | None = None,
            mask_dilation: int = 4,
            mask_blur: int = 4,
            mask_padding: int | tuple[int, int] | tuple[int, int, int, int] = 32,
            detector_padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
            model_path: str = None,
            device: str = 'cuda',
            detector_device: str = 'cuda',
            confidence: float = 0.3,
            mask_shape: str = 'rectangle',
            model_masks: bool = False,
            class_filter: set[int|str] | list[int|str] | None = None,
            index_filter: set[int] | list[int] | None = None,
            prompt_weighter: dgenerate.promptweighters.PromptWeighter | None = None,
            processing_size: int | None = None
    ):
        if pipeline_args is None:
            pipeline_args = {}

        if detector is None:
            detector = self.default_detector

        txt2img_images = [images] if not isinstance(images, Iterable) else images

        control_images = [None] * len(txt2img_images)

        if 'control_image' in pipeline_args:
            for idx, txt2img in enumerate(txt2img_images):
                ctrl_img_pipe = pipeline_args['control_image']
                if isinstance(ctrl_img_pipe, list):
                    control_images[idx] = ctrl_img_pipe[idx]
                else:
                    control_images[idx] = ctrl_img_pipe

        init_images = []
        final_images = []

        for i, pipe_images in enumerate(zip(txt2img_images, control_images)):
            init_image = pipe_images[0]
            control_image = pipe_images[1]

            init_images.append(init_image.copy())
            final_image = None

            masks = detector(
                init_image,
                confidence=confidence,
                device=detector_device,
                model_path=model_path,
                mask_shape=mask_shape,
                padding=detector_padding,
                class_filter=class_filter,
                index_filter=index_filter,
                model_masks=model_masks
            )

            if masks is None:
                _messages.log(
                    f"No object detected on {ordinal(i + 1)} image."
                )
                continue

            for k, mask in enumerate(masks):
                mask = mask.convert("L")
                mask = mask_dilate(mask, mask_dilation)
                bbox = mask.getbbox()
                if bbox is None:
                    _messages.log(f"No object in {ordinal(k + 1)} mask.")
                    continue
                mask = mask_gaussian_blur(mask, mask_blur)
                bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)
                inpaint_image = self.process_inpainting(
                    pipeline_args,
                    init_image,
                    control_image,
                    mask,
                    bbox_padded,
                    device,
                    prompt_weighter,
                    processing_size
                )
                final_image = composite(
                    init_image,
                    mask,
                    inpaint_image,
                    bbox_padded,
                )
                init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    def _get_inpaint_args(
            self, pipeline_args: dict[str, Any]
    ) -> dict[str, Any]:
        pipeline_args = dict(pipeline_args)
        sig = inspect.signature(self.inpaint_pipeline)
        if (
                "control_image" in sig.parameters
                and "control_image" not in pipeline_args
                and "image" in pipeline_args
        ):
            pipeline_args["control_image"] = pipeline_args.pop("image")
        return {
            **pipeline_args,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

    def process_inpainting(
            self,
            pipeline_args: dict[str, Any],
            init_image: Image.Image,
            control_image: Image.Image | None,
            mask: Image.Image,
            bbox_padded: tuple[int, int, int, int],
            device: str,
            prompt_weighter: dgenerate.promptweighters.PromptWeighter | None = None,
            processing_size: int | None = None
    ):
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
        original_crop_size = crop_image.size
        
        # Apply size-based processing if specified
        if processing_size is not None:
            # Scale based on the larger dimension to handle any aspect ratio consistently
            current_max_dim = max(crop_image.width, crop_image.height)
            size_ratio = processing_size / current_max_dim
            
            # Calculate aspect-correct target size based on the larger dimension
            if crop_image.width >= crop_image.height:
                # Width is larger or equal - scale based on width
                target_width = processing_size
                w_percent = (target_width / float(crop_image.width))
                target_height = int((float(crop_image.height) * float(w_percent)))
            else:
                # Height is larger - scale based on height
                target_height = processing_size
                h_percent = (target_height / float(crop_image.height))
                target_width = int((float(crop_image.width) * float(h_percent)))
            
            target_size = (target_width, target_height)
            
            # Use optimal resampling method for both upscaling and downscaling
            resampling = _image.best_pil_resampling(crop_image.size, target_size)
            
            scale_type = "upscaling" if size_ratio > 1.0 else "downscaling"
            _messages.debug_log(
                f'ADetailer {scale_type} detection area from {crop_image.size} to {target_size} using {resampling}'
            )
            
            crop_image = crop_image.resize(target_size, resampling)
            crop_mask = crop_mask.resize(target_size, resampling)
            
            # Process at scaled resolution
            inpaint_args = self._get_inpaint_args(pipeline_args)
            inpaint_args["image"] = crop_image
            inpaint_args["mask_image"] = crop_mask

            if control_image is not None:
                if self.crop_control_image and init_image.size == control_image.size:
                    control_crop = control_image.crop(bbox_padded)
                    control_resampling = _image.best_pil_resampling(control_crop.size, target_size)
                    inpaint_args["control_image"] = control_crop.resize(target_size, control_resampling)
                else:
                    if init_image.size != control_image.size:
                        _messages.log(
                            'adetailer could not crop the control image correctly as it is a different '
                            'size from your input image, they need to be the same dimension, '
                            'reverting to resize only mode.')
                    
                    control_resampling = _image.best_pil_resampling(control_image.size, target_size)
                    inpaint_args["control_image"] = control_image.resize(target_size, control_resampling)
            
            # Call pipeline with scaled images
            result = _pipelinewrapper.call_pipeline(
                pipeline=self.inpaint_pipeline,
                device=device,
                prompt_weighter=prompt_weighter,
                **inpaint_args)
            
            # Scale result back to original crop size for compositing
            processed_image = result[0][0]
            final_resampling = _image.best_pil_resampling(processed_image.size, original_crop_size)
            
            _messages.debug_log(
                f'ADetailer scaling result from {processed_image.size} to {original_crop_size} '
                f'for compositing using {final_resampling}'
            )
            
            processed_image = processed_image.resize(original_crop_size, final_resampling)
            return processed_image
        
        # Standard processing at original resolution (no scaling needed or no processing_size specified)
        inpaint_args = self._get_inpaint_args(pipeline_args)
        inpaint_args["image"] = crop_image
        inpaint_args["mask_image"] = crop_mask

        if control_image is not None:
            if self.crop_control_image and init_image.size == control_image.size:
                inpaint_args["control_image"] = control_image.crop(bbox_padded)
            else:
                if init_image.size != control_image.size:
                    _messages.log(
                        'adetailer could not crop the control image correctly as it is a different '
                        'size from your input image, they need to be the same dimension, '
                        'reverting to resize only mode.')

                inpaint_args["control_image"] = control_image.resize(
                    crop_image.size
                )
        
        # Call pipeline and return just the image
        result = _pipelinewrapper.call_pipeline(
            pipeline=self.inpaint_pipeline,
            device=device,
            prompt_weighter=prompt_weighter,
            **inpaint_args)
        
        return result[0][0]
