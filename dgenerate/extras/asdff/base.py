import inspect

from typing import Any, Callable, Iterable, List, Mapping, Optional

from PIL import Image

from functools import cached_property

import diffusers

import dgenerate.extras.kolors
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

        if hasattr(self.pipe, 'DGENERATE_SIZE_ESTIMATE'):
            pipe.DGENERATE_SIZE_ESTIMATE = self.pipe.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(
            f'AdPipelineBase (adetailer pipeline) initialized pipeline class: {pipe.__class__.__name__}')

        return pipe

    def __call__(  # noqa: C901
            self,
            pipeline_args: Mapping[str, Any] | None = None,
            images: Image.Image | Iterable[Image.Image] | None = None,
            detectors: DetectorType | Iterable[DetectorType] | None = None,
            mask_dilation: int = 4,
            mask_blur: int = 4,
            mask_padding: int | tuple[int, int] | tuple[int, int, int, int] = 32,
            detector_padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
            model_path: str = None,
            device: str = 'cuda',
            detector_device: str = 'cuda',
            confidence: float = 0.3,
            mask_shape: str = 'rectangle',
            prompt_weighter: Optional[str] = None
    ):
        if pipeline_args is None:
            pipeline_args = {}

        if detectors is None:
            detectors = [self.default_detector]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

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

            for j, detector in enumerate(detectors):
                masks = detector(
                    init_image,
                    confidence=confidence,
                    device=detector_device,
                    model_path=model_path,
                    mask_shape=mask_shape,
                    padding=detector_padding
                )

                if masks is None:
                    _messages.log(
                        f"No object detected on {ordinal(i + 1)} image with {ordinal(j + 1)} detector."
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
                    inpaint_output = self.process_inpainting(
                        pipeline_args,
                        init_image,
                        control_image,
                        mask,
                        bbox_padded,
                        device,
                        prompt_weighter
                    )
                    inpaint_image = inpaint_output[0][0]
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
            self, pipeline_args: Mapping[str, Any]
    ):
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
            pipeline_args: Mapping[str, Any],
            init_image: Image.Image,
            control_image: Image.Image,
            mask: Image.Image,
            bbox_padded: tuple[int, int, int, int],
            device: str,
            prompt_weighter: Optional[str] = None
    ):
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
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
        return _pipelinewrapper.call_pipeline(
            pipeline=self.inpaint_pipeline,
            device=device,
            prompt_weighter=prompt_weighter,
            **inpaint_args)
