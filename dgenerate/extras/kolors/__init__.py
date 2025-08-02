from dgenerate.extras.kolors.pipelines.pipeline_kolors_inpainting import KolorsInpaintPipeline
from dgenerate.extras.kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import KolorsControlNetImg2ImgPipeline
from dgenerate.extras.kolors.pipelines.pipeline_controlnet_xl_kolors import KolorsControlNetPipeline
from dgenerate.extras.kolors.pipelines.pipeline_controlnet_xl_kolors_inpaint import KolorsControlNetInpaintPipeline

import diffusers.pipelines.auto_pipeline

# create these entries globally due to some dependencies such as hi-diffusion
# relying on them for settings determination

diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.update(
    {
        'kolors': diffusers.KolorsImg2ImgPipeline,
        'kolors-control': KolorsControlNetImg2ImgPipeline
    }
)

diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.update(
    {
        'kolors': KolorsInpaintPipeline,
        'kolors-control': KolorsControlNetInpaintPipeline
    }
)

diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING.update(
    {
        'kolors': diffusers.KolorsPipeline,
        'kolors-pag': diffusers.KolorsPAGPipeline,
        'kolors-control': KolorsControlNetPipeline
    }
)