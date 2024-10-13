Diffusion Model Feature Support Table
=====================================

   * ``--model-type torch`` (SD 1.5 - SD 2.*)
   * ``--model-type torch-pix2pix`` (SD 1.5 - SD 2.* - Pix2Pix)
   * ``--model-type torch-sdxl`` (Stable Diffusion XL)
   * ``--model-type torch-if`` (Deep Floyd Stage 1)
   * ``--model-type torch-ifs`` (Deep Floyd Stage 2)
   * ``--model-type torch-if-img2img`` (Deep Floyd Stage 2 - Img2Img)
   * ``--model-type torch-sdxl-pix2pix`` (Stable Diffusion XL - Pix2Pix)
   * ``--model-type torch-upscaler-x2`` (Stable Diffusion x2 Upscaler)
   * ``--model-type torch-upscaler-x4`` (Stable Diffusion x4 Upscaler)
   * ``--model-type torch-s-cascade`` (Stable Cascade)
   * ``--model-type torch-sd3`` (Stable Diffusion 3)
   * ``--model-type torch-flux`` (Flux.1)


.. list-table:: Supported Features by ``--model-type``
   :widths: 40 10 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - Model Type
     - Txt2Img
     - Img2Img
     - Inpainting
     - LoRA
     - Textual Inversions
     - ControlNet
     - T2I Adapter
     - IP Adapter
     - Perturbed Attenuation Guidance (PAG)
     - sd-embed Prompt Weighting
     - compel Prompt Weighting

   * - ``torch``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - ``torch-pix2pix``
     - ❌
     - ✅
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
     - ✅
     - ✅

   * - ``torch-sdxl``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - ``torch-if``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``torch-ifs``
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``torch-ifs-img2img``
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``torch-sdxl-pix2pix``
     - ❌
     - ✅
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅

   * - ``torch-upscaler-x2``
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``torch-upscaler-x4``
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅

   * - ``torch-s-cascade``
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅

   * - ``torch-sd3``
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌

   * - ``torch-flux``
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌


PAG Support Caveats
-------------------

PAG is supported for txt2img in all cases, but there are some edge
cases in which PAG is not supported.

There is no support for using T2I Adapters with PAG.

Stable Diffusion 3 does not currently support PAG in img2img mode, or with Control Nets at all.

Stable Diffusion XL does not support PAG in (inpaint + Control Nets) mode.

Stable Diffusion 1.5 - 2.* does not support PAG in img2img, inpaint, or (img2img + Control Nets) mode.
It does however support PAG in inpaint + Control Nets mode.

