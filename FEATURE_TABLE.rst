Model Feature Support Table
===========================

.. list-table:: Supported Features by Model Type
   :widths: 40 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - Model Type
     - Txt2Img
     - Img2Img
     - Inpainting
     - LoRA
     - ControlNet
     - T2I Adapter
     - IP Adapter
     - sd-embed Prompt Weighting
     - compel Prompt Weighting

   * - torch
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - torch-pix2pix
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - torch-sdxl
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - torch-if
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - torch-ifs
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - torch-ifs-img2img
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - torch-sdxl-pix2pix
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅

   * - torch-upscaler-x2
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - torch-upscaler-x4
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅

   * - torch-s-cascade
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - torch-sd3
     - ✅
     - ✅
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌

   * - torch-flux
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌