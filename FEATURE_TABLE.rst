Diffusion Model Feature Support Tables
======================================

   * ``--model-type sd`` (SD 1.5 - SD 2.*)
   * ``--model-type pix2pix`` (SD 1.5 - SD 2.* - Pix2Pix)
   * ``--model-type sdxl`` (Stable Diffusion XL)
   * ``--model-type kolors`` (Kolors)
   * ``--model-type if`` (Deep Floyd Stage 1)
   * ``--model-type ifs`` (Deep Floyd Stage 2)
   * ``--model-type ifs-img2img`` (Deep Floyd Stage 2 - Img2Img)
   * ``--model-type sdxl-pix2pix`` (Stable Diffusion XL - Pix2Pix)
   * ``--model-type upscaler-x2`` (Stable Diffusion x2 Upscaler)
   * ``--model-type upscaler-x4`` (Stable Diffusion x4 Upscaler)
   * ``--model-type s-cascade`` (Stable Cascade)
   * ``--model-type sd3`` (Stable Diffusion 3 and 3.5)
   * ``--model-type sd3-pix2pix`` (Stable Diffusion 3 - Pix2Pix [UltraEdit])
   * ``--model-type flux`` (Flux.1)
   * ``--model-type flux-fill`` (Flux.1 - Infill / Outfill)
   * ``--model-type flux-kontext`` (Flux.1 - Pix2Pix like editing)


.. list-table:: Generation modes by ``--model-type``
   :widths: 40 10 10 10
   :header-rows: 1

   * - Model Type
     - Txt2Img
     - Img2Img
     - Inpainting

   * - ``sd``
     - ✅
     - ✅
     - ✅

   * - ``pix2pix``
     - ❌
     - ✅
     - 🚧

   * - ``sdxl``
     - ✅
     - ✅
     - ✅

   * - ``kolors``
     - ✅
     - ✅
     - ✅

   * - ``if``
     - ✅
     - ✅
     - ✅

   * - ``ifs``
     - ❌
     - ✅
     - ✅

   * - ``ifs-img2img``
     - ❌
     - ✅
     - ✅

   * - ``sdxl-pix2pix``
     - ❌
     - ✅
     - 🚧

   * - ``upscaler-x2``
     - ❌
     - ✅
     - ❌

   * - ``upscaler-x4``
     - ❌
     - ✅
     - ❌

   * - ``s-cascade``
     - ✅
     - ✅
     - ❌

   * - ``sd3``
     - ✅
     - ✅
     - ✅

   * - ``sd3-pix2pix``
     - ❌
     - ✅
     - ✅

   * - ``flux``
     - ✅
     - ✅
     - ✅

   * - ``flux-fill``
     - ❌
     - ❌
     - ✅

   * - ``flux-kontext``
     - ❌
     - ✅
     - ✅

.. list-table:: Guidance by ``--model-type``
   :widths: 40 10 10 10 10
   :header-rows: 1

   * - Model Type
     - LoRA
     - Textual Inversions
     - ControlNet
     - Perturbed Attention Guidance (PAG)

   * - ``sd``
     - ✅
     - ✅
     - ✅
     - ✅

   * - ``pix2pix``
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``sdxl``
     - ✅
     - ✅
     - ✅
     - ✅

   * - ``kolors``
     - ✅
     - ❌
     - ✅
     - ✅

   * - ``if``
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``ifs``
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``ifs-img2img``
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sdxl-pix2pix``
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``upscaler-x2``
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``upscaler-x4``
     - ❌
     - ✅
     - ❌
     - ❌

   * - ``s-cascade``
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sd3``
     - ✅
     - ❌
     - ✅
     - ✅

   * - ``sd3-pix2pix``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``flux``
     - ✅
     - ✅
     - ✅
     - ❌

   * - ``flux-fill``
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``flux-kontext``
     - ✅
     - ✅
     - ❌
     - ❌

.. list-table:: Adapters by ``--model-type``
   :widths: 40 10 10
   :header-rows: 1

   * - Model Type
     - T2I Adapter
     - IP Adapter

   * - ``sd``
     - ✅
     - ✅

   * - ``pix2pix``
     - ❌
     - ✅

   * - ``sdxl``
     - ✅
     - ✅

   * - ``kolors``
     - ❌
     - ✅

   * - ``if``
     - ❌
     - ❌

   * - ``ifs``
     - ❌
     - ❌

   * - ``ifs-img2img``
     - ❌
     - ❌

   * - ``sdxl-pix2pix``
     - ❌
     - ❌

   * - ``upscaler-x2``
     - ❌
     - ❌

   * - ``upscaler-x4``
     - ❌
     - ❌

   * - ``s-cascade``
     - ❌
     - ❌

   * - ``sd3``
     - ❌
     - ❌

   * - ``sd3-pix2pix``
     - ❌
     - ❌

   * - ``flux``
     - ❌
     - ✅

   * - ``flux-fill``
     - ❌
     - ❌

   * - ``flux-kontext``
     - ❌
     - ✅

.. list-table:: Prompt enhancement by ``--model-type``
   :widths: 40 10 10 10
   :header-rows: 1

   * - Model Type
     - sd-embed Prompt Weighting
     - compel Prompt Weighting
     - llm4gen Prompt Weighting

   * - ``sd``
     - ✅
     - ✅
     - ✅

   * - ``pix2pix``
     - ✅
     - ✅
     - ✅

   * - ``sdxl``
     - ✅
     - ✅
     - ❌

   * - ``kolors``
     - ❌
     - ❌
     - ❌

   * - ``if``
     - ❌
     - ❌
     - ❌

   * - ``ifs``
     - ❌
     - ❌
     - ❌

   * - ``ifs-img2img``
     - ❌
     - ❌
     - ❌

   * - ``sdxl-pix2pix``
     - ✅
     - ✅
     - ❌

   * - ``upscaler-x2``
     - ❌
     - ❌
     - ❌

   * - ``upscaler-x4``
     - ✅
     - ✅
     - ✅

   * - ``s-cascade``
     - ✅
     - ✅
     - ❌

   * - ``sd3``
     - ✅
     - ❌
     - ❌

   * - ``sd3-pix2pix``
     - ✅
     - ❌
     - ❌

   * - ``flux``
     - ✅
     - ❌
     - ❌

   * - ``flux-fill``
     - ✅
     - ❌
     - ❌

   * - ``flux-kontext``
     - ✅
     - ❌
     - ❌

.. list-table:: Generation Features by ``--model-type``
   :widths: 40 30 20 40 30 40 30
   :header-rows: 1

   * - Model Type
     - ADetailer
     - FreeU
     - Hi-Diffusion
     - DeepCache
     - Microsoft RAS
     - TeaCache

   * - ``sd``
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``pix2pix``
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌

   * - ``sdxl``
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``kolors``
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌

   * - ``if``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``ifs``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``ifs-img2img``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sdxl-pix2pix``
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌

   * - ``upscaler-x2``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``upscaler-x4``
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌

   * - ``s-cascade``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sd3``
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌

   * - ``sd3-pix2pix``
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌

   * - ``flux``
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``flux-fill``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``flux-kontext``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

PAG Support Caveats
-------------------

PAG is supported for txt2img in all cases, but there are some edge
cases in which PAG is not supported.

There is no support for using T2I Adapters with PAG.

Stable Diffusion 3 does not currently support PAG with ControlNets at all.

Stable Diffusion XL does not support PAG in (inpaint + ControlNets) mode.

Stable Diffusion 1.5 - 2.* does not support PAG in img2img, inpaint, or (img2img + ControlNets) mode.
It does however support PAG in (inpaint + ControlNets) mode.

Kolors only supports PAG in txt2img mode.

Generation Feature Notes
------------------------

FreeU parameters differ by model type and can be specified using the ``--freeu-params`` option. The recommended parameters for SD1.4, SD1.5, SD2.1, and SDXL can be reviewed `here <https://github.com/ChenyangSi/FreeU?tab=readme-ov-file#parameters>`__. Kolors is compatible with FreeU's SDXL settings.

Faster generation speeds can be achieved by using DeepCache, Microsoft RAS, or TeaCache, but may lead to reduced image quality. The default values for each of these features are conservative, providing some speed increases without major impacts on quality.

The DeepCache branch ID and interval can be specified with the ``--deep-cache-branch-ids`` and ``--deep-cache-intervals`` options. Benchmarks for different parameters can be reviewed `here <https://huggingface.co/docs/diffusers/main/en/optimization/deepcache#benchmark>`__.

Microsoft Region-Adaptive Sampling (RAS) has numerous configurable options that can be reviewed `here <https://github.com/microsoft/ras?tab=readme-ov-file#customize-hyperparameters>`__. Note that the ``--ras-index-fusion`` parameter is not compatible with SD3.5.

The TeaCache threshold can be specified with the ``--tea-cache-rel-l1-thresholds`` parameter. Information about this parameter can be reviewed `here <https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/README.md>`__.
