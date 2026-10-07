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
   * ``--model-type sd3-pix2pix`` (Stable Diffusion 3 - Pix2Pix [`UltraEdit <https://github.com/HaozheZhao/UltraEdit>`_])
   * ``--model-type flux`` (Flux.1)
   * ``--model-type flux-fill`` (Flux.1 - Infill / Outfill)
   * ``--model-type flux-kontext`` (Flux.1 - Pix2Pix like editing)
   * ``--model-type flux2`` (Flux.2 and Flux.2 Klein)
   * ``--model-type flux2-klein-kv`` (Flux.2 Klein KV)
   * ``--model-type z-image`` (Z-Image)
   * ``--model-type z-image-omni`` (Z-Image Omni)
   * ``--model-type qwen-image`` (Qwen-Image)
   * ``--model-type qwen-image-edit`` (Qwen-Image Edit and Edit-Plus)
   * ``--model-type qwen-image-layered`` (Qwen-Image Layered)
   * ``--model-type ltx`` (LTX-2.5 and LTX-Video, see `Video Model Feature Support`_)
   * ``--model-type wan`` (Wan 2.1 / 2.2 T2V, I2V, FLF2V, V2V, VACE)
   * ``--model-type wan-animate`` (Wan-Animate)

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

   * - ``flux2``
     - ✅
     - ❌
     - 🚧

   * - ``flux2-klein-kv``
     - ✅
     - ❌
     - ❌

   * - ``z-image``
     - ✅
     - ✅
     - ✅

   * - ``z-image-omni``
     - ✅
     - ❌
     - ❌

   * - ``qwen-image``
     - ✅
     - ✅
     - ✅

   * - ``qwen-image-edit``
     - ✅
     - ❌
     - 🚧

   * - ``qwen-image-layered``
     - ✅
     - ❌
     - ❌

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

   * - ``flux2``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``flux2-klein-kv``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``z-image``
     - ✅
     - ❌
     - ✅
     - ❌

   * - ``z-image-omni``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``qwen-image``
     - ✅
     - ❌
     - ✅
     - ❌

   * - ``qwen-image-edit``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``qwen-image-layered``
     - ✅
     - ❌
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

   * - ``flux2``
     - ❌
     - ❌

   * - ``flux2-klein-kv``
     - ❌
     - ❌

   * - ``z-image``
     - ❌
     - ❌

   * - ``z-image-omni``
     - ❌
     - ❌

   * - ``qwen-image``
     - ❌
     - ❌

   * - ``qwen-image-edit``
     - ❌
     - ❌

   * - ``qwen-image-layered``
     - ❌
     - ❌

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

   * - ``flux2``
     - ❌
     - ❌
     - ❌

   * - ``flux2-klein-kv``
     - ❌
     - ❌
     - ❌

   * - ``z-image``
     - ❌
     - ❌
     - ❌

   * - ``z-image-omni``
     - ❌
     - ❌
     - ❌

   * - ``qwen-image``
     - ❌
     - ❌
     - ❌

   * - ``qwen-image-edit``
     - ❌
     - ❌
     - ❌

   * - ``qwen-image-layered``
     - ❌
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

   * - ``flux2``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``flux2-klein-kv``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``z-image``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``z-image-omni``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``qwen-image``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``qwen-image-edit``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``qwen-image-layered``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

Video Model Feature Support
---------------------------

``--model-type ltx`` generates a whole clip in one pipeline call. The checkpoint's
``model_index.json`` selects LTX-2.5 (``LTX2Pipeline``) or the earlier LTX-Video (``LTXPipeline``).

.. list-table:: Features by LTX checkpoint
   :widths: 40 10 10
   :header-rows: 1

   * - Feature
     - LTX-2.5
     - LTX-Video

   * - Text to video
     - ✅
     - ✅

   * - First / last frame images
     - ✅
     - ✅

   * - Video conditioning
     - ✅
     - ✅

   * - Separate first / last frame processors
     - ✅
     - ✅

   * - IC-LoRA control
     - ✅
     - ❌

   * - LoRA with IC-LoRA
     - ✅
     - ❌

   * - Length predicted from prompt
     - ✅
     - ❌

   * - Audio
     - ✅
     - ❌

   * - Sigmas and audio guidance
     - ✅
     - ❌

   * - LoRA
     - ✅
     - ✅

   * - Quantization
     - ✅
     - ✅

   * - Replacement transformer
     - ✅
     - ✅

LTX does not support ControlNets, adapters, textual inversions, prompt weighters, inpainting,
or the acceleration features in the tables above. GGUF transformers are tested with LTX-Video.
See the video generation section in the manual for LTX-2.5 and the earlier LTX-Video checkpoint.

``--model-type wan``, ``--model-type wan-animate``, and ``--model-type wan-animate-2``
also generate a whole clip in one pipeline call. Clip length and frame rate are the
shared ``--video-lengths`` and ``--video-fps`` options.

.. list-table:: Features by Wan mode
   :widths: 40 10 10 12
   :header-rows: 1

   * - Feature
     - wan
     - wan-animate
     - wan-animate-2

   * - Text to video
     - ✅
     - ❌
     - ❌

   * - Image to video
     - ✅
     - ❌
     - ❌

   * - First / last frame (``last-frame=``)
     - ✅
     - ❌
     - ❌

   * - Video to video
     - ✅
     - ❌
     - ❌

   * - VACE (``control=`` / ``mask=`` / ``reference=``)
     - ✅
     - ❌
     - ❌

   * - Character animation (``wan-pose=`` / ``wan-face=``)
     - ❌
     - ✅
     - ❌

   * - ``wan-driving=`` as the motion clip
     - ❌
     - ❌
     - ✅

   * - ``wan-driving=`` preprocess (openpose + yolo)
     - ❌
     - ✅
     - ❌

   * - MoE second transformer
     - ✅
     - ❌
     - ❌

   * - LoRA
     - ✅
     - ✅
     - ✅

   * - Quantization / GGUF
     - ✅
     - ✅
     - ✅

Flow Image Model Notes
----------------------

``flux2``, ``flux2-klein-kv``, ``z-image-omni``, ``qwen-image-edit``, and
``qwen-image-layered`` take an ``--image-seeds`` value with no mask as a
condition image. That is the text-to-image column above. There is no img2img
strength.

``flux2`` inpainting is Flux.2 Klein only. Full Flux.2 has no inpaint pipeline.

``qwen-image-edit`` inpainting is the edit pipeline with a mask. Edit-plus has
no inpaint pipeline.

ControlNet on ``z-image`` and ``qwen-image`` is text-to-image and inpaint.
``z-image`` accepts one union model. ``qwen-image`` can take more than one.

These model types support LoRA. They do not support textual inversions, IP
adapters, T2I adapters, prompt weighters, PAG, or the acceleration features
in the generation features table.

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

``--torch-compile`` compiles repeated denoiser, ControlNet, and VAE decoder blocks. The first use of each waits while the graph builds. A later change of resolution or frame count compiles once more, then reuses that graph. CUDA and XPU need Triton. On macOS, PyTorch 2.13 and newer compile these blocks to a Metal kernel. CPU and quantized weights stay eager, and dgenerate warns. Text encoders, image encoders, and the LTX diffusion decoder stay eager. Wan-Animate-2 compiles its transformer and VAE blocks on every load. ``DGENERATE_TORCH_COMPILE=0`` leaves every model eager, including Wan-Animate-2.

Faster generation speeds can be achieved by using DeepCache, Microsoft RAS, or TeaCache, but may lead to reduced image quality. The default values for each of these features are conservative, providing some speed increases without major impacts on quality.

The DeepCache branch ID and interval can be specified with the ``--deep-cache-branch-ids`` and ``--deep-cache-intervals`` options. Benchmarks for different parameters can be reviewed `here <https://huggingface.co/docs/diffusers/main/en/optimization/deepcache#benchmark>`__.

Microsoft Region-Adaptive Sampling (RAS) has numerous configurable options that can be reviewed `here <https://github.com/microsoft/ras?tab=readme-ov-file#customize-hyperparameters>`__. Note that the ``--ras-index-fusion`` parameter is not compatible with SD3.5.

The TeaCache threshold can be specified with the ``--tea-cache-rel-l1-thresholds`` parameter. Information about this parameter can be reviewed `here <https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/README.md>`__.
