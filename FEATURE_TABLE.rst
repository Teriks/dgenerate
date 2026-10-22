Diffusion Model Feature Support Tables
======================================

   * ``--model-type sd`` (Stable Diffusion 1.5 - 2.*)
   * ``--model-type pix2pix`` (Stable Diffusion 1.5 - 2.* - Pix2Pix)
   * ``--model-type sdxl`` (Stable Diffusion XL)
   * ``--model-type kolors`` (Kolors)
   * ``--model-type if`` (DeepFloyd IF Stage 1)
   * ``--model-type ifs`` (DeepFloyd IF Stage 2)
   * ``--model-type ifs-img2img`` (DeepFloyd IF Stage 2 - Img2Img)
   * ``--model-type sdxl-pix2pix`` (Stable Diffusion XL - Pix2Pix)
   * ``--model-type upscaler-x2`` (Stable Diffusion x2 Upscaler)
   * ``--model-type upscaler-x4`` (Stable Diffusion x4 Upscaler)
   * ``--model-type s-cascade`` (Stable Cascade)
   * ``--model-type sd3`` (Stable Diffusion 3 and 3.5)
   * ``--model-type sd3-pix2pix`` (Stable Diffusion 3 - Pix2Pix [`UltraEdit <https://github.com/HaozheZhao/UltraEdit>`_])
   * ``--model-type flux`` (Flux.1)
   * ``--model-type flux-fill`` (Flux.1 Fill - Infill / Outfill)
   * ``--model-type flux-kontext`` (Flux.1 Kontext - Pix2Pix-like editing)
   * ``--model-type flux2`` (Flux.2 and Flux.2 Klein)
   * ``--model-type flux2-klein-kv`` (Flux.2 Klein KV)
   * ``--model-type z-image`` (Z-Image)
   * ``--model-type z-image-omni`` (Z-Image Omni)
   * ``--model-type qwen-image`` (Qwen-Image)
   * ``--model-type qwen-image-edit`` (Qwen-Image Edit and Edit-Plus)
   * ``--model-type qwen-image-layered`` (Qwen-Image Layered)
   * ``--model-type qwen-image-21`` (Qwen-Image 2.1)
   * ``--model-type krea2`` (Krea 2)
   * ``--model-type kandinsky5-image`` (Kandinsky 5.0 Image)
   * ``--model-type ltx`` (LTX-2.5 and LTX-Video, see `Video Model Feature Support`_)
   * ``--model-type wan`` (Wan 2.1 / 2.2 - T2V, I2V, FLF2V, V2V, VACE, see `Video Model Feature Support`_)
   * ``--model-type wan-animate`` (Wan-Animate, see `Video Model Feature Support`_)
   * ``--model-type wan-animate-2`` (Wan-Animate-2, see `Video Model Feature Support`_)
   * ``--model-type hunyuan-video-15`` (HunyuanVideo 1.5, see `Video Model Feature Support`_)
   * ``--model-type motif-video`` (Motif-Video, see `Video Model Feature Support`_)
   * ``--model-type kandinsky5-video`` (Kandinsky 5.0 Video, see `Video Model Feature Support`_)

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

   * - ``qwen-image-21``
     - ✅
     - ❌
     - ❌

   * - ``krea2``
     - ✅
     - ❌
     - ❌

   * - ``kandinsky5-image``
     - ✅
     - ✅
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

   * - ``qwen-image-21``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``krea2``
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``kandinsky5-image``
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

   * - ``qwen-image-21``
     - ❌
     - ❌

   * - ``krea2``
     - ❌
     - ❌

   * - ``kandinsky5-image``
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

   * - ``qwen-image-21``
     - ❌
     - ❌
     - ❌

   * - ``krea2``
     - ❌
     - ❌
     - ❌

   * - ``kandinsky5-image``
     - ❌
     - ❌
     - ❌

.. list-table:: Generation features by ``--model-type``
   :widths: 25 12 16 22 18 28 24 12 24 24 24
   :header-rows: 1

   * - Model Type
     - DyPE
     - Cache-DiT
     - ADetailer
     - FreeU
     - Hi-Diffusion
     - DeepCache
     - Microsoft RAS
     - SADA
     - TeaCache
     - Spectrum

   * - ``sd``
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌

   * - ``pix2pix``
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sdxl``
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅

   * - ``kolors``
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅

   * - ``if``
     - ❌
     - ❌
     - ❌
     - ❌
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
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``upscaler-x2``
     - ❌
     - ❌
     - ❌
     - ✅
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
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``s-cascade``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``sd3``
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅

   * - ``sd3-pix2pix``
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅

   * - ``flux``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - ``flux-fill``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - ``flux-kontext``
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - ``flux2``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``flux2-klein-kv``
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

   * - ``z-image``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``z-image-omni``
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

   * - ``qwen-image``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``qwen-image-edit``
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``qwen-image-layered``
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

   * - ``qwen-image-21``
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

   * - ``krea2``
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

   * - ``kandinsky5-image``
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

Video Model Feature Support
---------------------------

Every video model type (``ltx``, ``wan``, ``wan-animate``, ``wan-animate-2``,
``hunyuan-video-15``, ``motif-video``, and ``kandinsky5-video``) generates a whole
clip in one pipeline call. Clip length and frame rate are the shared ``--video-lengths``
and ``--video-fps`` options.

For ``--model-type ltx``, the checkpoint's ``model_index.json`` selects LTX-2.5
(``LTX2Pipeline``) or the earlier LTX-Video (``LTXPipeline``).

.. list-table:: Features by video model
   :widths: 34 8 8 8 10 10 10 10 10
   :header-rows: 1

   * - Feature
     - LTX-2.5 (``ltx``)
     - LTX-Video (``ltx``)
     - Wan (``wan``)
     - Wan-Animate (``wan-animate``)
     - Wan-Animate-2 (``wan-animate-2``)
     - HunyuanVideo 1.5 (``hunyuan-video-15``)
     - Motif-Video (``motif-video``)
     - Kandinsky 5.0 Video (``kandinsky5-video``)

   * - Text to video
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - Image to video
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅

   * - First / last frame images
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - First / last frame (``last-frame=``)
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Video conditioning
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Separate first / last frame processors
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Video to video
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - VACE (``control=`` / ``mask=`` / ``reference=``)
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Character animation (``wan-pose=`` / ``wan-face=``)
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌

   * - ``wan-driving=`` as the motion clip
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌

   * - ``wan-driving=`` preprocess (openpose + yolo)
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌

   * - IC-LoRA control
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - LoRA with IC-LoRA
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Length predicted from prompt
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Audio output
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Sigmas and audio guidance
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - LoRA
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - Quantization / GGUF
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

   * - Replacement transformer
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Cache-DiT
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅

   * - MoE second transformer
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

   * - Spectrum
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌

   * - DyPE
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌

   * - Torch compile
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅

LTX does not support ControlNets, adapters, textual inversions, prompt weighters, or inpainting.
Cache-DiT is supported for LTX; other acceleration features in the tables above are not. GGUF transformers
are tested with LTX-Video. LTX video conditioning covers first and last frames, reference audio, and IC-LoRA.

Wan video-to-video is the only Wan mode that uses ``--timesteps``. Wan timesteps and audio guidance follow
the Wan pipeline's scheduler. ``wan-animate-2`` expects a character still plus a driving clip.

HunyuanVideo 1.5 supports text-to-video and single-image-to-video. Motif-Video supports text-to-video and
first-frame image-to-video; ``--timesteps`` is Motif-Video specific. Kandinsky 5.0 Video has Lite and Pro
checkpoints for text-to-video and image-to-video. GGUF transformer replacement is available for all three;
the only published Kandinsky 5.0 Video GGUF checkpoints are T2V Pro. These three pipelines do not return
audio in dgenerate.

Flow Image Model Notes
----------------------

``flux2``, ``flux2-klein-kv``, ``z-image-omni``, ``qwen-image-edit``, and
``qwen-image-layered`` take an ``--image-seeds`` value with no mask as a
condition image. That is the Txt2Img column above. There is no img2img
strength.

``flux2`` inpainting is Flux.2 Klein only. Full Flux.2 has no inpaint pipeline.

``qwen-image-edit`` inpainting is the edit pipeline with a mask. Edit-Plus has
no inpaint pipeline.

``qwen-image-21`` accepts optional reference images but has no img2img or inpaint
pipeline. ``krea2`` is text-to-image only. ``kandinsky5-image`` supports text-to-image
and image-to-image editing, but not inpainting. All three support latent interchange
and cooperative denoising; see their examples under ``examples/latents_interchange``.

ControlNet on ``z-image`` and ``qwen-image`` is text-to-image and inpaint.
``z-image`` accepts one union model. ``qwen-image`` can take more than one.

The model-specific rows above show LoRA, Cache-DiT, DyPE, and Spectrum support. In particular,
``qwen-image-21`` supports Spectrum but not Cache-DiT or DyPE; ``kandinsky5-image`` supports
Cache-DiT and Spectrum but not DyPE; ``krea2`` supports all three. These flow image models do not
support textual inversions, IP adapters, T2I adapters, prompt weighters, PAG, DeepCache, TeaCache,
SADA, Microsoft RAS, or Hi-Diffusion.

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

Faster generation speeds can be achieved by using Cache-DiT, DyPE, DeepCache, Microsoft RAS, SADA, TeaCache, or Spectrum, but may lead to reduced image quality. The default values for each of these features are conservative, providing some speed increases without major impacts on quality.

The DeepCache branch ID and interval can be specified with the ``--deep-cache-branch-ids`` and ``--deep-cache-intervals`` options. Benchmarks for different parameters can be reviewed `here <https://huggingface.co/docs/diffusers/main/en/optimization/deepcache#benchmark>`__.

Microsoft Region-Adaptive Sampling (RAS) has numerous configurable options that can be reviewed `here <https://github.com/microsoft/ras?tab=readme-ov-file#customize-hyperparameters>`__. Note that the ``--ras-index-fusion`` parameter is not compatible with SD3.5.

The TeaCache threshold can be specified with the ``--tea-cache-rel-l1-thresholds`` parameter. Information about this parameter can be reviewed `here <https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/README.md>`__.

SADA cannot be combined with Hi-Diffusion, DeepCache, or TeaCache.

DyPE (``--dype``) adapts spatial RoPE frequencies during denoising. Use ``--dype-methods`` to try ``yarn`` (default), ``ntk``, or the ``base`` RoPE baseline as combinatoric values. The Flux.1 and Qwen-Image adapters follow the upstream architecture; Flux.1 Fill / Kontext, Qwen-Image Edit / Layered, Flux.2, Z-Image, Wan, Krea 2, HunyuanVideo 1.5, and Motif-Video are experimental adapters. Reference long-side defaults are 1024 pixels for Flux.1, Flux.2, and Z-Image, 1328 for Qwen-Image, and 832 for Wan. ``--dype-mscales`` controls the ``yarn`` method's attention magnitude scale: ``constant`` (default), ``decay``, or ``off``. Use ``decay`` or ``off`` if ``yarn`` produces a fine repeating texture, as it does with FLUX.1-dev. ``--dype-axis-scalings`` controls non-square images: ``shared`` (default) applies the long side's scaling to both axes, as ComfyUI-DyPE does; ``independent`` scales each axis separately, as upstream DyPE does. DyPE cannot be combined with Cache-DiT, TeaCache, Spectrum, or SADA, because those features reuse or skip denoiser work from an earlier step. See the `DyPE source <https://github.com/guyyariv/DyPE>`__ and the vendored adapter notes under ``dgenerate/extras/dype``.

Spectrum (``--spectrum``) forecasts denoiser block outputs and skips those blocks on later steps. Transformers skip their blocks. SDXL and Kolors skip the UNet down, mid, and up blocks and still run the output convolution. It is supported on SD3, SDXL, Kolors, Flux.1, Flux.2, Z-Image, Qwen-Image (including 2.1), Krea 2, Kandinsky 5.0 Image, Wan, HunyuanVideo 1.5, and Motif-Video. The moderate schedule is the default (window ``2``, flex ``0.75``). ``--spectrum-flex-windows 3`` is the paper's aggressive schedule. Details are at `Spectrum <https://github.com/hanjq17/Spectrum>`__.