.. |Documentation| image:: https://readthedocs.org/projects/dgenerate/badge/?version=version_6.0.0
   :target: http://dgenerate.readthedocs.io/en/version_6.0.0/

.. |Latest Release| image:: https://img.shields.io/github/v/release/Teriks/dgenerate
   :target: https://github.com/Teriks/dgenerate/releases/latest
   :alt: GitHub Latest Release

.. |Support Dgenerate| image:: https://img.shields.io/badge/Ko–fi-support%20dgenerate%20-hotpink?logo=kofi&logoColor=white
   :target: https://ko-fi.com/teriks
   :alt: ko-fi

=========
dgenerate
=========

|Documentation| |Latest Release| |Support Dgenerate|

``dgenerate`` is a scriptable command-line tool (and library) for generating and editing images, generating whole video clips with LTX and Wan,
and processing animated inputs with AI.

Whether you're generating or editing single images, batch processing hundreds of variations, generating a short mp4 from a prompt,
or transforming entire videos frame-by-frame, dgenerate provides a flexible, scriptable interface for a multitude of generation and editing tasks.

For the extensive usage manual, manual installation guide, and API documentation, visit `readthedocs <http://dgenerate.readthedocs.io/en/version_6.0.0/>`_.

What You Can Do
===============

Image Generation
----------------

* Text to image, image to image, inpainting, and diffusion-based upscaling with SD 1.5 / 2.x, SDXL, SD3 / SD3.5, Stable Cascade, DeepFloyd IF, Kolors, Flux, Flux.2 (dev and Klein), Z-Image, Qwen-Image, Qwen-Image 2.1, Krea 2, and Kandinsky 5
* Edit images with instruction or reference based pipelines: Flux Fill and Kontext, Flux.2, Z-Image Omni, Qwen-Image Edit and Layered, and InstructPix2Pix (SD 1.5, SDXL, SD3)
* Control Nets, T2I Adapters, IP Adapters, LoRA, and Textual Inversion (embeddings), with models from HuggingFace and CivitAI
* Advanced prompt weighting (LPW), SD-WebUI (Common syntax), InvokeAI syntax, and ``llm4gen`` (SD1.5 only)
* Run large models on limited hardware with inference optimizations, quantization, and pre-quantized GGUF and ComfyUI FP8 / INT8 / NVFP4 checkpoints, with hardware accelerated quantized matmuls where supported
* Batch process parameter combinations combinatorially to generate variations

Video Generation
----------------

* Generate whole clips in one pipeline call: LTX-2.5 (``--model-type ltx2``) and LTX-Video (``--model-type ltx-video``), Wan 2.1 / 2.2 (``--model-type wan``), Hunyuan Video 1.5 (``hyvideo-15``), Kandinsky 5 Video (``kandinsky5-video``), and Motif Video (``motif-video``)
* Text-to-video, or image-to-video with a still in ``--image-seeds`` as the opening frame. With LTX, use a clip as the opening, ``last-frame=`` as the closing frame (a still or clip), or both, to extend existing footage or generate a lead-in; slice with ``--frame-start`` / ``--frame-end``
* Wan modes: image-to-video, first-last-frame, video-to-video, and VACE;``--model-type wan-animate`` and ``wan-animate-2`` animate a character still from a driving clip (``wan-driving=``, ``wan-pose=``, ``wan-face=``, ``--wan-animate-preprocess``)
* LTX-2.5 guided by an IC-LoRA from ``--ltx-ic-lora`` (canny, depth, pose, etc.) using a reference clip processed by ``--control-image-processors``, and style or effect LoRAs from ``--loras``
* Set clip length and frame rate with ``--video-lengths`` and ``--video-fps`` (LTX-2.5 can predict duration from the prompt, and muxes generated audio into mp4 output); compile blocks with ``--torch-compile``
* Example configs under ``examples/ltx2``, ``examples/ltx_video``, ``examples/wan``, ``examples/wan_animate``, and ``examples/wan_animate_2``; see the `video generation manual <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#video-generation>`_

Image Processing
----------------

* Easily chain image processors together for advanced scripted image manipulation
* Utilize built-in image processors for edge detection, depth mapping, segmentation, feature detection, and more
* Run upscaling / image restoration models such as ESRGAN, SwinIR, etc... via `spandrel <https://github.com/chaiNNer-org/spandrel>`_
* Run image processors generically on any image

Animation & video processing (per frame)
----------------------------------------

* Run image diffusion once per frame to transform videos into artistic, non-temporally consistent animations (distinct from whole-clip generation above)
* Process GIF, WebP, APNG, MP4, and any other video format supported by `av <https://github.com/PyAV-Org/PyAV>`_ (ffmpeg)
* Memory-efficient, streamed processing of video content from disk
* Apply image processors to any animated input, for example upscaling / classification / mask generation

Scripting
---------

* Utilize the built-in shell language to script generation tasks, work in REPL mode from the Console UI
* Write scripted workflows with intelligent VRAM/RAM memory management, garbage collection, and caching
* Write plugins such as image processors, prompt weighters, shell language features, etc. in Python if desired

Config generation
-----------------

A local Qwen model writes a dgenerate config from a plain language request. It retrieves the closest example configs and documentation, then checks the result with dgenerate and asks the model to fix anything that is rejected.

.. code-block:: bash

    dgenerate --sub-command assistant -o cat.dgen "generate a cute cat, 30 steps"

This needs the ``xllamacpp`` extra. The Console UI Generate Code menu runs the same command. See the `assistant sub-command <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#sub-command-assistant>`_ in the manual.

Getting Started
===============

Quick Install
-------------

Download an install wizard for your platform from the `releases page <https://github.com/Teriks/dgenerate/releases>`_ for a hassle-free setup into an isolated Python environment. The same binary installs from the command line. See the `network installer <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#network-installer>`_ section of the manual.

Manual Install
--------------

* `Windows <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#windows-install>`_
* `Linux / WSL <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#linux-or-wsl-install>`_
* `Linux ROCm <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#linux-with-rocm-amd-cards>`_
* `MacOS <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#macos-install-apple-silicon-only>`_
* `Google Colab <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#google-colab-install>`_
* `XPU (Intel) <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#install-with-xpu-support>`_
* `Installing From Development Branches <https://dgenerate.readthedocs.io/en/version_6.0.0/manual.html#installing-from-development-branches>`_

System Requirements
-------------------

* **GPU**: NVIDIA with CUDA 12.6 or newer, AMD (ROCm 7.2 or 7.14 on Linux, or AMD's Windows wheel index), or Apple Silicon
* **Python**: 3.11 or newer, except 3.14.1, and older than 3.15
* **OS**: Windows, macOS, or Linux

Note: CPU rendering is possible but extremely slow unless the given model is tailored for it.

Two Ways to Use dgenerate
=========================

Command Line
------------

Perfect for automation and batch processing:

.. code-block:: bash

    dgenerate stable-diffusion-v1-5/stable-diffusion-v1-5 --prompts "a cute cat" --inference-steps 15 20 30

    dgenerate --file workflow-config.dgen

Interactive GUI
---------------


.. code-block:: bash

    # launch the Console UI

    dgenerate --console


Features a syntax-highlighting console / editor:

* REPL / code editor for the built in shell language to assist with building complex workflows
* Generate Code writes a config script from a plain language request with a local Qwen model
* Preview plays finished GIF, WebP, APNG, and MP4 clips, including audio. The timeline under the picture has play, scrub, a speaker button (sound waves, or a red X when muted), and a volume slider
* Vulkan is the default preview on Windows and Linux with the ``console_ui_vulkan`` extra. ``console_ui_opengl`` is the Windows and Linux fallback when Vulkan is not installed. On macOS the preview is Metal and does not need an extra; set ``DGENERATE_CONSOLE_UI_METAL=0`` and ``DGENERATE_CONSOLE_UI_VULKAN=1`` to use Vulkan
* Smooth zoom / pan, and a bounding box / coordinate picker
* Various templating utilities (recipes, and URI builders) for quickly creating scripts and working interactively
* In editor documentation for all arguments, and built in image processors / plugins
* Lightweight multiplatform Tkinter-based UI

----

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui5.gif
   :alt: Console UI Demo