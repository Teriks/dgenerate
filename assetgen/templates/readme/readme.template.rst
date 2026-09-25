.. |Documentation| image:: https://readthedocs.org/projects/dgenerate/badge/?version=@REVISION
   :target: http://dgenerate.readthedocs.io/en/@REVISION/

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

``dgenerate`` is a scriptable command-line tool (and library) for generating and editing images, generating whole video clips with LTX,
and processing animated inputs with AI.

Whether you're generating or editing single images, batch processing hundreds of variations, generating a short mp4 from a prompt,
or transforming entire videos frame-by-frame, dgenerate provides a flexible, scriptable interface for a multitude of generation and editing tasks.

For the extensive usage manual, manual installation guide, and API documentation, visit `readthedocs <http://dgenerate.readthedocs.io/en/@REVISION/>`_.

What You Can Do
===============

Image Generation
----------------

* Generate images using a number of popular model architectures such as: SD, SDXL, SD3, Flux, and Kolors
* Batch process multiple parameter combinations combinatorially to generate variations
* Run large models on limited hardware with inference optimizations and quantization
* Utilize models from HuggingFace and CivitAI for generation
* Advanced prompt weighting (LPW), SD-WebUI (Common syntax), InvokeAI syntax, and ``llm4gen`` (SD1.5 only)
* Control Nets, T2I Adapters, IP Adapters, LoRA, and Textual Inversion (embeddings)
* Text to image, image to image, and inpainting
* Diffusion-based image upscaling

Video generation (LTX)
----------------------

* Generate a whole clip in one pipeline call with ``--model-type ltx`` (`Lightricks LTX-2.5 <https://huggingface.co/Lightricks/LTX-2.5-Diffusers>`_ and the earlier `LTX-Video <https://huggingface.co/Lightricks/LTX-Video>`_ checkpoints)
* Text-to-video, or condition with ``--image-seeds``: a still or clip as the opening frames, ``end=`` as the closing frame, or both together
* Use a video, GIF, or other animated file as conditioning to extend existing footage or generate a lead-in; slice with ``--frame-start`` / ``--frame-end``
* Guide LTX-2.5 with an IC-LoRA from ``--ic-lora``, such as canny, depth, or pose control, from a reference clip processed by ``--control-image-processors``
* Set clip length and frame rate with ``--video-lengths`` and ``--video-fps``; LTX-2.5 can predict duration from the prompt when length is omitted
* LTX-2.5 muxes generated audio into mp4 output; the earlier LTX-Video model accepts a GGUF diffusion transformer
* Example configs under ``examples/ltx``; see the `video generation manual <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#video-generation>`_

Image Processing
----------------

* Easily chain image processors together for advanced scripted image manipulation
* Utilize built-in image processors for edge detection, depth mapping, segmentation, feature detection, and more
* Run upscaling / image restoration models such as ESRGAN, SwinIR, etc... via `spandrel <https://github.com/chaiNNer-org/spandrel>`_
* Run image processors generically on any image

Animation & video processing (per frame)
----------------------------------------

* Run image diffusion once per frame to transform videos into artistic, non-temporally consistent animations (distinct from LTX whole-clip generation above)
* Process GIF, WebP, APNG, MP4, and any other video format supported by `av <https://github.com/PyAV-Org/PyAV>`_ (ffmpeg)
* Memory-efficient, streamed processing of video content from disk
* Apply image processors to any animated input, for example upscaling / classification / mask generation

Scripting
---------

* Utilize the built-in shell language to script generation tasks, work in REPL mode from the Console UI
* Write scripted workflows with intelligent VRAM/RAM memory management, garbage collection, and caching
* Write plugins such as image processors, prompt weighters, shell language features, etc. in Python if desired

Getting Started
===============

Quick Install
-------------

Download an install wizard for your platform from the `releases page <https://github.com/Teriks/dgenerate/releases>`_ for a hassle-free setup into an isolated Python environment.

Manual Install
--------------

* `Windows <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#windows-install>`_
* `Linux / WSL <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#linux-or-wsl-install>`_
* `Linux ROCm <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#linux-with-rocm-amd-cards>`_
* `MacOS <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#macos-install-apple-silicon-only>`_
* `Google Colab <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#google-colab-install>`_
* `XPU (Intel) <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#install-with-xpu-support>`_
* `Installing From Development Branches <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#installing-from-development-branches>`_

System Requirements
-------------------

* **GPU**: NVIDIA (CUDA 12.1+), AMD (ROCm on Linux), or Apple Silicon
* **Python**: 3.11 to 3.13
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
* OpenGL accelerated image preview, featuring smooth zoom / pan, and bounding box / coordinate picker
* Various templating utilities (recipes, and URI builders) for quickly creating scripts and working interactively
* In editor documentation for all arguments, and built in image processors / plugins
* Lightweight multiplatform Tkinter-based UI

----

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui5.gif
   :alt: Console UI Demo