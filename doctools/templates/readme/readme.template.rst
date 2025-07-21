.. |Documentation| image:: https://readthedocs.org/projects/dgenerate/badge/?version=v@VERSION
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

``dgenerate`` is a scriptable command-line tool (and library) for generating/editing images and processing animated inputs with AI.

Whether you're generating or editing single images, batch processing hundreds of variations, or transforming entire videos frame-by-frame,
dgenerate provides a flexible, scriptable interface for a multitude of image generation and editing tasks.

For the extensive usage manual, manual installation guide, and API documentation, visit `readthedocs <http://dgenerate.readthedocs.io/en/@REVISION/>`_.

What You Can Do
===============

Image Generation
----------------

* Generate images using a number of popular model architectures such as: SD, SDXL, SD3, Flux, and Kolors
* Batch process multiple parameter combinations combinatorially to generate variations
* Utilize models from HuggingFace and CivitAI for generation
* Advanced prompt weighting (LPW), generation control & guidance
* Text to image, image to image, and inpainting with diffusion
* Diffusion-based image upscaling
* Run large models on limited hardware with inference optimizations and quantization

Image Processing
----------------

* Easily chain image processors together for advanced scripted image manipulation
* Utilize built-in image processors for edge detection, depth mapping, segmentation, feature detection, and more
* Run upscaling / image restoration models such as ESRGAN, SwinIR, etc... via `spandrel <https://github.com/chaiNNer-org/spandrel>`_
* Run image processors generically on any image

Animation & Video Processing
----------------------------

* Transform videos into artistic non-temporally consistent animations
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

Quick Install (Windows)
------------------------

Download the Windows Installer MSI (multipart zip) from the `releases page <https://github.com/Teriks/dgenerate/releases>`_ for a hassle-free setup into a frozen / isolated Python environment.

Manual Install
--------------

* `Windows <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#windows-install>`_
* `Linux / WSL <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#linux-or-wsl-install>`_
* `Linux ROCm <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#linux-with-rocm-amd-cards>`_
* `MacOS <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#macos-install-apple-silicon-only>`_
* `Google Colab <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#google-colab-install>`_
* `Installing From Development Branches <https://dgenerate.readthedocs.io/en/@REVISION/manual.html#installing-from-development-branches>`_

System Requirements
-------------------

* **GPU**: NVIDIA (CUDA 12.1+), AMD (ROCm on Linux), or Apple Silicon
* **Python**: 3.10 to 3.13
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

Features a syntax-highlighting console / editor:

* REPL / code editor for the built in shell language to assist with building complex workflows
* Image preview and various utilities for quickly creating scripts and working interactively
* Lightweight multiplatform Tkinter-based UI

----

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui5.gif
   :alt: Console UI Demo