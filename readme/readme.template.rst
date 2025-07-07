.. |Documentation| image:: https://readthedocs.org/projects/dgenerate/badge/?version=v@VERSION
   :target: http://dgenerate.readthedocs.io/en/@REVISION/

.. |Latest Release| image:: https://img.shields.io/github/v/release/Teriks/dgenerate
   :target: https://github.com/Teriks/dgenerate/releases/latest
   :alt: GitHub Latest Release

.. |Support Dgenerate| image:: https://img.shields.io/badge/Ko–fi-support%20dgenerate%20-hotpink?logo=kofi&logoColor=white
   :target: https://ko-fi.com/teriks
   :alt: ko-fi

Overview
========

|Documentation| |Latest Release| |Support Dgenerate|

``dgenerate`` is a cross-platform command line tool and library for generating images
and animation sequences using Stable Diffusion and related models.

Alongside the command line tool, this project features a syntax-highlighting
REPL Console UI for the dgenerate configuration / scripting language, which is built on
Tkinter to be lightweight and portable. This GUI serves as an interface to dgenerate running
in the background via the ``--shell`` option.

You can use dgenerate to generate multiple images or animated outputs using multiple
combinations of diffusion input parameters in batch, so that the differences in
generated output can be compared / curated easily.  This can be accomplished via a single command,
or through more advanced scripting with the built-in interpreted shell-like language if needed.

Animated output can be produced by processing every frame of a Video, GIF, WebP, or APNG through
various implementations of diffusion in img2img or inpainting mode, as well as with ControlNets and
control guidance images, in any combination thereof. MP4 (h264) video can be written without memory
constraints related to frame count. GIF, WebP, and PNG/APNG can be written WITH memory constraints,
IE: all frames exist in memory at once before being written.

Video input of any runtime can be processed without memory constraints related to the video size.
Many video formats are supported through the use of PyAV (ffmpeg).

Animated image input such as GIF, APNG (extension must be .apng), and WebP, can also be processed
WITH memory constraints, IE: all frames exist in memory at once after an animated image is read.

PNG, JPEG, JPEG-2000, TGA (Targa), BMP, and PSD (Photoshop) are supported for static image inputs.

In addition to diffusion, dgenerate also supports the processing of any supported image, video, or
animated image using any of its built-in image processors, which include various edge detectors,
depth detectors, segment generation, normal map generation, pose detection, non-diffusion based
AI upscaling, and more.  dgenerate's image processors may be used to pre-process image / video
input to diffusion, post-process diffusion output, or to process images and video directly.

dgenerate brings many major features of the HuggingFace ``diffusers`` library directly to the
command line in a very flexible way with a near one-to-one mapping, akin to ffmpeg, allowing
for creative uses as powerful as direct implementation in python with less effort and
environmental setup.

dgenerate is compatible with HuggingFace as well as typical CivitAI-hosted models,
prompt weighting and many other useful generation features are supported.

dgenerate can be easily installed on Windows via a Windows Installer MSI containing a
frozen python environment, making setup for Windows users easy, and likely to "just work"
without any dependency issues. This installer can be found in the release artifact under each
release located on the `github releases page <https://github.com/Teriks/dgenerate/releases>`_.

This software requires a Nvidia GPU supporting CUDA 12.1+, AMD GPU supporting ROCm (Linux Only),
or MacOS on Apple Silicon, and supports ``python>=3.10,<3.14``. CPU rendering is possible for
some operations but extraordinarily slow.

For command line usage manual and library documentation,
please visit `readthedocs <http://dgenerate.readthedocs.io/en/@REVISION/>`_.

===

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui.gif
   :alt: console ui