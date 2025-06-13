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
REPL `Console UI`_ for the dgenerate configuration / scripting language, which is built on
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

For library documentation, and a better README reading experience which
includes proper syntax highlighting for examples, and side panel navigation,
please visit `readthedocs <http://dgenerate.readthedocs.io/en/@REVISION/>`_.

----

* `Help Output`_
* `Diffusion Feature Table <https://github.com/Teriks/dgenerate/blob/@REVISION/FEATURE_TABLE.rst>`_

* How To Install
@TOC[{"from_index": "Windows Install", "to_index": "Basic Usage", "indent": 4, "max_depth": 1}]

* Usage Manual
@TOC[{"from_index": "Basic Usage", "indent": 4}]

Help Output
===========

@COMMAND_OUTPUT[dgenerate --no-stdin --help]

@INCLUDE[sections/install/windows_install.template.rst]

@INCLUDE[sections/install/linux_install.template.rst]

@INCLUDE[sections/install/opencv_headless.template.rst]

@INCLUDE[sections/install/mac_install.template.rst]

@INCLUDE[sections/install/google_colab_install.template.rst]

@INCLUDE[sections/install/installing_development_branches.template.rst]

@INCLUDE[sections/basic_usage.template.rst]

@INCLUDE[sections/basic_prompting.template.rst]

@INCLUDE[sections/image_inputs/image_seeds.template.rst]

@INCLUDE[sections/image_inputs/inpainting.template.rst]

@INCLUDE[sections/image_inputs/per_image_seed_resizing.template.rst]

@INCLUDE[sections/image_inputs/animated_output.template.rst]

@INCLUDE[sections/image_inputs/animation_slicing.template.rst]

@INCLUDE[sections/image_inputs/inpainting_animations.template.rst]

@INCLUDE[sections/image_inputs/latents_interchange.template.rst]

@INCLUDE[sections/deterministic_output.template.rst]

@INCLUDE[sections/specifying_gpu.template.rst]

@INCLUDE[sections/specifying_scheduler.template.rst]

@INCLUDE[sections/specifying_sigmas.template.rst]

@INCLUDE[sections/sub_models/specifying_vae.template.rst]

@INCLUDE[sections/sub_models/vae_tiling_slicing.template.rst]

@INCLUDE[sections/sub_models/specifying_unet.template.rst]

@INCLUDE[sections/sub_models/specifying_transformer.template.rst]

@INCLUDE[sections/sub_models/specifying_sdxl_refiner.template.rst]

@INCLUDE[sections/sub_models/specifying_stable_cascade_decoder.template.rst]

@INCLUDE[sections/sub_models/specifying_loras.template.rst]

@INCLUDE[sections/sub_models/specifying_textual_inversions.template.rst]

@INCLUDE[sections/sub_models/specifying_controlnets.template.rst]

@INCLUDE[sections/sub_models/sdxl_controlnet_union_mode.template.rst]

@INCLUDE[sections/sub_models/flux_controlnet_union_mode.template.rst]

@INCLUDE[sections/sub_models/specifying_t2i_adapters.template.rst]

@INCLUDE[sections/sub_models/specifying_ip_adapters.template.rst]

@INCLUDE[sections/sub_models/specifying_text_encoders.template.rst]

@INCLUDE[sections/prompt_upscaling.template.rst]

@INCLUDE[sections/prompt_weighting.template.rst]

@INCLUDE[sections/embedded_prompt_arguments.template.rst]

@INCLUDE[sections/utilizing_civitai_links_and_hosted_models.template.rst]

@INCLUDE[sections/specifying_generation_batch_size.template.rst]

@INCLUDE[sections/image_inputs/batching_input_images_and_inpaint_masks.template.rst]

@INCLUDE[sections/image_processing/image_processors.template.rst]

@INCLUDE[sections/image_processing/latents_processors.template.rst]

@INCLUDE[sections/sub_commands.template.rst]

@INCLUDE[sections/image_processing/upscaling_images.template.rst]

@INCLUDE[sections/image_processing/adetailer_yolo_based_inpainting.template.rst]

@INCLUDE[sections/writing_and_running_configs.template.rst]

@INCLUDE[sections/console_ui.template.rst]

@INCLUDE[sections/writing_plugins.template.rst]

@INCLUDE[sections/environment/auth_tokens_environment.template.rst]

@INCLUDE[sections/environment/file_cache_control.template.rst]