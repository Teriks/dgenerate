Upscaling Images
================

dgenerate implements four different methods of upscaling images, animated images, or video.

Upscaling with the Stable Diffusion based x2 and x4 upscalers from the `diffusers <https://github.com/huggingface/diffusers>`_ library.

With the ``upscale`` image processor, which is compatible with torch models implemented in the `spandrel <https://github.com/chaiNNer-org/spandrel>`_ library.

And with the ``upscaler-ncnn`` image processor, which implements upscaling with generic NCNN upscaling models using the `ncnn <https://github.com/Tencent/ncnn>`_ library.

The `spandrel <https://github.com/chaiNNer-org/spandrel>`_ library supports the use of most torch models on: https://openmodeldb.info/

The `ncnn <https://github.com/Tencent/ncnn>`_ library supports models compatible with `upscayl <https://github.com/upscayl/upscayl>`_ as well as `chaiNNer <https://github.com/chaiNNer-org/chaiNNer>`_.

ONNX upscaler models can be converted to NCNN format for use with the ``upscaler-ncnn`` image processor.


Upscaling with Diffusion Upscaler Models
----------------------------------------

Stable diffusion image upscaling models can be used via the model types:

    * ``--model-type torch-upscaler-x2``
    * ``--model-type torch-upscaler-x4``

The image used in the example below is this `low resolution cat <https://raw.githubusercontent.com/Teriks/dgenerate/@REVISION/examples/media/low_res_cat.png>`_

.. code-block:: bash

    #!/usr/bin/env bash

    # The image produced with this model will be
    # two times the --output-size dimension IE: 512x512 in this case
    # The image is being resized to 256x256, and then upscaled by 2x

    dgenerate stabilityai/sd-x2-latent-upscaler --variant fp16 --dtype float16 \
    --model-type torch-upscaler-x2 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256


    # The image produced with this model will be
    # four times the --output-size dimension IE: 1024x1024 in this case
    # The image is being resized to 256x256, and then upscaled by 4x

    dgenerate stabilityai/stable-diffusion-x4-upscaler --variant fp16 --dtype float16 \
     --model-type torch-upscaler-x4 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256 \
    --upscaler-noise-levels 20


Upscaling with chaiNNer Compatible Torch Upscaler Models
--------------------------------------------------------

`chaiNNer <https://github.com/chaiNNer-org/chaiNNer>`_ compatible torch upscaler models from https://openmodeldb.info/
and elsewhere can be utilized for tiled upscaling using dgenerate's ``upscaler`` image processor and the
``--post-processors`` option.  The ``upscaler`` image processor can also be used for processing
input images via the other options mentioned in `Image Processors`_ such as ``--seed-image-processors``

The ``upscaler`` image processor can make use of URLs or files on disk.

In this example we reference a link to the SwinIR x4 upscaler from the creators github release.

This uses the upscaler to upscale the output image by x4 producing an image that is 4096x4096

The ``upscaler`` image processor respects the ``--device`` option of dgenerate, and is CUDA accelerated by default.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork" \
    --post-processors "upscaler;model=https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"


In addition to this the ``\image_process`` config directive, or ``--sub-command image-process`` can be used to upscale
any file that you want including animated images and videos. It is worth noting that the sub-command and directive
will work with any named image processor implemented by dgenerate.


.. code-block:: bash

    #!/usr/bin/env bash

    # print the help output of the sub command "image-process"
    # the image-process sub-command can process multiple files and do
    # and several other things, it is worth reading :)

    dgenerate --sub-command image-process --help

    # any directory mentioned in the output spec is created automatically

    dgenerate --sub-command image-process my-file.png \
    --output output/my-file-upscaled.png \
    --processors "upscaler;model=https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"


For more information see: ``dgenerate --image-processor-help upscaler``

Control over tiling parameters and specifics are discussed in the image processor
help documentation from the above command.


Upscaling with NCNN Upscaler Models
-----------------------------------

The ``upscaler-ncnn`` image processor will be available if you have manually installed dgenerate
with the ``[ncnn]`` extra, or if you are using dgenerate from the packaged windows installer or portable
windows install zip from the releases page.

NCNN can use Vulkan for hardware accelerated inference and is also heavily optimized
for CPU use if needed.

When using the ``upscaler-ncnn`` processor, you must specify both the ``model`` and ``param`` arguments,
these refer to the ``model.bin`` and ``model.param`` file associated with the model.

These arguments may be a path to a file on disk or a hard link to a downloadable model in raw form.

This upscaler utilizes the same tiling algorithm as the ``upscaler`` image processor
and features the same ``tile`` and ``overlap`` arguments, albeit with slightly different
defaults and constraints.  The ``tile`` argument may not exceed 400 pixels and defaults
to the max value of 400. Tiling can be disabled for input images under 400 pixels by
passing ``tile=0``.

By default the ``upscaler-ncnn`` processor does not run on the GPU, you must
enable this with the ``use-gpu`` argument.

When using this processor as a pre-processor or post-processor for diffusion,
GPU memory will be fenced, any cached models related to diffusion on the GPU
will be evacuated entirely before this processor runs if they exist on the same GPU
as the processor, this is to prevent catastrophic interaction between the Vulkan
and Torch cuda allocators.

Once a Vulkan allocator exists on a specific GPU it cannot be destroyed except
via the process exiting due to issues with the ncnn python binding.  If you
create this processor on a GPU you intend to perform diffusion on, you are
going to run into memory errors after the first image generation and
there on out until the process exits.

When the process exits it is very likely to exit with a non-zero return
code after using this processor even if the upscale operations were successful,
this is due to problems with the ncnn python binding creating a segfault at exit.
If you are using dgenerate interactively in shell mode or from the Console UI,
this will occur without consequence when the interpreter process exits.

Note that if any other process runs diffusion / inference via torch on
the same GPU as this image processor while ncnn is preforming inference,
you will likely encounter a segfault in either of the processes and
a very hard crash.

You can safely run this processor in parallel with diffusion, or other torch
based image processors with GPU acceleration, by placing it on a separate gpu
using the ``gpu-index`` argument.

Since the ncnn upscaler can run on GPUs other than Nvidia GPUs, figuring out what index
you need to use is platform specific, but for Nvidia users just use the ``nvidia-smi`` command
from a terminal to get this value.

If you do not specify a ``gpu-index``, index 0 is used, which is most likely your main GPU.

The ``--device`` argument to dgenerate and the ``image-process`` sub-command / ``\image_process`` directive
is ignored by this image processor.

 .. code-block:: bash

     #! /usr/bin/env bash

     # this auto downloads x2 upscaler models from the upscayl repository into
     # dgenerate's web cache, and then use them

     MODEL=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.bin
     PARAM=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.param

     dgenerate --sub-command image-process my-file.png \
     --output output/my-file-upscaled.png \
     --processors "upscaler-ncnn;model=${MODEL};param=${PARAM};use-gpu=true"

If you are upscaling using the CPU, you can specify a thread count using the ``threads`` argument.

This argument can be an integer quantity of threads, the keyword ``auto``
(max logical processors, max threads) or the keyword ``half`` (half your logical processors).

 .. code-block:: bash

     #! /usr/bin/env bash

     # this auto downloads x2 upscaler models from the upscayl repository into
     # dgenerate's web cache, and then use them

     MODEL=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.bin
     PARAM=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.param

     dgenerate --sub-command image-process my-file.png \
     --output output/my-file-upscaled.png \
     --processors "upscaler-ncnn;model=${MODEL};param=${PARAM};threads=half"


The argument ``winograd=true`` can be used to enable the winograd convolution when running on CPU,
similarly the ``sgemm=true`` argument can be used to enable the sgemm convolution optimization.

In addition, you can control OpenMP blocktime using the ``blocktime`` argument, which should be
an integer value between 0 and 400 inclusive, representing milliseconds.

These arguments can only be used when running on the CPU and will throw an argument error otherwise.

When they are not specified, optimal defaults from ncnn for your platform are used.


For more information see: ``dgenerate --image-processor-help upscaler-ncnn``


