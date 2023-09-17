Overview
========

**dgenerate** is a command line tool for generating images and animation sequences using stable diffusion.

**dgenerate** can generate multiple images or animated outputs using multiple combinations of input parameters
for stable diffusion in batch, so that the differences in generated output can be compared / curated easily.

Animated output can be produced by processing every frame of a video, gif, webp through stable diffusion as
an image seed with a given prompt and generation parameters.

Video of infinite runtime can be processed without memory constraints.

GIF's and WebP can also be processed, with memory constraints.

This software requires an Nvidia GPU supporting CUDA 11.8+, CPU rendering is possible but extraordinarily slow.

----

.. _Windows Install: /#windows-install
.. _Linux or WSL Install: /#linux-or-wsl-install
.. _Usage Examples: /#usage-examples

For Windows install instructions jump to: `Windows Install`_

For Linux or WSL (Windows subsystem for Linux) install instructions jump to: `Linux or WSL Install`_

For usage examples jump to: `Usage Examples`_

Help
----

.. code-block::

    usage: dgenerate [-h] [--model-type MODEL_TYPE] [--revision REVISION] [--variant VARIANT]
                     [--subfolder SUBFOLDER] [--auth-token AUTH_TOKEN] [--vae VAE] [--lora LORA]
                     [--textual-inversions TEXTUAL_INVERSIONS [TEXTUAL_INVERSIONS ...]] [--scheduler SCHEDULER]
                     [--sdxl-refiner SDXL_REFINER] [--sdxl-original-size SDXL_ORIGINAL_SIZE]
                     [--sdxl-target-size SDXL_TARGET_SIZE] [--safety-checker] [--version] [-d DEVICE] [-t DTYPE]
                     [-s OUTPUT_SIZE] [-o OUTPUT_PATH] [-op OUTPUT_PREFIX] [-p PROMPTS [PROMPTS ...]]
                     [-se SEEDS [SEEDS ...] | -gse GEN_SEEDS] [-af ANIMATION_FORMAT] [-fs FRAME_START]
                     [-fe FRAME_END] [-is [IMAGE_SEEDS ...]] [-iss [IMAGE_SEED_STRENGTHS ...]]
                     [-gs [GUIDANCE_SCALES ...]] [-ifs [INFERENCE_STEPS ...]]
                     [-hnf [SDXL_HIGH_NOISE_FRACTIONS ...]]
                     model_path

    Stable diffusion batch image generation tool with support for video / gif / webp animation transcoding.

    positional arguments:
      model_path            huggingface model repository slug, huggingface blob link to a model file, path to
                            folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file.

    options:
      -h, --help            show this help message and exit
      --model-type MODEL_TYPE
                            Use when loading different model types. Currently supported: torch or torch-sdxl.
                            (default: torch)
      --revision REVISION   The model revision to use when loading from a huggingface repository, (The git branch /
                            tag, default is "main")
      --variant VARIANT     If specified when loading from a huggingface repository or folder, load weights from
                            "variant" filename, e.g. "pytorch_model.<variant>.safetensors". Defaults to automatic
                            selection. This option is ignored if using flax.
      --subfolder SUBFOLDER
                            Main model subfolder. If specified when loading from a huggingface repository or
                            folder, load weights from the specified subfolder.
      --auth-token AUTH_TOKEN
                            Huggingface auth token. Required to download restricted repositories that have access
                            permissions granted to your huggingface account.
      --vae VAE             Specify a VAE. When using torch models the syntax is:
                            "AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)".
                            Examples: "AutoencoderKL;model=vae.pt",
                            "AsymmetricAutoencoderKL;model=huggingface/vae",
                            "AutoencoderTiny;model=huggingface/vae". When using a Flax model, there is currently
                            only one available encoder class: "FlaxAutoencoderKL;model=huggingface/vae". The
                            AutoencoderKL model argument accepts huggingface repository slugs, .pt, .pth, .bin,
                            .ckpt, and .safetensors files. Other encoders can only accept huggingface repository
                            slugs/blob links or a path to a folder on disk with the model configuration and model
                            file(s). Aside from the "model" argument, there are four other optional arguments that
                            can be specified, these include "revision", "variant", "subfolder", "dtype". They can
                            be specified as so in any order, they are not positional: "AutoencoderKL;model=huggingf
                            ace/vae;revision=main;variant=fp16;subfolder=sub_folder;dtype=float16". The "revision"
                            argument specifies the model revision to use for the VAE when loading from huggingface
                            repository or blob link, (The git branch / tag, default is "main"). The "variant"
                            argument specifies the VAE model variant and defaults to the value of --variant, when
                            "variant" is specified when loading from a huggingface repository or folder, weights
                            will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors.
                            "variant" defaults to automatic selection and is ignored if using flax. The "subfolder"
                            argument specifies the VAE model subfolder, if specified when loading from a
                            huggingface repository or folder, weights from the specified subfolder. The "dtype"
                            argument specifies the VAE model precision, it defaults to the value of -t/--dtype and
                            should be one of: float16 / float32 / auto. If you wish to load a weights file directly
                            from disk, the simplest way is: --vae "AutoencoderKL;my_vae.safetensors", or with a
                            dtype "AutoencoderKL;my_vae.safetensors;dtype=float16", all other loading arguments are
                            unused in this case and may produce an error message if used. If you wish to load a
                            specific weight file from a hugging face repository, use the blob link loading syntax:
                            --vae "AutoencoderKL;https://huggingface.co/UserName/repository-
                            name/blob/main/vae_model.safetensors", the revision argument may be used with this
                            syntax.
      --lora LORA, --loras LORA
                            Specify a LoRA model (flax not supported). This should be a huggingface repository slug
                            / blob link, path to model file on disk (for example, a .pt, .pth, .bin, .ckpt, or
                            .safetensors file), or model folder containing model files. Optional arguments can be
                            provided after the LoRA model specification, these include: "scale", "revision",
                            "subfolder", and "weight-name". They can be specified as so in any order, they are not
                            positional: "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-
                            name=lora.safetensors". The "scale" argument indicates the scale factor of the LoRA.
                            The "revision" argument specifies the model revision to use for the VAE when loading
                            from huggingface repository, (The git branch / tag, default is "main"). The "subfolder"
                            argument specifies the VAE model subfolder, if specified when loading from a
                            huggingface repository or folder, weights from the specified subfolder. The "weight-
                            name" argument indicates the name of the weights file to be loaded when loading from a
                            huggingface repository or folder on disk. If you wish to load a weights file directly
                            from disk, the simplest way is: --lora "my_lora.safetensors", or with a scale
                            "my_lora.safetensors;scale=1.0", all other loading arguments are unused in this case
                            and may produce an error message if used.
      --textual-inversions TEXTUAL_INVERSIONS [TEXTUAL_INVERSIONS ...]
                            Specify one or more Textual Inversion models (flax and SDXL not supported). This should
                            be a huggingface repository slug / blob link, path to model file on disk (for example,
                            a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder containing model
                            files. Optional arguments can be provided after the Textual Inversion model
                            specification, these include: "revision", "subfolder", and "weight-name". They can be
                            specified as so in any order, they are not positional:
                            "huggingface/ti_model;revision=main;subfolder=repo_subfolder;weight-
                            name=lora.safetensors". The "revision" argument specifies the model revision to use for
                            the Textual Inversion model when loading from huggingface repository, (The git branch /
                            tag, default is "main"). The "subfolder" argument specifies the Textual Inversion model
                            subfolder, if specified when loading from a huggingface repository or folder, weights
                            from the specified subfolder. The "weight-name" argument indicates the name of the
                            weights file to be loaded when loading from a huggingface repository or folder on disk.
                            If you wish to load a weights file directly from disk, the simplest way is: --textual-
                            inversions "my_ti_model.safetensors", all other loading arguments are unused in this
                            case and may produce an error message if used.
      --scheduler SCHEDULER
                            Specify a Scheduler by name. Torch compatible schedulers: (DDIMScheduler,
                            DDPMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                            HeunDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
                            DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
                            DEISMultistepScheduler, UniPCMultistepScheduler, DPMSolverSDEScheduler).
      --sdxl-refiner SDXL_REFINER
                            Stable Diffusion XL (torch-sdxl) refiner model path. This should be a huggingface
                            repository slug / blob link, path to model file on disk (for example, a .pt, .pth,
                            .bin, .ckpt, or .safetensors file), or model folder containing model files. Optional
                            arguments can be provided after the SDXL refiner model specification, these include:
                            "revision", "variant", "subfolder", and "dtype". They can be specified as so in any
                            order, they are not positional: "huggingface/refiner_model_xl;revision=main;variant=fp1
                            6;subfolder=repo_subfolder;dtype=float16". The "revision" argument specifies the model
                            revision to use for the Textual Inversion model when loading from huggingface
                            repository, (The git branch / tag, default is "main"). The "variant" argument specifies
                            the SDXL refiner model variant and defaults to the value of --variant, when "variant"
                            is specified when loading from a huggingface repository or folder, weights will be
                            loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant"
                            defaults to automatic selection. The "subfolder" argument specifies the SDXL refiner
                            model subfolder, if specified when loading from a huggingface repository or folder,
                            weights from the specified subfolder. If you wish to load a weights file directly from
                            disk, the simplest way is: --sdxl-refiner "my_sdxl_refiner.safetensors", all other
                            loading arguments are unused in this case and may produce an error message if used. If
                            you wish to load a specific weight file from a hugging face repository, use the blob
                            link loading syntax: --sdxl-refiner "https://huggingface.co/UserName/repository-
                            name/blob/main/refiner_model.safetensors", the revision argument may be used with this
                            syntax.
      --sdxl-original-size SDXL_ORIGINAL_SIZE
                            Stable Diffusion XL (torch-sdxl) micro-conditioning parameter in the format
                            (WIDTHxHEIGHT). If not the same as --sdxl-target-size the image will appear to be down
                            or upsampled. --sdxl-original-size defaults to --output-size if not specified. Part of
                            SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]
      --sdxl-target-size SDXL_TARGET_SIZE
                            Stable Diffusion XL (torch-sdxl) micro-conditioning parameter in the format
                            (WIDTHxHEIGHT). For most cases, --sdxl-target-size should be set to the desired height
                            and width of the generated image. If not specified it will default to --output-size.
                            Part of SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]
      --safety-checker      Enable safety checker loading, this is off by default. When turned on images with NSFW
                            content detected may result in solid black output. Some pretrained models have settings
                            indicating a safety checker is not to be loaded, in that case this option has no
                            effect.
      --version             show program's version number and exit
      -d DEVICE, --device DEVICE
                            cuda / cpu. (default: cuda). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific
                            GPU.
      -t DTYPE, --dtype DTYPE
                            Model precision: float16 / float32 / auto. (default: auto)
      -s OUTPUT_SIZE, --output-size OUTPUT_SIZE
                            Image output size. If an image seed is used it will be resized to this dimension with
                            aspect ratio maintained, width will be fixed and a new height will be calculated. If
                            only one integer value is provided, that is the value for both dimensions. X/Y
                            dimension values should be separated by "x". (default: 512x512 when no image seeds are
                            specified)
      -o OUTPUT_PATH, --output-path OUTPUT_PATH
                            Output path for generated images and files. This directory will be created if it does
                            not exist. (default: ./output)
      -op OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                            Name prefix for generated images and files. This prefix will be added to the beginning
                            of every generated file, followed by an underscore.
      -p PROMPTS [PROMPTS ...], --prompts PROMPTS [PROMPTS ...]
                            List of prompts to try, an image group is generated for each prompt, prompt data is
                            split by ; (semi-colon). The first value is the positive text influence, things you
                            want to see. The Second value is negative influence IE. things you don't want to see.
                            Example: --prompts "shrek flying a tesla over detroit; clouds, rain, missiles".
                            (default: [(empty string)])
      -se SEEDS [SEEDS ...], --seeds SEEDS [SEEDS ...]
                            List of seeds to try, define fixed seeds to achieve deterministic output. This argument
                            may not be used when --gse/--gen-seeds is used. (default: [randint(0, 99999999999999)])
      -gse GEN_SEEDS, --gen-seeds GEN_SEEDS
                            Auto generate N random seeds to try. This argument may not be used when -se/--seeds is
                            used.
      -af ANIMATION_FORMAT, --animation-format ANIMATION_FORMAT
                            Output format when generating an animation from an input video / gif / webp etc. Value
                            must be one of: mp4, gif, or webp. (default: mp4)
      -fs FRAME_START, --frame-start FRAME_START
                            Starting frame slice point for animated files, the specified frame will be included.
      -fe FRAME_END, --frame-end FRAME_END
                            Ending frame slice point for animated files, the specified frame will be included.
      -is [IMAGE_SEEDS ...], --image-seeds [IMAGE_SEEDS ...]
                            List of image seeds to try when processing image seeds, these may be URLs or file
                            paths. Videos / GIFs / WEBP files will result in frames being rendered as well as an
                            animated output file being generated if more than one frame is available in the input
                            file. Inpainting for static images can be achieved by specifying a black and white mask
                            image in each image seed string using a semicolon as the separating character, like so:
                            "my-seed-image.png;my-image-mask.png", white areas of the mask indicate where generated
                            content is to be placed in your seed image. Output dimensions specific to the image
                            seed can be specified by placing the dimension at the end of the string following a
                            semicolon like so: "my-seed-image.png;512x512" or "my-seed-image.png;my-image-
                            mask.png;512x512". Inpainting masks can be downloaded for you from a URL or be a path
                            to a file on disk.
      -iss [IMAGE_SEED_STRENGTHS ...], --image-seed-strengths [IMAGE_SEED_STRENGTHS ...]
                            List of image seed strengths to try. Closer to 0 means high usage of the seed image
                            (less noise convolution), 1 effectively means no usage (high noise convolution). Low
                            values will produce something closer or more relevant to the input image, high values
                            will give the AI more creative freedom. (default: [0.8])
      -gs [GUIDANCE_SCALES ...], --guidance-scales [GUIDANCE_SCALES ...]
                            List of guidance scales to try. Guidance scale effects how much your text prompt is
                            considered. Low values draw more data from images unrelated to text prompt. (default:
                            [5])
      -ifs [INFERENCE_STEPS ...], --inference-steps [INFERENCE_STEPS ...]
                            Lists of inference steps values to try. The amount of inference (de-noising) steps
                            effects image clarity to a degree, higher values bring the image closer to what the AI
                            is targeting for the content of the image. Values between 30-40 produce good results,
                            higher values may improve image quality and or change image content. (default: [30])
      -hnf [SDXL_HIGH_NOISE_FRACTIONS ...], --sdxl-high-noise-fractions [SDXL_HIGH_NOISE_FRACTIONS ...]
                            High noise fraction for Stable Diffusion XL (torch-sdxl), this fraction of inference
                            steps will be processed by the base model, while the rest will be processed by the
                            refiner model. Multiple values to this argument will result in additional generation
                            steps for each value.


Windows Install
===============

You can install using the Windows installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_, or you can manually
install with pipx, (or pip if you want) as described below.


Manual Install
--------------


Install Visual Studios (Community or other), make sure "Desktop development with C++" is selected, unselect anything you do not need.

https://visualstudio.microsoft.com/downloads/


Install rust compiler using rustup-init.exe (x64), use the default install options.

https://www.rust-lang.org/tools/install

Install Python:

https://www.python.org/ftp/python/3.11.3/python-3.11.3-amd64.exe

Make sure you select the option "Add to PATH" in the python installer,
otherwise invoke python directly using it's full path while installing the tool.

Install GIT for Windows:

https://gitforwindows.org/


Install dgenerate
-----------------

Using Windows CMD

Install pipx:

.. code-block:: bash

    pip install pipx
    pipx ensurepath

    # Log out and log back in so PATH takes effect

Install dgenerate:

.. code-block:: bash

    pipx install git+https://github.com/Teriks/dgenerate.git ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # If you want a specific version

    pipx install git+https://github.com/Teriks/dgenerate.git@v1.0.0 ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"


Run **dgenerate** to generate images:

.. code-block:: bash

    dgenerate --help

    dgenerate CompVis/stable-diffusion-v1-4 ^
    --prompts "an astronaut riding a horse" ^
    --output-path output ^
    --inference-steps 40 ^
    --guidance-scales 10

Linux or WSL Install
====================

First update your system and install build-essential

.. code-block:: bash

    sudo apt update && sudo apt upgrade
    sudo apt install build-essential


Install CUDA Toolkit 12.*: https://developer.nvidia.com/cuda-downloads

I recommend using the runfile option:

.. code-block:: bash

    # CUDA Toolkit 12.2.1 For Ubuntu / Debian / WSL

    wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
    sudo sh cuda_12.2.1_535.86.10_linux.run

Do not attempt to install a driver from the prompts if using WSL.

Add libraries to linker path:

.. code-block:: bash

    # Add to ~/.bashrc

    # For Linux add the following
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # For WSL add the following
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # Add this in both cases as well
    export PATH=/usr/local/cuda/bin:$PATH


When done editing ``~/.bashrc`` do:

.. code-block:: bash

    source ~/.bashrc


Install Python 3.10+ (Debian / Ubuntu) and pipx
-----------------------------------------------

.. code-block:: bash

    sudo apt install python3.10 python3-pip pipx python3.10-venv python3-wheel
    pipx ensurepath

    source ~/.bashrc


Install dgenerate
-----------------

.. code-block:: bash

    pipx install git+https://github.com/Teriks/dgenerate.git \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # With flax/jax support

    pipx install "dgenerate[flax] @ git+https://github.com/Teriks/dgenerate.git" \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

    # If you want a specific version

    pipx install git+https://github.com/Teriks/dgenerate.git@v1.0.0 \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # Specific version with flax/jax support

    pipx install "dgenerate[flax] @ git+https://github.com/Teriks/dgenerate.git@v1.0.0" \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"


Run **dgenerate** to generate images:

.. code-block:: bash

    dgenerate --help

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10

Usage Examples
==============

Generate an astronaut riding a horse using 5 different random seeds, 3 different inference-steps values, 3 different guidance-scale values.

Adjust output size to 512x512 and output generated images to 'astronaut' folder.

45 uniquely named images will be generated (5x3x3)

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


Loading models from huggingface blob links is also supported

.. code-block:: bash

    dgenerate https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/diffusion_pytorch_model.safetensors \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


SDXL is supported and can be used to generate highly realistic images.

Prompt only generation, img2img, and inpainting is supported for SDXL.

Refiner models can be specified, fp16 model variant and a datatype of float16 is
recommended to prevent out of memory conditions on the average GPU :)

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --sdxl-high-noise-fractions 0.6 0.7 0.8 \
    --gen-seeds 5 \
    --inference-steps 50 \
    --guidance-scale 12 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "real photo of an astronaut riding a horse on the moon" \
    --variant fp16 --dtype float16 \
    --output-size 1024
    
    
Negative Prompt
---------------

In order to specify a negative prompt, each prompt argument is split
into two parts separated by ``;``

The prompt text occuring after ``;`` is the negative influence prompt.

To attempt to avoid rendering of a saddle on the horse being ridden, you
could for example add the negative prompt "saddle" or "wearing a saddle"
or "horse wearing a saddle" etc.


.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse; horse wearing a saddle" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512
    
    
Multiple Prompts
----------------
 
Multiple prompts can be specified one after another in quotes in order
to generate images using multiple prompt variations.
 
The following command generates 10 uniquely named images using two 
prompts and five random seeds (2x5)
 
5 of them will be from the first prompt and 5 of them from the second prompt.
 
All using 50 inference steps, and 10 for guidance scale value.
 
 
.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" "an astronaut riding a donkey" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Image Seed
----------

Use a photo of Buzz Aldrin on the moon to generate a photo of an astronaut standing on mars, this uses an image seed downloaded from wikipedia.

Disk file paths may also be used for image seeds, multiple image seeds may be provided, images will be generated from each image seed individually.

Generate this image using 5 different seeds, 3 different inference-step values, 3 different guidance-scale values as above.

In addition this image will be generated using 3 different image seed strengths.

Adjust output size to 512x512 and output generated images to 'astronaut' folder, if the image seed
is not a 1:1 aspect ratio the width will be fixed to the requested width and the height of the output image
calculated to maintain aspect ratio.

If you do not adjust the output size of the generated image, the size of the input image seed will be used.

135 uniquely named images will be generated (5x3x3x3)

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut walking on mars" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg \
    --image-seed-strengths 0.2 0.5 0.8 \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


Inpainting
----------

Inpainting on an image can be preformed by providing a mask image with your image seed. This mask should be a black and white image
of identical size to your image seed.  White areas of the mask image will be used to tell the AI what areas of the seed image should be filled
in with generated content.

.. _Inpainting Animations: /#inpainting-animations

For using inpainting on animated image seeds, jump to: `Inpainting Animations`_

In order to use inpainting, specify your image seed like so: ``--image-seeds "my-image-seed.png;my-mask-image.png"``

The format is your image seed and mask image seperated by ``;``

Mask images can be downloaded from URL's just like image seeds, however for this example the syntax specifies a file on disk for brevity.

**my-image-seed.png**: https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png

**my-mask-image.png**: https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png

The command below generates a cat sitting on a bench with the images from the links above, the mask image masks out
areas over the dog in the original image, causing the dog to be replaced with an AI generated cat.

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --image-seeds "my-image-seed.png;my-mask-image.png" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scale 10 \
    --inference-steps 100


Per Image Seed Resizing
-----------------------

If you want to specify multiple image seeds that will have different output sizes irrespective
of their input size or a globally defined output size defined with ``--output-size``,
You can specify their output size individually at the end of each provided image seed.

This will work when using a mask image for inpainting as well, including when using animated inputs.

The syntax is: ``--image-seeds "my-image-seed.png;512x512"`` or ``--image-seeds "my-image-seed.png;my-mask-image.png;512x512"``

When one dimension is specified, that dimension is the width, and the height is calculated from the aspect ratio of the input image.

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --image-seeds "my-image-seed.png;1024" "my-image-seed.png;my-mask-image.png;512x512" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scale 10 \
    --inference-steps 100


Animated Output
---------------

**dgenerate** supports many video formats through the use of PyAV, as well as GIF & WebP.

When an animated image seed is given, animated output will be produced in the format of your choosing.

In addition, every frame will be written to the output folder as a uniquely named image.

Use a GIF of a man riding a horse to create an animation of an astronaut riding a horse.

Output to an MP4.  See ``--help`` for information about formats supported by ``--animation-format``

If the animation is not 1:1 aspect ratio, the width will be fixed to the width of the
requested output size, and the height calculated to match the aspect ratio of the animation.

If you do not set an output size, the size of the input animation will be used.

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4


Animation Slicing
-----------------

Animated inputs can be sliced by a frame range, currently this only works globally so
if you provide multiple animated inputs they will all be sliced in an identical manner 
using the provided slice setting. Individual slice settings per image seed will probably 
be added in the future.

Perhaps you only want to run diffusion on the first frame of an animated input in
order to save time in finding good parameters for generating every frame. You could
do something like this in order to test different parameters on only the first frame,
which will be much faster than rendering the entire video/gif outright.

The slice range is inclusive, meaning that the frames pecified by ``--frame-start`` and ``--frame-end``
will be included in the slice.  Both slice points do not have to be specified at the same time, IE, you can slice
the tail end of a video out, or seek to a certain frame in the video and start from there if you wanted, by only
specifying a start, or an end parameter instead of both simultaneously.

If your slice only results in the processing of a single frame, it will be treated as a normal image seed and only
image output will be produced instead of an animation.


.. code-block:: bash
    
    # Generate using only the first frame
    
    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4 \
    --frame-start 0 \
    --frame-end 0


Inpainting Animations
---------------------

Image seeds can be supplied an animated or static image mask to define the areas for inpainting while generating an animated output.

All combinations of animated seed and animated / or static mask can be handled.

When an animated seed is used with an animated mask, the mask for every corresponding frame in the input is taken from the animated mask,
the runtime of the animated output will be equal to the shorter of the two animated inputs. IE: If the seed animation and the mask animation
have different length, the animated output is clipped to the length of the shorter of the two.

When a static image is used as a mask, that image is used as an inpaint mask for every frame of the animated seed.

When an animated mask is used with a static image seed, the animated output length is that of the animated mask. A video is
created by duplicating the image seed for every frame of the animated mask, the animated output being generated by masking
them together.


.. code-block:: bash

    # A video with a static inpaint mask over the entire video

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-static-mask.png" \
    --output-path inpaint \
    --animation-format mp4

    # Zip two videos together, masking the left video with corrisponding frames
    # from the right video. The two animated inputs do not have to be the same file format
    # you can mask videos with gif/webp and vice versa

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.webp" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.webp;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4 \

    # etc...

    # Use a static image seed and mask it with every frame from an
    # Animated mask file

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4 \

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.webp" \
    --output-path inpaint \
    --animation-format mp4 \

    # etc...

    

Manual Seed Specification / Deterministic Output
------------------------------------------------

If you generate an image you like using a random seed, you can later reuse that seed in another generation.

Output images have the name format: ``s_(seed)_st_(image-seed-strength)_g_(guidance-scale)_i_(inference-steps)_step_(generation-step).png``,
the first number being the random seed used for generation of that particular image.

When using SDXL there will be an extra component to the file name after inference steps, that being ``hnf``, or high noise fraction.

Reusing a seed has the effect of perfectly reproducing the image in the case that all other parameters are left alone, 
including prompt, output size, and model version.

Updates to the backing model may affect determinism in the generation.

Specifying a seed directly and changing the prompt slightly, or parameters such as image seed strength if using a seed image,
guidance scale, or inference steps, will allow for generating variations close to the original
image which may possess all of the original qualities about the image that you liked as well as
additional qualities.  You can further manipulate the AI into producing results that you want with this method.

Changing output resolution will drastically affect image content when reusing a seed to the point where trying to
reuse a seed with a different output size is pointless.

The following command demonstrates manually specifying two different seeds to try: **1234567890**, and **9876543210**

.. code-block:: bash

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --seeds 1234567890 9876543210 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Specifying a VAE
----------------

To specify a VAE directly use ``--vae``.

The syntax for ``--vae`` is ``AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)``

Named arguments when loading a VAE are seperated by the ``;`` character and are
not positional, meaning they can be defined in any order.

The only named argument compatible with loading a .safetensors file directly off disk is ``model`` and ``dtype``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes for torch models are:

* AutoencoderKL
* AsymmetricAutoencoderKL
* AutoencoderTiny

Available encoder classes for flax models are:

* FlaxAutoencoderKL


The AutoencoderKL encoder class accepts huggingface repository slugs/blob links,
.pt, .pth, .bin, .ckpt, and .safetensors files. Other encoders can only accept huggingface
repository slugs/blob links, or a path to a folder on disk with the model
configuration and model file(s).


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/sd-vae-ft-mse" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you wish to specify a weights variant IE: load "pytorch_model.<variant>.safetensors", from a huggingface
repository that has variants of the same model, use the named argument ``variant``.  This usage is only
valid when loading VAE's if ``--model-type`` is either ``torch`` or ``torch-sdxl``.  Attempting
to use it with FlaxAutoencoderKL with produce an error message. By default this value is the same as
``--variant`` when that option is specified for the main model.


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --variant fp16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae;variant=fp16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to specify the model precision, use the named argument ``dtype``,
accepted values are the same as ``--dtype``, IE: 'float32', 'float16', 'auto'

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae;dtype=float16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512

If you are loading a .safetensors or other file from a path on disk, only the ``model`` and ``dtype`` arguments are available.

.. code-block:: bash
    # These are only syntax examples

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors" \
    --prompts "Syntax example"

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors;dtype=float16" \
    --prompts "Syntax example"


Specifying a LoRA finetune
--------------------------

To specify a LoRA finetune model use ``--lora``

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

The LoRA scale can be specified after the model path by placing a ``;`` (semicolon) and
then using the named argument ``scale``

When a scale is not specified, 1.0 is assumed.

Named arguments when loading a LoRA are seperated by the ``;`` character and are
not positional, meaning they can be defined in any order.

Loading arguments available when specifying a LoRA are: ``scale``, ``revision``, ``subfolder``, and ``weight-name``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``scale``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

This example shows loading a LoRA using a huggingface repository slug and specifying scale for it.

.. code-block:: bash

    # Don't expect great results with this example,
    # Try models and LoRA's downloaded from CivitAI

    dgenerate runwayml/stable-diffusion-v1-5 \
    --lora "pcuenq/pokemon-lora;scale=0.5" \
    --prompts "Gengar standing in a field at night under a full moon, highquality, masterpiece, digital art" \
    --inference-steps 40 \
    --guidance-scales 10 \
    --gen-seeds 5 \
    --output-size 800


Specifying the file in a repository directly can be done with the named argument ``weight-name``

Shown below is an SDXL compatible LoRA being used with the SDXL base model and a refiner

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --inference-steps 30 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "sketch of a horse by Leonardo da Vinci" \
    --variant fp16 --dtype float16 \
    --lora "goofyai/SDXL-Lora-Collection;scale=1.0;weight-name=leonardo_illustration.safetensors" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    dgenerate runwayml/stable-diffusion-v1-5 \
    --lora "pcuenq/pokemon-lora;scale=0.5;revision=main" \
    --prompts "Gengar standing in a field at night under a full moon, highquality, masterpiece, digital art" \
    --inference-steps 40 \
    --guidance-scales 10 \
    --gen-seeds 5 \
    --output-size 800


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with a LoRA weight in a subfolder :)
    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --lora "huggingface/lora_repo;scale=1.0;subfolder=repo_subfolder;weight-name=lora_weights.safetensors"


If you are loading a .safetensors or other file from a path on disk, only the ``scale`` argument is available

.. code-block:: bash
    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --lora "my_lora.safetensors;scale=1.0"


Specifying Textual Inversions
-----------------------------

One or more Textual Inversion models may be specified with ``--textual-inversions``

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

Arguments pertaining to the loading of each textual inversion model my be specified in the same
way as when using ``--lora`` minus the scale argument.

Available arguments are: ``revision``, ``subfolder``, and ``weight-name``

Named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk they should not be used.


.. code-block::
    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the Isometric_Dreams, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    # This is a non working example as I do not know of a repo that utilizes revisions with
    # textual inversion weights :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with a textual
    # inversion weight in a subfolder :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;subfolder=repo_subfolder;weight-name=ti_model.safetensors"


If you are loading a .safetensors or other file from a path on disk, simply do

.. code-block:: bash
    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --textual-inversions "my_ti_model.safetensors"



Specifying an SDXL Refiner
--------------------------

When the main model is an SDXL model and ``--model-type torch-sdxl`` is specified,
you may specify a refiner model with ``--sdxl-refiner-path``.

You can provide paths to a huggingface repo or a model file on disk such as a .safetensors file.

This argument is parsed in much the same way as the argument ``--vae``, except the
model is the first value specified.

Loading arguments available when specifying a refiner are: ``revision``, ``variant``, ``subfolder``, and ``dtype``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``dtype``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

.. code-block:: bash
    # Basic usage of SDXL with a refiner

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"



If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0;revision=main \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you wish to specify a weights variant IE: load "pytorch_model.<variant>.safetensors", from a huggingface
repository that has variants of the same model, use the named argument ``variant``. By default this
value is the same as ``--variant`` when that option is specified for the main model.

.. code-block:: bash

    # This is a non working example as I do not know of a repo with an SDXL refiner
    # in a subfolder :) this is only a syntax example

    huggingface/sdxl_model --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner huggingface/sdxl_refiner;variant=fp16


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with an SDXL refiner
    # in a subfolder :) this is only a syntax example

    huggingface/sdxl_model --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner huggingface/sdxl_refiner;subfolder=repo_subfolder


Batch Processing Configuration From STDIN
-----------------------------------------

Program configuration can be read from STDIN and processed in batch with model caching,
in order to increase speed when many invocations with different arguments are desired.

Loading the necessary libraries and bringing models into memory is quite slow, so using the program this
way allows for multiple invocations using different arguments, without needing to load the libraries and
models multiple times, only the first time, or in the case of models the first time the model is encountered.

Changing ``--model-type``, ``--revision``, ``--variant``, ``--lora``, ``--vae``, ``--textual-inversions``,
``--scheduler``, or ``--safety-checker`` when loading a model from a repository or file path that has
already been used will cause a cache miss, and a new instance of the model will be created in memory for
what is specified in those arguments.

When loading multiple different models be aware that they will all be retained in memory for
the duration of program execution, unless all models are flushed using the ``\clear_model_cache`` directive.
Memory consumption may become and issue if you are not careful.

Environmental variables will be expanded in the provided input to **STDIN** when using this feature,
you may use Unix style notation for environmental variables even on Windows.

Empty lines and comments starting with ``#`` will be ignored.

You can create a multiline continuation using ``\`` to indicate that a line continues.

The Following is an example input file **my-config.txt**:

.. code-block::

    # Comments in the file will be ignored

    # Guarantee unique file names are generated under the output directory by specifying unique seeds

    CompVis/stable-diffusion-v1-4 --prompts "an astronaut riding a horse" --seeds 41509644783027 --output-path output --inference-steps 30 --guidance-scales 10
    CompVis/stable-diffusion-v1-4 --prompts "a cowboy riding a horse" --seeds 78553317097366 --output-path output --inference-steps 30 --guidance-scales 10
    CompVis/stable-diffusion-v1-4 --prompts "a martian riding a horse" --seeds 22797399276707 --output-path output --inference-steps 30 --guidance-scales 10

    # Guarantee that no overwrites happen by specifying different output paths for each invocation

    stabilityai/stable-diffusion-2-1 --prompts "an astronaut riding a horse" --output-path unique_output_1  --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a cowboy riding a horse" --output-path unique_output_2 --inference-steps 30 --guidance-scales 10

    # Multiline continuations are possible by using \

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" \
    --output-path unique_output_3  \

    # There can be comments or newlines within the continuation
    --inference-steps 30 \
    --guidance-scales 10


    # A clear model cache directive can be used inbetween invocations if cached models that
    # are no longer needed in your generation pipeline start causing out of memory issues

    \clear_model_cache


    # This model was used before but will have to be fully instantiated from scratch again
    # after a cache flush which may take some time

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" \
    --output-path unique_output_4  \


To utilize the file on Linux, pipe it into the command or use redirection:

.. code-block:: bash

    # Pipe
    cat my-config.txt | dgenerate

    # Redirection
    dgenerate < my-config.txt


On Windows CMD:

.. code-block:: bash

    dgenerate < my-arguments.txt


On Windows Powershell:

.. code-block:: powershell

    Get-Content my-arguments.txt | dgenerate


Choosing a specific GPU for CUDA
--------------------------------

The desired GPU to use for CUDA acceleration can be selected using ``--device cuda:N`` where ``N`` is
the device number of the GPU as reported by ``nvidia-smi``.

.. code-block:: bash

    # Console 1, run on GPU 0

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut_1 \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --device cuda:0

    # Console 2, run on GPU 1 in parallel

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a cow" \
    --output-path astronaut_2 \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --device cuda:1



