Overview
========

**dgenerate** is a command line tool for generating images and animation sequences using stable diffusion.

**dgenerate** can generate multiple images or animated outputs using multiple combinations of input parameters
for stable diffusion in batch, so that the differences in generated output can be compared / curated easily.

Animated output can be produced by processing every frame of a video, gif, webp through stable diffusion as
an image seed with a given prompt and generation parameters.

Video of infinite runtime can be processed without memory constraints.

GIF's and WebP can also be processed, with memory constraints.

Generation utilizes https://huggingface.co pytorch and optionally flax models.


Help
----

.. code-block::

    usage: dgenerate [-h] [--model-type MODEL_TYPE] [--revision REVISION] [-d DEVICE] [-t DTYPE] [-s OUTPUT_SIZE]
                     [-o OUTPUT_PATH] [-p PROMPTS [PROMPTS ...]] [-se SEEDS [SEEDS ...] | -gse GEN_SEEDS]
                     [-af ANIMATION_FORMAT] [-fs FRAME_START] [-fe FRAME_END] [-is [IMAGE_SEEDS ...]]
                     [-iss [IMAGE_SEED_STRENGTHS ...]] [-gs [GUIDANCE_SCALES ...]] [-ifs [INFERENCE_STEPS ...]]
                     model_path

    Stable diffusion batch image generation tool with support for video / gif / webp animation transcoding.

    positional arguments:
      model_path            huggingface model repository, repository slug/URI or path to folder on disk.

    options:
      -h, --help            show this help message and exit
      --model-type MODEL_TYPE
                            Use when loading different model types. Currently supported: torch. (default: torch)
      --revision REVISION   The model revision to use, (The git branch / tag, default is "main")
      -d DEVICE, --device DEVICE
                            cuda / cpu. (default: cuda)
      -t DTYPE, --dtype DTYPE
                            Model precision: float16 / float32 / auto. (default: auto)
      -s OUTPUT_SIZE, --output-size OUTPUT_SIZE
                            Image output size. If an image seed is used it will be resized to this dimension with aspect
                            ratio maintained, width will be fixed and a new height will be calculated. If only one integer
                            value is provided, that is the value for both dimensions. X/Y dimension values should be
                            separated by "x". (default: 512x512)
      -o OUTPUT_PATH, --output-path OUTPUT_PATH
                            Output path for generated images and files. This directory will be created if it does not
                            exist. (default: ./output)
      -p PROMPTS [PROMPTS ...], --prompts PROMPTS [PROMPTS ...]
                            List of prompts to try, an image group is generated for each prompt, prompt data is split by ;
                            (semi-colon). The first value is the positive text influence, things you want to see. The
                            Second value is negative influence IE. things you don't want to see. Example: --prompts "shrek
                            flying a tesla over detroit; clouds, rain, missiles". (default: [(empty string)])
      -se SEEDS [SEEDS ...], --seeds SEEDS [SEEDS ...]
                            List of seeds to try, define fixed seeds to achieve deterministic output. This argument may
                            not be used when --gse/--gen-seeds is used. (default: [randint(0, 99999999999999)])
      -gse GEN_SEEDS, --gen-seeds GEN_SEEDS
                            Auto generate N random seeds to try. This argument may not be used when -se/--seeds is used.
      -af ANIMATION_FORMAT, --animation-format ANIMATION_FORMAT
                            Output format when generating an animation from an input video / gif / webp etc. Value must be
                            one of "mp4", "gif", or "webp". (default: mp4)
      -fs FRAME_START, --frame-start FRAME_START
                            Starting frame slice point for animated files, the specified frame will be included.
      -fe FRAME_END, --frame-end FRAME_END
                            Ending frame slice point for animated files, the specified frame will be included.
      -is [IMAGE_SEEDS ...], --image-seeds [IMAGE_SEEDS ...]
                            List of image seeds to try when processing image seeds, these may be URLs or file paths.
                            Videos / GIFs / WEBP files will result in frames being rendered as well as an animated output
                            file being generated if more than one frame is available in the input file.
      -iss [IMAGE_SEED_STRENGTHS ...], --image-seed-strengths [IMAGE_SEED_STRENGTHS ...]
                            List of image seed strengths to try. Closer to 0 means high usage of the seed image (less
                            noise convolution), 1 effectively means no usage (high noise convolution). Low values will
                            produce something closer or more relevant to the input image, high values will give the AI
                            more creative freedom. (default: [0.8])
      -gs [GUIDANCE_SCALES ...], --guidance-scales [GUIDANCE_SCALES ...]
                            List of guidance scales to try. Guidance scale effects how much your text prompt is
                            considered. Low values draw more data from images unrelated to text prompt. (default: [5])
      -ifs [INFERENCE_STEPS ...], --inference-steps [INFERENCE_STEPS ...]
                            Lists of inference steps values to try. The amount of inference (denoising) steps effects
                            image clarity to a degree, higher values bring the image closer to what the AI is targeting
                            for the content of the image. Values between 30-40 produce good results, higher values may
                            improve image quality and or change image content. (default: [30])



Windows Install
===============

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

Create a virtual environment using virtualenv from the command prompt in a directory of your choosing:

.. code-block:: bash

    pip install virtualenv wheel
    python -m venv dgenerate_environment


Activate the environment:

.. code-block:: bash

    dgenerate_environment\Scripts\activate

Install into environment:

.. code-block:: bash

    pip install git+https://github.com/Teriks/dgenerate.git --extra-index-url https://download.pytorch.org/whl/cu118/

    # if you want a specific version

    pip install git+https://github.com/Teriks/dgenerate.git@v0.2.1 --extra-index-url https://download.pytorch.org/whl/cu118/

Run **dgenerate** to generate images, you must have the environment active for the command to be found:

.. code-block:: bash

    dgenerate --help

    dgenerate CompVis/stable-diffusion-v1-4 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10

Linux or WSL Install
====================

Install CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive

OR

Install CUDA Toolkit 12.*: https://developer.nvidia.com/cuda-downloads


I recommend using the runfile option:

.. code-block:: bash

    # CUDA Toolkit 11.8 For Ubuntu / Debian / WSL

    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run

    # CUDA Toolkit 12.1.1 For Ubuntu / Debian / WSL

    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    sudo sh cuda_12.1.1_530.30.02_linux.run


Do not attempt to install a driver from the prompts if using WSL.

Add libraries to linker path:

.. code-block:: bash

    # Add to .bashrc or environment in general

    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH


Install Python 3.10+ (Debian / Ubuntu)
--------------------------------------


.. code-block:: bash

    sudo apt update && sudo apt upgrade
    sudo apt install python3.10 python3-virtualenv python3-wheel


Optional Prerequisite JAX / Flax
--------------------------------

Install Jax / Flax to add the ability to load flax models. This is very buggy / slow and I don't recommend.

.. code-block:: bash

    # Select what is appropriate considering which CUDA toolkit you installed

    # CUDA 11 installation
    pip install --upgrade flax "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CUDA 12 installation
    pip install --upgrade flax "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Install dgenerate
-----------------

Create a virtual environment using virtualenv from the command prompt in a directory of your choosing:

.. code-block:: bash

    python3 -m venv dgenerate_environment

Activate the environment:

.. code-block:: bash

    source dgenerate_environment/bin/activate

Install into environment:

.. code-block:: bash

    pip3 install git+https://github.com/Teriks/dgenerate.git

    # if you want a specific version

    pip3 install git+https://github.com/Teriks/dgenerate.git@v0.2.1


Run **dgenerate** to generate images, you must have the environment active for the command to be found:

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

Adjust output size to 512x512 and output generated images to 'astronaut' folder.

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


Animated Output
---------------

When an animated image seed is given, animated output will be produced in the format of your choosing.

Use a GIF of a man riding a horse to create an animation of an astronaut riding a horse.

Output to an MP4.  See ``--help`` for information about formats supported by ``--animation-format``

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
