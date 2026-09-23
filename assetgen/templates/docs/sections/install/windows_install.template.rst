Windows Install
===============

You can install using the Windows installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_, or you can manually
install with pipx, (or pip if you want) as described below.


Manual Install
--------------

Install Visual Studios build tools, make sure "Desktop development with C++" is selected, unselect anything you do not need.

https://aka.ms/vs/17/release/vs_BuildTools.exe

Or

https://visualstudio.microsoft.com/downloads/

Install rust compiler using rustup-init.exe (x64), use the default install options.

https://www.rust-lang.org/tools/install

Install Python:

https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe

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

    # possible dgenerate package extras:

    # * ncnn
    # * xllamacpp
    #   The GPU and CPU wheels share one version, so the GPU index must be --index-url.
    #   CUDA 13.2+ (this uses the cu130 torch index):
    #   pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple
    #   CUDA 12.8 through 12.9: the same command with /cu128 and the cu126 torch index.
    #   CUDA 13.0 through 13.1: the same command with /cu128 and the cu130 torch index.
    #   Older NVIDIA, AMD, or Intel Arc: /vulkan instead of /cu132.
    # * bitsandbytes
    # * triton_windows
    # * console_ui_opengl (OpenGL accelerated Console UI image viewer)

    # The commands below use the CUDA 13.0 torch index.
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    pipx install dgenerate ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu130/"

    # with NCNN upscaler support

    pipx install dgenerate[ncnn] ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu130/"

    # If you want a specific version

    pipx install dgenerate==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu130/"

    # with NCNN upscaler support and a specific version

    pipx install dgenerate[ncnn]==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu130/"

    # You can install without pipx into your own environment like so

    pip install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu130/

    # Or with NCNN

    pip install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/cu130/


It is recommended to install dgenerate with pipx if you are just intending
to use it as a command line program, if you want to develop you can install it from
a cloned repository like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python -m venv venv
    venv\Scripts\activate

    # Install with pip into the environment

    # possible dgenerate package extras:

    # * ncnn
    # * xllamacpp
    #   The GPU and CPU wheels share one version, so the GPU index must be --index-url.
    #   CUDA 13.2+ (this uses the cu130 torch index):
    #   pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple
    #   CUDA 12.8 through 12.9: the same command with /cu128 and the cu126 torch index.
    #   CUDA 13.0 through 13.1: the same command with /cu128 and the cu130 torch index.
    #   Older NVIDIA, AMD, or Intel Arc: /vulkan instead of /cu132.
    # * bitsandbytes
    # * triton_windows
    # * console_ui_opengl (OpenGL accelerated Console UI image viewer)

    # The commands below use the CUDA 13.0 torch index.
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu130/

    # Install with pip into the environment, include NCNN

    pip install --editable .[dev,ncnn] --extra-index-url https://download.pytorch.org/whl/cu130/


Run ``dgenerate`` to generate images:

.. code-block:: bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate sd2-community/stable-diffusion-2-1 ^
    --prompts "an astronaut riding a horse" ^
    --output-path output ^
    --inference-steps 40 ^
    --guidance-scales 10