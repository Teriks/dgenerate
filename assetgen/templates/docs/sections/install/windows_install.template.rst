Windows Install
===============

You can install using the Windows installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_. Running
``dgenerate-network-installer.exe`` with no arguments opens the window. The
command line is described under :ref:`network-installer`. You can also install
manually with pipx, (or pip if you want) as described below.


Manual Install
--------------

Install Python 3.14. dgenerate requires Python >=3.11, except 3.14.1, and older than 3.15.
Use 3.14.8. Do not install 3.14.1.

https://www.python.org/ftp/python/3.14.8/python-3.14.8-amd64.exe

Make sure you select the option "Add to PATH" in the python installer,
otherwise invoke python directly using it's full path while installing the tool.

The published packages install from Windows wheels. That includes the Rust
extensions (``tokenizers``, ``safetensors``, ``hf-xet``, ``pydantic-core``) and
the C extensions (``sentencepiece``, PyAV, ``patchmatch-cython``, spaCy and its
compiled dependencies). Visual Studio Build Tools and a Rust compiler are not
required. They are only needed if pip is told to build a package from source.

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
    # * xllamacpp (used for the llama prompt upscaler plugin)
    # * bitsandbytes
    # * triton_windows
    # * console_ui_opengl (OpenGL Console UI preview; plays video and audio)
    # * console_ui_vulkan (default Console UI preview on Windows, Linux, and macOS;
    #   plays video and audio. The network installer selects this extra.
    #   DGENERATE_CONSOLE_UI_VULKAN=0 keeps the OpenGL viewer)

    # The commands below use the CUDA 13.2 torch index.
    # CUDA 13.0 through 13.1 use --extra-index-url https://download.pytorch.org/whl/cu130/
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    pipx install dgenerate ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # with NCNN upscaler support

    pipx install dgenerate[ncnn] ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # The GPU and CPU wheels share one version, so the GPU index is --index-url.

    # CUDA 13.2+

    pipx install "dgenerate[xllamacpp]" ^
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple"

    # CUDA 12.8 through 12.9

    pipx install "dgenerate[xllamacpp]" ^
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple"

    # CUDA 13.0 through 13.1

    pipx install "dgenerate[xllamacpp]" ^
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple"

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Windows AMD torch uses https://repo.amd.com/rocm/whl-multi-arch/
    # with the Vulkan xllamacpp index. Intel Arc uses the XPU index.

    pipx install "dgenerate[xllamacpp]" ^
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple"

    # If you want a specific version

    pipx install dgenerate==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # with NCNN upscaler support and a specific version

    pipx install dgenerate[ncnn]==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # You can install without pipx into your own environment like so

    pip install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu132/

    # Or with NCNN

    pip install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/cu132/

    # CUDA 13.2+

    pip install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple

    # CUDA 12.8 through 12.9

    pip install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple

    # CUDA 13.0 through 13.1

    pip install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Windows AMD torch uses https://repo.amd.com/rocm/whl-multi-arch/
    # with the Vulkan xllamacpp index. Intel Arc uses the XPU index.

    pip install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple


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
    # * xllamacpp (used for the llama prompt upscaler plugin)
    # * bitsandbytes
    # * triton_windows
    # * console_ui_opengl (OpenGL Console UI preview; plays video and audio)
    # * console_ui_vulkan (default Console UI preview on Windows, Linux, and macOS;
    #   plays video and audio. The network installer selects this extra.
    #   DGENERATE_CONSOLE_UI_VULKAN=0 keeps the OpenGL viewer)

    # The commands below use the CUDA 13.2 torch index.
    # CUDA 13.0 through 13.1 use --extra-index-url https://download.pytorch.org/whl/cu130/
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu132/

    # Install with pip into the environment, include NCNN

    pip install --editable .[dev,ncnn] --extra-index-url https://download.pytorch.org/whl/cu132/

    # CUDA 13.2+

    pip install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple

    # CUDA 12.8 through 12.9

    pip install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple

    # CUDA 13.0 through 13.1

    pip install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Windows AMD torch uses https://repo.amd.com/rocm/whl-multi-arch/
    # with the Vulkan xllamacpp index. Intel Arc uses the XPU index.

    pip install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple


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