Linux or WSL Install
====================

You can install using the Linux installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_. Running
``dgenerate-network-installer`` with no arguments opens the window. The
command line is described under :ref:`network-installer`. You can also install
manually with pipx, (or pip if you want) as described below.

First update your system and install build-essential

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt update && sudo apt upgrade
    sudo apt install build-essential

Install CUDA Toolkit 12.6 or newer, or CUDA 13: https://developer.nvidia.com/cuda-downloads

I recommend using the runfile option.

Do not attempt to install a driver from the prompts if using WSL.

Add libraries to linker path:

.. code-block:: bash

    #!/usr/bin/env bash

    # Add to ~/.bashrc

    # For Linux add the following
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # For WSL add the following
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # Add this in both cases as well
    export PATH=/usr/local/cuda/bin:$PATH


When done editing ``~/.bashrc`` do:

.. code-block:: bash

    #!/usr/bin/env bash

    source ~/.bashrc


Install Python >=3.11, except 3.14.1, and older than 3.15 (Debian / Ubuntu) and pipx
-------------------------------------------------------------------------------------

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt install python3 python3-pip python3-wheel python3-venv pipx

    # if you want to use the Tk based GUI, install Tk
    sudo apt install python3-tk

    pipx ensurepath

    source ~/.bashrc


Install dgenerate
-----------------

.. code-block:: bash

    #!/usr/bin/env bash

    # possible dgenerate package extras:

    # * ncnn
    # * xllamacpp (used for the llama prompt upscaler plugin)
    # * bitsandbytes
    # * console_ui_opengl (OpenGL Console UI preview; plays video and audio)
    # * console_ui_vulkan (default Console UI preview on Windows and Linux (opt-in on macOS with DGENERATE_CONSOLE_UI_VULKAN=1);
    #   plays video and audio. The network installer selects this extra.
    #   DGENERATE_CONSOLE_UI_VULKAN=0 keeps the OpenGL viewer)

    # The commands below use the CUDA 13.2 torch index.
    # CUDA 13.0 through 13.1 use --extra-index-url https://download.pytorch.org/whl/cu130/
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    # install with just support for torch

    pipx install dgenerate \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # With NCNN upscaler support (extra)

    pipx install dgenerate[ncnn] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # The GPU and CPU wheels share one version, so the GPU index is --index-url.

    # CUDA 13.2+

    pipx install "dgenerate[xllamacpp]" \
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple"

    # CUDA 12.8 through 12.9

    pipx install "dgenerate[xllamacpp]" \
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple"

    # CUDA 13.0 through 13.1

    pipx install "dgenerate[xllamacpp]" \
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple"

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Intel Arc uses the XPU index. AMD on Linux uses the ROCm section.

    pipx install "dgenerate[xllamacpp]" \
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple"

    # If you want a specific version

    pipx install dgenerate==@VERSION \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu132/"

    # You can install without pipx into your own environment like so

    pip3 install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu132/

    # Or with NCNN

    pip3 install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/cu132/

    # CUDA 13.2+

    pip3 install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple

    # CUDA 12.8 through 12.9

    pip3 install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple

    # CUDA 13.0 through 13.1

    pip3 install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Intel Arc uses the XPU index. AMD on Linux uses the ROCm section.

    pip3 install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple


It is recommended to install dgenerate with pipx if you are just intending
to use it as a command line program, if you want to install into your own
virtual environment you can do so like this:

.. code-block:: bash

    #!/usr/bin/env bash

    # in the top of the repo make
    # an environment and activate it

    python3 -m venv venv
    source venv/bin/activate

    # Install with pip into the environment (editable, for development)

    # The commands below use the CUDA 13.2 torch index.
    # CUDA 13.0 through 13.1 use --extra-index-url https://download.pytorch.org/whl/cu130/
    # CUDA 12.6 through 12.9, and Maxwell (5.x), Pascal (6.x), or Volta (7.0), use
    # --extra-index-url https://download.pytorch.org/whl/cu126/

    pip3 install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu132/

    # Install with pip into the environment (non-editable)

    pip3 install . --extra-index-url https://download.pytorch.org/whl/cu132/

    # CUDA 13.2+

    pip3 install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu132/ --extra-index-url https://pypi.org/simple

    # CUDA 12.8 through 12.9

    pip3 install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple

    # CUDA 13.0 through 13.1

    pip3 install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple

    # Older NVIDIA (Maxwell, Pascal, Volta, or a driver before CUDA 12.8).
    # Intel Arc uses the XPU index. AMD on Linux uses the ROCm section.

    pip3 install --editable ".[dev,xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple


Run ``dgenerate`` to generate images:

.. code-block:: bash

    #!/usr/bin/env bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate sd2-community/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10


Linux with ROCm (AMD Cards)
===========================

On Linux you can use the ROCm torch backend with AMD cards. pytorch.org publishes
those wheels for Linux. ROCm 7.2 uses
``--extra-index-url https://download.pytorch.org/whl/rocm7.2/``. ROCm 7.14 uses
``--extra-index-url https://download.pytorch.org/whl/rocm7.14/``. The commands
below use 7.2.

Windows AMD does not use those indexes. Torch comes from
``--extra-index-url https://repo.amd.com/rocm/whl-multi-arch/``.
xllamacpp uses ``--index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan``.
The network installer selects those indexes on its own.

ROCm has been minimally verified to work with dgenerate using a rented
MI300X AMD GPU instance / space, and has not been tested extensively.

When specifying any ``--device`` value use ``cuda``, ``cuda:1``, etc. as you would for Nvidia GPUs.

You need to first install ROCm support, follow: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

Then use the ROCm index above when installing via ``pip`` or ``pipx``.

Install Python >=3.11, except 3.14.1, and older than 3.15 (Debian / Ubuntu) and pipx
-------------------------------------------------------------------------------------

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt install python3 python3-pip pipx python3-venv python3-wheel

    # if you want to use the Tk based GUI, install Tk
    sudo apt install python3-tk

    pipx ensurepath

    source ~/.bashrc


Setup Environment
-----------------

You may need to export the environmental variable ``PYTORCH_ROCM_ARCH`` before attempting to use dgenerate.

This value will depend on the model of your card, you may wish to add this and any other necessary
environmental variables to ``~/.bashrc`` so that they persist in your shell environment.

For details, see: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html

Generally, this information can be obtained by running the command: ``rocminfo``

.. code-block:: bash

    # example

    export PYTORCH_ROCM_ARCH="gfx1030"


Install dgenerate
-----------------

.. code-block:: bash

    #!/usr/bin/env bash

    # possible dgenerate package extras: ncnn, xllamacpp
    # xllamacpp is used for the llama prompt upscaler plugin

    # install with just support for torch

    pipx install dgenerate \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm7.2/"

    # With NCNN upscaler support

    pipx install dgenerate[ncnn] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm7.2/"

    # ROCm 7.2

    pipx install "dgenerate[xllamacpp]" \
    --pip-args "--index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-7.2.4 --extra-index-url https://download.pytorch.org/whl/rocm7.2/ --extra-index-url https://pypi.org/simple"

    # If you want a specific version

    pipx install dgenerate==@VERSION \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm7.2/"


    # bitsandbytes is the bitsandbytes extra. It installs the version dgenerate
    # pins, which is what --quantizer bnb uses. sdnq is installed with
    # dgenerate and does not need this extra.

    pipx install "dgenerate[bitsandbytes]" \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm7.2/"


    # You can install without pipx into your own environment like so

    pip3 install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/rocm7.2/

    # Or with NCNN

    pip3 install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/rocm7.2/

    # ROCm 7.2

    pip3 install "dgenerate[xllamacpp]==@VERSION" --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-7.2.4 --extra-index-url https://download.pytorch.org/whl/rocm7.2/ --extra-index-url https://pypi.org/simple


    pip3 install "dgenerate[bitsandbytes]==@VERSION" --extra-index-url https://download.pytorch.org/whl/rocm7.2/

