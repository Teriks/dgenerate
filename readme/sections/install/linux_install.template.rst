Linux or WSL Install
====================

First update your system and install build-essential

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt update && sudo apt upgrade
    sudo apt install build-essential

Install CUDA Toolkit 12.*: https://developer.nvidia.com/cuda-downloads

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


Install Python >=3.10,<3.13 (Debian / Ubuntu) and pipx
------------------------------------------------------

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt install python3 python3-pip python3-wheel python3-venv python3-tk
    pipx ensurepath

    source ~/.bashrc


Install dgenerate
-----------------

.. code-block:: bash

    #!/usr/bin/env bash

    # possible dgenerate package extras: ncnn, gpt4all, gpt4all_cuda

    # install with just support for torch

    pipx install dgenerate \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # With NCNN upscaler support

    pipx install dgenerate[ncnn] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # If you want a specific version

    pipx install dgenerate==@VERSION \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # You can install without pipx into your own environment like so

    pip3 install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu124/

    # Or with NCNN

    pip3 install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/cu124/


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

    pip3 install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu124/

    # Install with pip into the environment (non-editable)

    pip3 install . --extra-index-url https://download.pytorch.org/whl/cu124/


Run ``dgenerate`` to generate images:

.. code-block:: bash

    #!/usr/bin/env bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10


Linux with ROCm (AMD Cards)
===========================

On Linux you can use the ROCm torch backend with AMD cards. This is only supported on Linux, as
torch does not distribute this backend for Windows.

ROCm has been minimally verified to work with dgenerate using a rented
MI300X AMD GPU instance / space, and has not been tested extensively.

When specifying any ``--device`` value use ``cuda``, ``cuda:1``, etc. as you would for Nvidia GPUs.

You need to first install ROCm support, follow: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

Then use: ``--extra-index-url https://download.pytorch.org/whl/rocm6.2.4/`` when installing via ``pip`` or ``pipx``.

Install Python >=3.10,<3.13 (Debian / Ubuntu) and pipx
------------------------------------------------------

.. code-block:: bash

    #!/usr/bin/env bash

    sudo apt install python3 python3-pip pipx python3-venv python3-wheel
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

    # install with just support for torch

    pipx install dgenerate \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm6.2.4/"

    # With NCNN upscaler support

    pipx install dgenerate[ncnn] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm6.2.4/"

    # If you want a specific version

    pipx install dgenerate==@VERSION \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/rocm6.2.4/"

    # You can install without pipx into your own environment like so

    pip3 install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/rocm6.2.4/

    # Or with NCNN

    pip3 install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/rocm6.2.4/
