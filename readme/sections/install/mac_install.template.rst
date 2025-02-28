MacOS Install (Apple Silicon Only)
==================================

MacOS on Apple Silicon (arm64) is experimentally supported.

Rendering can be preformed in CPU only mode, and with hardware acceleration using ``--device mps`` (Metal Performance Shaders).

The default device on MacOS is ``mps`` unless specified otherwise.

You can install on MacOS by first installing python from the universal ``pkg`` installer
located at: https://www.python.org/downloads/release/python-3126/

It is also possible to install Python using `homebrew <homebrew_1_>`_, though tkinter will
not be available meaning that you cannot run the Console UI.

Once you have done so, you can install using ``pipx`` (recommended), or create a virtual
environment in a directory of your choosing and install ``dgenerate`` into it.

Do not specify any ``--extra-index-url`` to ``pip``, it is not necessary on MacOS.

When using SDXL on MacOS with ``--dtype float16``, you might need to specify
``--vae AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix`` if your images
are rendering solid black.

MacOS pipx install
------------------

Installing with ``pipx`` allows you to easily install ``dgenerate`` and
have it available globally from the command line without installing
global python site packages.

.. code-block:: bash

    #!/usr/bin/env bash

    # install pipx

    pip3 install pipx

    # install dgenerate into an isolated
    # environment with pipx

    # possible dgenerate package extras: ncnn, gpt4all

    pipx install dgenerate==@VERSION
    pipx ensurepath

    # open a new terminal or logout & login

    # launch the Console UI to test the install.
    # tkinter will be available when you install
    # python using the dmg from pythons official
    # website

    dgenerate --console

    # or generate images

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10


If you want to upgrade dgenerate, uninstall it first and then install the new version with ``pipx``.

.. code-block:: bash

    pipx uninstall dgenerate
    pipx install dgenerate==@VERSION


MacOS venv install
------------------

You can also manually install into a virtual environment
of your own creation.

.. code-block:: bash

    #!/usr/bin/env bash

    # create the environment

    python3 -m venv dgenerate_venv

    # you must activate this environment
    # every time you want to use dgenerate
    # with this install method

    source dgenerate_venv/bin/activate

    # install dgenerate into an isolated environment

    # possible dgenerate package extras: ncnn, gpt4all

    pip3 install dgenerate==@VERSION

    # launch the Console UI to test the install.
    # tkinter will be available when you install
    # python using the dmg from pythons official
    # website

    dgenerate --console

    # or generate images

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10
