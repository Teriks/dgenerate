MacOS Install (Apple Silicon Only)
==================================

MacOS on Apple Silicon (arm64) is experimentally supported.

Rendering can be performed in CPU only mode, and with hardware acceleration using ``--device mps`` (Metal Performance Shaders).

The default device on MacOS is ``mps`` unless specified otherwise.

You can install on MacOS by first installing python from the universal ``pkg`` installer
located at: https://www.python.org/downloads/release/python-3126/

It is also possible to install Python using `homebrew <https://brew.sh/>`_, though tkinter will
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
    pipx ensurepath

    # install dgenerate into an isolated
    # environment with pipx

    # possible dgenerate package extras:

    # * ncnn
    # * gpt4all
    # * console_ui_opengl (OpenGL accelerated Console UI image viewer)

    pipx install dgenerate==@VERSION

    # or with extras

    pipx install dgenerate[ncnn,gpt4all,console_ui_opengl]==@VERSION


    # you can attempt to install the pre-release bitsandbytes
    # multiplatform version for MacOS, though, I am not sure if it will
    # function correctly, this will allow use of the --quantizer option
    # and quantizer URI arguments with bitsandbytes.

    pipx inject dgenerate https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.45.1.dev0-py3-none-macosx_13_1_x86_64.whl


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

    # possible dgenerate package extras:

    # * ncnn
    # * gpt4all
    # * console_ui_opengl (OpenGL accelerated Console UI image viewer)

    pip3 install dgenerate==@VERSION

    # or with extras

    pip3 install dgenerate[ncnn,gpt4all,console_ui_opengl]==@VERSION

    # you can attempt to install the pre-release bitsandbytes
    # multiplatform version for MacOS, though, I am not sure if it will
    # function correctly, this will allow use of the --quantizer option
    # and quantizer URI arguments with bitsandbytes.

    pip3 install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.45.1.dev0-py3-none-macosx_13_1_x86_64.whl


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
