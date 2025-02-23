Windows Install
===============

You can install using the Windows installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_, or you can manually
install with pipx, (or pip if you want) as described below.


Manual Install
--------------

Install Visual Studios build tools, make sure "Desktop development with C++" is selected, unselect anything you do not need.

https://aka.ms/vs/17/release/vs_BuildTools.exe

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

    pipx install dgenerate ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # with NCNN upscaler support

    pipx install dgenerate[ncnn] ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # If you want a specific version

    pipx install dgenerate==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # with NCNN upscaler support and a specific version

    pipx install dgenerate[ncnn]==@VERSION ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu124/"

    # You can install without pipx into your own environment like so

    pip install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu124/

    # Or with NCNN

    pip install dgenerate[ncnn]==@VERSION --extra-index-url https://download.pytorch.org/whl/cu124/


It is recommended to install dgenerate with pipx if you are just intending
to use it as a command line program, if you want to develop you can install it from
a cloned repository like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python -m venv venv
    venv\Scripts\activate

    # Install with pip into the environment

    pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu124/

    # Install with pip into the environment, include NCNN

    pip install --editable .[dev, ncnn] --extra-index-url https://download.pytorch.org/whl/cu124/


Run ``dgenerate`` to generate images:

.. code-block:: bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate stabilityai/stable-diffusion-2-1 ^
    --prompts "an astronaut riding a horse" ^
    --output-path output ^
    --inference-steps 40 ^
    --guidance-scales 10