Installing From Development Branches
====================================

You can also install dgenerate directly from git to try out versions in development.

In this instance, replace ``BRANCH_NAME`` with the branch you want to install from.

Note that the name of the ``pip`` executable may be named ``pip3`` on some systems.

.. code-block:: bash

    # cuda

    pip install git+https://github.com/Teriks/dgenerate@BRANCH_NAME --extra-index-url https://download.pytorch.org/whl/cu132

    # ROCm

    pip install git+https://github.com/Teriks/dgenerate@BRANCH_NAME --extra-index-url https://download.pytorch.org/whl/rocm7.2/

    # With extras. quant is not an extra. These are:
    # ncnn, xllamacpp, bitsandbytes, console_ui_vulkan, console_ui_opengl

    pip install "dgenerate[ncnn,console_ui_vulkan] @ git+https://github.com/Teriks/dgenerate@BRANCH_NAME" --extra-index-url https://download.pytorch.org/whl/cu132


This same syntax should work with ``pipx`` as well, as long as you have ``git`` installed.