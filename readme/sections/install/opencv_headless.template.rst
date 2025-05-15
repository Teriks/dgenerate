Linux with opencv-python-headless (libGL.so.1 issues)
=====================================================

If you are running into issues with OpenCV being unable to load ``libGL.so.1``
because your system is headless and you are using the extra: ``ncnn``

If it is applicable, install these: ``libgl1 libglib2.0-0``

.. code-block:: bash

    sudo apt install libgl1 libglib2.0-0

If that does not sound reasonable for your systems setup. install dgenerate into
a virtual environment as described above in the linux install section.

Then activate the environment and remove ``opencv-python`` and ``opencv-python-headless``,
then reinstall ``opencv-python-headless``.

.. code-block:: bash

    source venv\bin\activate

    pip uninstall opencv-python-headless opencv-python

    pip install opencv-python-headless~=@COMMAND_OUTPUT[{"command": "python ../../scripts/get_cur_headless_opencv_ver.py", "block":false}]


This work around is needed because ``ncnn`` depends on ``opencv-python`` and pip
gives no way to prevent it from being installed when installing from a wheel.

``opencv-python`` expects you to probably have a window manager and GL, maybe mesa.

dgenerate does not use anything that requires ``opencv-python`` over ``opencv-python-headless``, so you can
just replace the package in the environment with the headless version.

If you are using pipx, you can do this:

.. code-block:: bash

    pipx runpip dgenerate uninstall opencv-python-headless opencv-python

    pipx inject dgenerate opencv-python-headless~=@COMMAND_OUTPUT[{"command": "python ../../scripts/get_cur_headless_opencv_ver.py", "block":false}]