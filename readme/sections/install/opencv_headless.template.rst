Linux with opencv-python-headless (libGL.so.1 issues)
=====================================================

If you are running into issues with OpenCV being unable to load ``libGL.so.1``
because your system is headless.

If it is applicable, install these: ``libgl1 libglib2.0-0``

.. code-block:: bash

    sudo apt install libgl1 libglib2.0-0

If that does not sound reasonable for your systems setup. install dgenerate into
a virtual environment as described above in the linux install section.

Then activate the environment and remove ``python-opencv`` and ``python-opencv-headless``,
then reinstall ``python-opencv-headless``.

.. code-block:: bash

    source venv\bin\activate

    pip uninstall python-opencv-headless python-opencv

    pip install python-opencv-headless==@COMMAND_OUTPUT[{"command": "python ../../scripts/get_cur_headless_opencv_ver.py", "block":false}]


This work around is needed because some of dgenerates dependencies depend on ``python-opencv`` and pip
gives no way to prevent it from being installed when installing from a wheel.

``python-opencv`` expects you to probably have a window manager and GL, maybe mesa.

dgenerate does not use anything that requires ``python-opencv`` over ``python-opencv-headless``, so you can
just replace the package in the environment with the headless version.

This is not compatible with ``pipx`` unfortunately.