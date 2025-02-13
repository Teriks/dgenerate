Google Colab Install
====================

The following cell entries will get you started in a Google Collab environment.

Make sure you select a GPU runtime for your notebook, such as the T4 runtime.


1.) Install venv.

.. code-block:: bash

    !apt install python3-venv

2.) Create a virtual environment.

.. code-block:: bash

    !python3 -m venv venv

3.) Install dgenerate, you must activate the virtual environment in the same cell.

.. code-block:: bash

    !source /content/venv/bin/activate; pip install dgenerate==@VERSION --extra-index-url https://download.pytorch.org/whl/cu121

4.) Finally you can run dgenerate, you must prefix all calls to dgenerate with an activation of the virtual environment, as
the virtual environment is not preserved between cells.  For brevity, and as an example, just print the help text here.

.. code-block:: bash

    !source /content/venv/bin/activate; dgenerate --help