.. _network-installer:

Network Installer
=================

Each release on the `Releases Page <https://github.com/Teriks/dgenerate/releases>`_
includes a compiled installer. The Windows file is ``dgenerate-network-installer.exe``.
The Linux and macOS file is ``dgenerate-network-installer``. Running the binary
with no arguments opens the installer window.

The same binary installs and uninstalls from a terminal. ``--silent`` is required
for ``--version``, ``--branch``, and ``--extras``. A silent install overwrites an
existing installation. ``--version`` and ``--branch`` cannot be used together.
``--uninstall`` removes the installation and does not open a window.

``--version`` is a release tag from that page, including the leading ``v``.
``--branch`` is a branch name. With neither one, ``--silent`` installs the
``master`` branch. Omit ``--extras`` and the installer chooses extras for the
detected GPU, including ``console_ui_vulkan`` when the downloaded source
defines that extra.

.. code-block::

    # Windows
    dgenerate-network-installer.exe --silent --version v5.0.0

    # Linux or macOS. Mark the download executable once.
    chmod +x dgenerate-network-installer
    ./dgenerate-network-installer --silent --version v5.0.0

    # a development branch
    dgenerate-network-installer.exe --silent --branch version_6.0.0

    # name the extras instead of accepting the GPU defaults
    dgenerate-network-installer.exe --silent --extras bitsandbytes xllamacpp console_ui_vulkan

    # remove the installation
    dgenerate-network-installer.exe --uninstall

On Linux and macOS, use ``./dgenerate-network-installer`` in place of
``dgenerate-network-installer.exe`` for the branch, extras, and uninstall
commands above.
