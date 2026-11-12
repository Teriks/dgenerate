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

On Windows, for dgenerate 6.0.0 and newer, ``--msvc`` installs the current
Visual Studio Build Tools so ``torch.compile`` can find ``cl`` on CPU and Intel
XPU. Older releases do not offer that option. With ``--silent`` the installer
runs Microsoft's bootstrapper. Without ``--silent`` the window opens with that
option selected. The checkbox on the extras screen does the same thing, and it
starts selected when the installer detects an Intel XPU. A silent install still
runs Build Tools only when ``--msvc`` is passed.
dgenerate does not ship the compiler. The C++ workload is several GB.
Microsoft's installer stays hidden; Windows asks for administrator approval,
and continuing accepts Microsoft's license. An existing MSVC toolset is left
as it is.

``--version`` is a release tag from that page, including the leading ``v``.
``--branch`` is a branch name. With neither one, ``--silent`` installs the
``master`` branch. Omit ``--extras`` and the installer chooses extras for the
detected GPU, including ``console_ui_vulkan`` when the downloaded source
defines that extra.

After the package is installed, the installer runs
``dgenerate --sub-command kernel-warmup`` with the new binary. That caches
the GGUF and ComfyUI quant kernels for this computer. A dgenerate version
that does not provide the subcommand skips the step.

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

    # also install the MSVC compiler used by torch.compile (Windows)
    dgenerate-network-installer.exe --silent --msvc

    # remove the installation
    dgenerate-network-installer.exe --uninstall

On Linux and macOS, use ``./dgenerate-network-installer`` in place of
``dgenerate-network-installer.exe`` for the branch, extras, and uninstall
commands above.
