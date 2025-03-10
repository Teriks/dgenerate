Writing Plugins
===============

dgenerate has the capability of loading in additional functionality through the use of
the ``--plugin-modules`` option and ``\import_plugins`` config directive.

You simply specify one or more module directories on disk, paths to python files, or references
to modules installed in the python environment using the argument or import directive.

dgenerate supports implementing image processors, config directives, config template functions,
prompt weighters, and sub-commands through plugins.

~~~~

Image processor plugins
-----------------------

A code example as well as a usage example for image processor plugins can be found
in the `writing_plugins/image_processor <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/image_processor>`_
folder of the examples folder.

The source code for the built in `canny <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/canny.py>`_ processor,
the `openpose <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/openpose.py>`_ processor, and the simple
`pillow image operations <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/imageops.py>`_ processors can also
be of reference as they are written as internal image processor plugins.

~~~~


Config directive and template function plugins
----------------------------------------------

An example for writing config directives can be found in the `writing_plugins/config_directive <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/config_directive>`_  example folder.

Config template functions can also be implemented by plugins, see: `writing_plugins/template_function <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/template_function>`_

Currently the only internal directive that is implemented as a plugin is the ``\image_process`` directive, who's source file
`can be located here <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/batchprocess/image_process_directive.py>`_.

The source file for the ``\image_process`` directive is terse as most of it is implemented as reusable code.

The behavior of ``\image_process`` which is also used for ``--sub-command image-process`` is
`is implemented here <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/image_process>`_.

~~~~


Sub-command plugins
-------------------

Reference for writing sub-commands can be found in the `image-process <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/subcommands/image_process.py>`_
sub-command implementation, and a plugin skeleton file for sub-commands can be found in the
`writing_plugins/sub_command <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/sub_command>`_ example folder.

~~~~


Prompt weighter plugins
-----------------------

Reference for writing prompt weighters can be found in the `CompelPromptWeighter <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptweighters/compelpromptweighter.py>`_
and `SdEmbedPromptWeighter <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptweighters/sdembedpromptweighter.py>`_ internal prompt weighter implementations.

A plugin skeleton file for prompt weighters can be found in the
`writing_plugins/prompt_weighter <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/prompt_weighter>`_
example folder.

~~~~


