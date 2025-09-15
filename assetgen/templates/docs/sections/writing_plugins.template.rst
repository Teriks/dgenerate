Writing Plugins
===============

dgenerate has the capability of loading in additional functionality through the use of
the ``--plugin-modules`` option and ``\import_plugins`` config directive.

You simply specify one or more module directories on disk, paths to python files, or references
to modules installed in the python environment using the argument or import directive.

dgenerate supports implementing image processors, latents processors, config directives, config template functions,
prompt weighters, prompt upscalers, and sub-commands through plugins.

~~~~

Image Processors / Latents Processors
----------------------------------------------

A code example as well as a usage example for image processor plugins can be found
in the `writing_plugins/image_processor <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/image_processor>`_
folder of the examples folder.

The source code for the built in `canny <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/canny.py>`_ processor,
the `openpose <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/openpose.py>`_ processor, and the simple
`pillow image operations <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/imageprocessors/imageops.py>`_ processors can also
be of reference as they are written as internal image processor plugins.

~~~~

Latents processors operate on the latent-space representation used by diffusion models. They can run on
raw/partially denoised latents, on latents used for ``img2img``, or on fully denoised latents being
written to disk. For user-facing usage details, see the "Latents Processors" section of the manual.

Reference implementations can be found in the internal latents processors:
`scale <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/latentsprocessors/scale.py>`_,
`noise <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/latentsprocessors/noise.py>`_, and
`interposer <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/latentsprocessors/interposer.py>`_.

The base interface is implemented in
`LatentsProcessor <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/latentsprocessors/latentsprocessor.py>`_.

An example skeleton for a latents processor plugin can be found in
`writing_plugins/latents_processor <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/latents_processor>`_.

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


Prompt Weighters / Prompt Upscalers
----------------------------------------

Reference for writing prompt weighters can be found in the `CompelPromptWeighter <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptweighters/compelpromptweighter.py>`_
and `SdEmbedPromptWeighter <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptweighters/sdembedpromptweighter.py>`_ internal prompt weighter implementations.

A plugin skeleton file for prompt weighters can be found in the
`writing_plugins/prompt_weighter <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/prompt_weighter>`_
example folder.

In addition to prompt weighters, dgenerate also supports prompt upscaler plugins that can preprocess or
expand prompt text before it is fed to the pipeline. They can be enabled globally with
``--prompt-upscaler`` or inline using the embedded prompt argument ``<upscaler: (uri here)>``. Multiple
upscalers can be chained by repeating the embedded argument.

Reference implementations can be found in the internal prompt upscalers:
`DynamicPrompts <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptupscalers/dynamicpromptsupscaler.py>`_,
`MagicPrompt <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptupscalers/magicpromptupscaler.py>`_,
`Attention <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptupscalers/attentionpromptupscaler.py>`_,
`Translate <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptupscalers/translatepromptupscaler.py>`_, and
`GPT4All <https://github.com/Teriks/dgenerate/blob/@REVISION/dgenerate/promptupscalers/gpt4allpromptupscaler.py>`_.

An example skeleton for writing a prompt upscaler plugin can be found in
`writing_plugins/prompt_upscaler <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/writing_plugins/prompt_upscaler>`_.

For usage details, see the "Prompt upscaling" section of the user manual.
