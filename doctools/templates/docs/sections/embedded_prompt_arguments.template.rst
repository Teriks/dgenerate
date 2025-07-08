Embedded Prompt Arguments
=========================

You can embed certain diffusion arguments into your prompt on a per-prompt basis.

Meaning those arguments only apply to that prompt.

Notably, the special embedded arguments ``<weighter: ...>`` and ``<upscaler: ...>`` can be used
to define the ``--prompt-weighter`` and ``--prompt-upscaler`` plugins that act on your prompt.

``<upscaler: ...>`` is unlike other embedded arguments in that it can be mentioned multiple times
in a row to create a chain of prompt upscaling operations using different prompt upscaler plugin URIs.

The rest of the specifiable arguments are select members of the `DiffusionArguments <https://dgenerate.readthedocs.io/en/@REVISION/dgenerate_submodules.html#dgenerate.pipelinewrapper.DiffusionArguments>`_
class from dgenerate's library API.

You may not specify prompt related arguments aside from the aforementioned ``weighter`` and ``upscaler``.

You may not specify arguments related to image inputs either.

All other arguments are fair game, for example ``inference_steps``

.. code-block:: jinja

    # override inference steps for the
    # second prompt variation in particular

    stabilityai/stable-diffusion-2-1
    --inference-steps 30
    --guidance-scales 5
    --clip-skips 0
    --gen-seeds 1
    --output-path output
    --output-size 512x512
    --prompts "hello world!" "<inference-steps: 50> hello world!"


Of the arguments mentioned in the `DiffusionArguments <https://dgenerate.readthedocs.io/en/@REVISION/dgenerate_submodules.html#dgenerate.pipelinewrapper.DiffusionArguments>`_ class,
these are the arguments that are available for use:

@COMMAND_OUTPUT[python ../../../scripts/prompt_embedded_args_list.py]