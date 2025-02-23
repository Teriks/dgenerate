Prompt Upscaling
================

Prompt upscaler plugins can preprocess your prompt text, and or expand the number of prompts used automatically
by the use of txt2txt LLMs or other methods.

They can be specified globally with the ``--prompt-upscaler`` related arguments of dgenerate, or
per prompt by using the ``<upscaler: ...>`` embedded prompt argument.

Prompt upscalers can be chained together sequentially, simply by specifying multiple plugin URIs.

This works even with prompt upscalers that expand your original prompt into multiple prompts.

You can see which prompt upscalers dgenerate implements via: ``dgenerate --prompt-upscaler-help``
or ``\prompt_upscaler_help`` from within a config script.

Specifying: ``dgenerate --prompt-upscaler-help NAME1 NAME2`` will return help for the named upscaler plugins.

The dynamicprompts prompt upscaler
----------------------------------

`dynamicprompts <https://github.com/adieyal/dynamicprompts>`_ is a library for generating combinatorial
prompt variations using a special prompting syntax.

.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the dynamicprompts prompt upscaler

    dgenerate --prompt-upscaler-help dynamicprompts


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help dynamicprompts]


The magicprompt prompt upscaler
-------------------------------

The ``magicprompt`` upscaler can make use of LLMs to enhance your prompt text.

The default model used is: `MagicPrompt <https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion>`_

Which is a GPT2 finetune focused specifically on prompt generation.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the dynamicprompts prompt upscaler

    dgenerate --prompt-upscaler-help magicprompt


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help magicprompt]


Basic prompt upscaling example
------------------------------

The following is an example making use of the ``dynamicprompts`` prompt upscaler plugin.

@EXAMPLE[../../examples/prompt_upscaler/dynamic-magic-config.dgen]


Prompt upscaling with any LLM
-----------------------------

Pretty much any LLM that is supported by ``transformers`` can be used to upscale prompts.

Here is an example using `Phi-3 Mini Abliterated by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3>`_

@EXAMPLE[../../examples/prompt_upscaler/magic-phi3-config.dgen]
