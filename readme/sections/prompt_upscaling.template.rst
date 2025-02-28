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

The ``magicprompt`` upscaler can make use of LLMs via ``transformers`` to enhance your prompt text.

The default model used is: `MagicPrompt <https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion>`_

Which is a GPT2 finetune focused specifically on prompt generation.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the magicprompt prompt upscaler

    dgenerate --prompt-upscaler-help magicprompt


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help magicprompt]


The gpt4all prompt upscaler
---------------------------

The ``gpt4all`` upscaler can make use of LLMs via ``gpt4all`` to enhance your prompt text.

The default model used is: `Phi-3 Mini Abliterated Q4 GGUF by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF>`_

This prompt upscaler can support any LLM model supported by ``gpt4all==2.8.2``.

Note that this does not currently include ``DeepSeek`` as the native binaries provided by the python packaged
are a bit out of date with mainline ``GPT4ALL`` binaries.

You must have chosen the ``gpt4all`` or ``gpt4all_cuda`` install extra to use this prompt upscaler,
e.g. ``pip install dgenerate[gpt4all]`` or ``pip install dgenerate[gpt4all_cuda]``.

This prompt upscaler use the ``gpt4all`` python binding to perform inference on the cpu or gpu using the native backend provided by ``gpt4all``.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the gpt4all prompt upscaler

    dgenerate --prompt-upscaler-help gpt4all


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help gpt4all]


The attention prompt upscaler
-----------------------------

The ``attention`` upscaler can locate noun chunks in your prompt text and add random ``sd-embed`` or ``compel``
compatible attention values to your prompt.

This is supported for multiple languages, though, CLIP usually really only understands english.

.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the attention prompt upscaler

    dgenerate --prompt-upscaler-help attention


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help attention]


Basic prompt upscaling example
------------------------------

The following is an example making use of the ``dynamicprompts``, ``magicprompt``, and ``attention`` prompt upscaler plugins.

@EXAMPLE[../../examples/prompt_upscaler/dynamic-magic-config.dgen]


Prompt upscaling with LLMs (transformers)
-----------------------------------------

Any LLM that is supported by ``transformers`` can be used to upscale prompts via the ``magicprompt`` prompt upscaler plugin.

Here is an example using `Phi-3 Mini Abliterated by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3>`_

The ``magicprompt`` plugin supports quantization backends when ``bitsandbytes`` or ``torchao`` is installed.

Quantization backend packages will be installed by dgenerate's packaging on platforms where they are supported.

@EXAMPLE[../../examples/prompt_upscaler/magic-phi3-config.dgen]


Prompt upscaling with LLMs (gpt4all)
------------------------------------

Any LLM that is supported by ``gpt4all==2.8.2`` can be used to upscale prompts via the ``gpt4all`` prompt upscaler plugin.

This plugin supports loading LLM models in ``gguf`` format and uses a native inference backend provided by ``gpt4all``
for memory efficient inference on the cpu or gpu.

Here is an example using `Phi-3 Mini Abliterated Q4 GGUF by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF>`_

@EXAMPLE[../../examples/prompt_upscaler/gpt4all-phi3-config.dgen]
