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


The translate prompt upscaler
-----------------------------

The ``translate`` upscaler can use ``argostranslate`` or `Helsinki-NLP <https://huggingface.co/Helsinki-NLP>`_ opus models via
``transformers`` to translate your prompts from one language to another locally.

All translation models require a one time download that is preformed when the ``translate`` prompt upscaler is first invoked
with specific ``input`` and ``output`` values.

The translator upscaler defaults to translating your provided ``input`` language code to english, which is useful for CLIP
based diffusion models which usually only understand english.

This can be used to translate between any language supported by ``argostranslate`` or ``Helsinki-NLP``.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the attention prompt upscaler

    dgenerate --prompt-upscaler-help translate


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help translate]


Basic prompt upscaling example
------------------------------

The following is an example making use of the ``dynamicprompts``, ``magicprompt``, and ``attention`` prompt upscaler plugins.

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/basic_llm/dynamic-magic-config.dgen]


Prompt upscaling with LLMs (transformers)
-----------------------------------------

Any LLM that is supported by ``transformers`` can be used to upscale prompts via the ``magicprompt`` prompt upscaler plugin.

Here is an example using `Phi-3 Mini Abliterated by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3>`_

The ``magicprompt`` plugin supports quantization with ``bitsandbytes`` and ``sdnq``.

Quantization backend packages will be installed by dgenerate's packaging on platforms where they are supported.

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/basic_llm/magic-phi3-config.dgen]


Prompt upscaling with LLMs (gpt4all)
------------------------------------

Any LLM that is supported by ``gpt4all==2.8.2`` can be used to upscale prompts via the ``gpt4all`` prompt upscaler plugin.

This plugin supports loading LLM models in ``gguf`` format and uses a native inference backend provided by ``gpt4all``
for memory efficient inference on the cpu or gpu.

Here is an example using `Phi-3 Mini Abliterated Q4 GGUF by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF>`_

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/basic_llm/gpt4all-phi3-config.dgen]


Customizing LLM output cleanup
------------------------------

You may want to implement custom regex based substitutions or python text processing
on the output generated by the ``magicprompt`` or ``gpt4all`` prompt upscaler plugins.

This can be accomplished using the URI argument ``cleanup-config``, which is a path to a ``.json``, ``.toml``, or ``.yaml`` file.

For example, in ``.json``, you would specify a list of text processing operations to perform on the text generated by the LLM.


.. code-block:: json

    [
      {
        "function": "cleanup.py:my_function"
      },
      {
        "pattern": "\\byes\\b",
        "substitution": "no",
        "ignore_case": true,
        "multiline": false,
        "dotall": false,
        "count": 0
      },
      {
        "pattern": "\\bthe\\b",
        "substitution": "and",
        "ignore_case": true,
        "multiline": false,
        "dotall": false,
        "count": 0
      },
      {
        "function": "cleanup.py:my_function2"
      }
    ]

These operations occur in the order that you specify, python files are loaded relative
to the directory of the config unless you specify an absolute path.

The options ``ignore_case`` / ``ignorecase``, ``multiline``, and ``dotall``` of the pattern operation are optional, and default to ``false``.

You may also optionally specify ``count``, which defaults to zero (meaning replace all).

These arguments are passed straight into pythons ``re.sub`` method, for reference.

The python function in ``cleanup.py``, would be defined as so:

.. code-block:: python

    def my_function(text: str) -> str:
        # modify the text here and return it

        return text


    def my_function2(text: str) -> str:
        # modify the text here and return it

        return text

In ``.toml``, an equivalent config would look like this:

.. code-block:: toml


    [[operations]]
    function = "cleanup.py:my_function"

    [[operations]]
    pattern = "\\byes\\b"
    substitution = "no"
    ignore_case = true
    multiline = false
    dotall = false
    count = 0

    [[operations]]
    pattern = "\\bthe\\b"
    substitution = "and"
    ignore_case = true
    multiline = false
    dotall = false
    count = 0

    [[operations]]
    function = "cleanup.py:my_function2"

And in ``.yaml``:

.. code-block:: yaml

    - function: "cleanup.py:my_function"

    - pattern: "\\byes\\b"
      substitution: "no"
      ignore_case: true
      multiline: false
      dotall: false
      count: 0

    - pattern: "\\bthe\\b"
      substitution: "and"
      ignore_case: true
      multiline: false
      dotall: false
      count: 0

    - function: "cleanup.py:my_function2"





