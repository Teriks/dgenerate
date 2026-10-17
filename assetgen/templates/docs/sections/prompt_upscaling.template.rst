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


The llama prompt upscaler
-------------------------

The ``llama`` upscaler runs GGUF models through llama.cpp.

The default model used is: `Phi-3 Mini Abliterated Q4 GGUF by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF>`_

``xllamacpp`` is used for the llama prompt upscaler plugin. The GPU and CPU wheels share one version, so the GPU
index has to be ``--index-url`` or pip keeps the PyPI build (CPU on Linux and Windows,
Metal on macOS). PyPI and the torch index stay on ``--extra-index-url``.
The network installer selects the index for you.

CUDA 13.0+:

.. code-block:: bash

    pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu130 --extra-index-url https://download.pytorch.org/whl/cu130/ --extra-index-url https://pypi.org/simple

CUDA 12.8 through 12.9:

.. code-block:: bash

    pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple

Linux ROCm 7.2:

.. code-block:: bash

    pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-7.2.4 --extra-index-url https://download.pytorch.org/whl/rocm7.2/ --extra-index-url https://pypi.org/simple

Older NVIDIA, Windows AMD, and Intel Arc:

.. code-block:: bash

    pip install "dgenerate[xllamacpp]" --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan --extra-index-url https://download.pytorch.org/whl/cu126/ --extra-index-url https://pypi.org/simple


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the llama prompt upscaler

    dgenerate --prompt-upscaler-help llama


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-upscaler-help llama]


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

All translation models require a one time download that is performed when the ``translate`` prompt upscaler is first invoked
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


Prompt upscaling with LLMs (llama)
-----------------------------------

Any GGUF model that llama.cpp can load can be used via the ``llama`` prompt upscaler.

The plugin uses the chat template stored in the GGUF file. ``gpu-layers`` accepts
``auto``, ``all``, or an integer, and defaults to ``auto``, which offloads as many
layers as fit in VRAM. ``all`` offloads every layer. ``0`` keeps the model on the CPU.

Here is an example using `Phi-3 Mini Abliterated Q4 GGUF by failspy <https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF>`_

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/basic_llm/xllamacpp-phi3-config.dgen]

Models that are not chat models can be used the same way. This example loads a GGUF
conversion of the original MagicPrompt weights, prepends the prompt, and uses a
1024 token context:

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/basic_llm/xllamacpp-magicprompt-config.dgen]


Writing prompts from an image
-----------------------------

``llama`` and ``magicprompt`` can write a prompt from a picture. The prompt text is the
instruction sent with the image, for example a request to describe it as an image-generation prompt.

For ``llama``, pass the picture as ``image`` and the matching GGUF multimodal projector as ``mmproj``.
Vision models need a larger ``context-tokens`` value than text models, because the image occupies
part of the context. Animated files use the first frame. ``thinking`` defaults to false so the
reply is the prompt. Set it to true on models that support a thought before the answer.

This example uses `heretic MiniCPM-V 4.6 GGUF <https://huggingface.co/mradermacher/MiniCPM-V-4.6-heretic-GGUF>`_
with the photo in ``examples/media/dog-on-bench.png``:

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/image_to_prompt/llama-config.dgen]

``magicprompt`` loads a vision-language model through ``transformers`` instead of a GGUF projector.
With an image, ``max-length`` is the maximum number of new tokens. This example uses
`heretic MiniCPM-V 4.6 <https://huggingface.co/heretic-org/MiniCPM-V-4.6-heretic>`_:

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/image_to_prompt/magicprompt-config.dgen]

``variations`` asks the model for more than one prompt from the same image:

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/image_to_prompt/llama-variations-config.dgen]

The description can be the positive prompt for a ControlNet Canny generation. This example
asks `heretic MiniCPM-V 4.6 <https://huggingface.co/heretic-org/MiniCPM-V-4.6-heretic>`_ to
describe ``examples/media/horse2.jpeg``, keeps the written negative prompt, and uses a Canny
edge map of that photo as the control image:

@EXAMPLE[@PROJECT_DIR/examples/prompt_upscaler/image_to_prompt/llama-canny-config.dgen]


Customizing LLM output cleanup
------------------------------

You may want to implement custom regex based substitutions or python text processing
on the output generated by the ``magicprompt`` or ``llama`` prompt upscaler plugins.

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





