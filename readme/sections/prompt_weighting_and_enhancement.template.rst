Prompt Weighting and Enhancement
================================

By default, the prompt token weighting syntax that you may be familiar with from other software such as
`ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_, `Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1_>`_,
and `CivitAI <CivitAI_1_>`_ etc. is not enabled, and prompts over ``77`` tokens in length are not supported.

However! dgenerate implements prompt weighting and prompt enhancements through internal plugins
called prompt weighters, which can be selectively enabled to process your prompts. They support
special token weighting syntaxes, and overcome limitations on prompt length.

The names of all prompt weighter implementations can be seen by using the argument ``--prompt-weighter-help``,
and specific documentation for a prompt weighter can be printed py passing its name to this argument.

You may also use the config directive ``\prompt_weighter_help`` inside of a config, or
more likely when you are working inside the `Console UI`_ shell.

There are currently three prompt weighter implementations, the ``compel`` prompt weighter,
the ``sd-embed`` prompt weighter, and the ``llm4gen`` prompt weighter.

Prompt weighters can be specified via ``--prompt-weighter`` or inside your prompt argument using ``<weighter: (uri here)>``
anywhere in the prompt.  A prompt weighter specified in the prompt text applies the prompt weighter to just
that prompt alone (both negative and positive prompts).

You can specify different prompt weighters for the SDXL Refiner or Stable Cascade
decoder using ``--sdxl-refiner-prompt-weighter`` and ``-s-cascade-decoder-prompt-weighter``, or in the prompt
arguments ``--sdxl-refiner-prompts`` and ``--s-cascade-decoder-prompts``.

Specifying ``<weighter: (uri here)>`` in a ``--prompts`` value will default
the secondary models to the same prompt weighter unless you specify otherwise.
Either by using ``<weighter: (uri here)>`` in their respective prompt arguments,
or in the respective global prompt-weighter arguments.


The compel prompt weighter
--------------------------

The ``compel`` prompt weighter uses the `compel <https://github.com/damian0815/compel>`_ library to
support `InvokeAI <https://github.com/invoke-ai/InvokeAI>`_ style prompt token weighting syntax for
Stable Diffusion 1/2, and Stable Diffusion XL.

You can read about InvokeAI prompt syntax here: `Invoke AI prompting documentation <https://invoke-ai.github.io/InvokeAI/features/PROMPTS/>`_

It is a bit different than `Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1_>`_ syntax,
which is a syntax used by the majority of other image generation software. It possesses some neat
features not mentioned in this documentation, that are worth reading about in the links provided above.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the compel prompt weighter

    dgenerate --prompt-weighter-help compel


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-weighter-help compel]


You can enable the ``compel`` prompt weighter by specifying it with the ``--prompt-weighter`` argument.

.. code-block:: bash

    #!/usr/bin/env bash

    # Some very simple examples

    # Increase the weight of (picking apricots)

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter compel \
    --prompts "a tall man (picking apricots)++"

    # Specify a weight

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter compel \
    --prompts "a tall man (picking apricots)1.3"


If you prefer the prompt weighting syntax used by Stable Diffusion Web UI, you can specify
the plugin argument ``syntax=sdwui`` which will translate your prompt from that syntax into
compel / InvokeAI syntax for you.


.. code-block:: bash

    #!/usr/bin/env bash

    # Some very simple examples

    # Increase the weight of (picking apricots)

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter "compel;syntax=sdwui" \
    --prompts "a tall man ((picking apricots))"

    # Specify a weight

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter "compel;syntax=sdwui" \
    --prompts "a tall man (picking apricots:1.3)"


The weighting algorithm is not entirely identical to other pieces of software, so if
you are migrating prompts they will likely require some adjustment.


The sd-embed prompt weighter
----------------------------

The ``sd-embed`` prompt weighter uses the `sd_embed <https://github.com/xhinker/sd_embed>`_ library to support
`Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1_>`_ style prompt token
weighting syntax for Stable Diffusion 1/2, Stable Diffusion XL, and Stable Diffusion 3.


The syntax that ``sd-embed`` uses is the more wide spread prompt syntax used by software such as
`Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1_>`_ and `CivitAI <CivitAI_1_>`_


Quite notably, the ``sd-embed`` prompt weighter supports Stable Diffusion 3 and Flux, where
as the ``compel`` prompt weighter currently does not.


.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the sd-embed prompt weighter

    dgenerate --prompt-weighter-help sd-embed


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-weighter-help sd-embed]


You can enable the ``sd-embed`` prompt weighter by specifying it with the ``--prompt-weighter`` argument.


.. code-block:: bash

    #!/usr/bin/env bash

    # You need a huggingface API token to run this example

    dgenerate stabilityai/stable-diffusion-3-medium-diffusers \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024x1024 \
    --model-sequential-offload \
    --prompt-weighter sd-embed \
    --auth-token $HF_TOKEN \
    --prompts "a (man:1.2) standing on the (beach:1.2) looking out in to the water during a (sunset)"


The llm4gen prompt weighter
---------------------------

The llm4gen prompt weighter is designed to enhance semantic understanding of prompts with Stable Diffusion 1.5 models specifically.

It uses a T5 RankGen encoder model and a translation model (CAM, cross adapter model) to extract a representation of a prompt using
a combination of the LLM model (T5) and the CLIP encoder model that is normally used with Stable Diffusion.

See: `LLM4GEN <https://github.com/YUHANG-Ma/LLM4GEN>`_

.. code-block:: bash

    #!/usr/bin/env bash

    # print out the documentation for the llm4gen prompt weighter

    dgenerate --prompt-weighter-help llm4gen


@COMMAND_OUTPUT[dgenerate --no-stdin --prompt-weighter-help llm4gen]