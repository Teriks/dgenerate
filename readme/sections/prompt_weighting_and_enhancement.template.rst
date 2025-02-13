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

There are currently two prompt weighter implementations, the ``compel`` prompt weighter, and
the ``sd-embed`` prompt weighter.


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


.. code-block:: text

    compel:
        arguments:
            syntax: str = "compel"

        Implements prompt weighting syntax for Stable Diffusion 1/2 and Stable Diffusion XL using
        compel. The default syntax is "compel" which is analogous to the syntax used by InvokeAI.

        Specifying the syntax "sdwui" will translate your prompt from Stable Diffusion Web UI syntax
        into compel / InvokeAI syntax before generating the prompt embeddings.

        If you wish to use prompt syntax for weighting tokens that is similar to ComfyUI, Automatic1111,
        or CivitAI for example, use: 'compel;syntax=sdwui'

        The underlying weighting behavior for tokens is not exactly the same as other software that uses
        the more common "sdwui" syntax, so your prompt may need adjusting if you are reusing a prompt
        from those other pieces of software.

        You can read about compel here: https://github.com/damian0815/compel

        And InvokeAI here: https://github.com/invoke-ai/InvokeAI

        This prompt weighter supports the model types:

        --model-type torch
        --model-type torch-pix2pix
        --model-type torch-upscaler-x4
        --model-type torch-sdxl
        --model-type torch-sdxl-pix2pix
        --model-type torch-s-cascade

        The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
        implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored
        with a warning message.

    ====================================================================================================


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


.. code-block:: text

    sd-embed:

        Implements prompt weighting syntax for Stable Diffusion 1/2, Stable Diffusion XL, and Stable
        Diffusion 3, and Flux using sd_embed.

        sd_embed uses a Stable Diffusion Web UI compatible prompt syntax.

        See: https://github.com/xhinker/sd_embed

        @misc{sd_embed_2024,
          author       = {Shudong Zhu(Andrew Zhu)},
          title        = {Long Prompt Weighted Stable Diffusion Embedding},
          howpublished = {\url{https://github.com/xhinker/sd_embed}},
          year         = {2024},
        }

        This prompt weighter supports the model types:

        --model-type torch
        --model-type torch-pix2pix
        --model-type torch-upscaler-x4
        --model-type torch-sdxl
        --model-type torch-sdxl-pix2pix
        --model-type torch-s-cascade
        --model-type torch-sd3
        --model-type torch-flux

        The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
        implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored with
        a warning message.

        The secondary prompt option for SD3 --sd3-second-prompts is not supported by this prompt
        weighter implementation. Neither is --sd3-third-prompts. The prompts from these arguments will
        be ignored.

        The secondary prompt option for Flux --flux-second-prompts is supported by this prompt weighter.

        Flux does not support negative prompting in either prompt.

    ====================================================================================================


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