Negative Prompt
===============

In order to specify a negative prompt, each prompt argument is split
into two parts separated by ``;``

The prompt text occurring after ``;`` is the negative influence prompt.

To attempt to avoid rendering of a saddle on the horse being ridden, you
could for example add the negative prompt ``saddle`` or ``wearing a saddle``
or ``horse wearing a saddle`` etc.


.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate sd2-community/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse; horse wearing a saddle" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


A semicolon that belongs in the prompt text is written ``\\;``. The first
``;`` that is not escaped starts the negative prompt. The dgenerate config
shell, Bash, cmd, and PowerShell all leave ``\\;`` unchanged inside quotes,
so the spelling is the same in each. The examples below produce the positive
text ``a sign that says hello; world`` and the negative text ``blurry``.

In a ``.dgen`` file:

.. code-block:: jinja

    --prompts "a sign that says hello\; world; blurry"

In ``bash``:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate sd2-community/stable-diffusion-2-1 \
    --prompts "a sign that says hello\; world; blurry"

In ``cmd``:

.. code-block:: text

    dgenerate sd2-community/stable-diffusion-2-1 --prompts "a sign that says hello\; world; blurry"

In ``powershell``:

.. code-block:: powershell

    dgenerate sd2-community/stable-diffusion-2-1 --prompts "a sign that says hello\; world; blurry"

Quote the prompt in PowerShell. An unquoted ``;`` starts a new statement.
Do not write ``\\\\;``. Bash removes one of those backslashes and PowerShell
does not, so the two shells would not agree.

In a config template, ``prompt(pos, neg)`` escapes the semicolons for you.
``neg`` can be omitted. The result is already shell-quoted:

.. code-block:: jinja

    --prompts {{ prompt('a sign that says hello; world', 'blurry; rain') }}

``read_text(file)`` reads a whole text file as UTF-8, which is the encoding
config files and the shell use. The return value is the file's text:

.. code-block:: jinja

    \setp caption read_text('caption.txt')
    --prompts {{ prompt(caption) }}

Name ``encoding`` only when that file was saved as something else.


Multiple Prompts
================

Multiple prompts can be specified one after another in quotes in order
to generate images using multiple prompt variations.

The following command generates 10 uniquely named images using two
prompts and five random seeds ``(2x5)``

5 of them will be from the first prompt and 5 of them from the second prompt.

All using 50 inference steps, and 10 for guidance scale value.


.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate sd2-community/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" "an astronaut riding a donkey" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512