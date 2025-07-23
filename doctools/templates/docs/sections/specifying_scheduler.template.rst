Specifying a Scheduler (sampler)
================================

A scheduler (otherwise known as a sampler) for the main model can be selected via the use of ``--scheduler``.

And in the case of SDXL and Stable Cascade the refiner / decoder scheduler can be
selected independently with ``--second-model-scheduler``.

Both of these default to the value of ``--scheduler``, which in turn defaults to automatic selection.

Available schedulers for a specific combination of dgenerate arguments can be
queried using ``--scheduler help`` or ``--second-model-scheduler help``.

In order to use the query feature it is ideal that you provide all the other arguments
that you plan on using while making the query, as different combinations of arguments
will result in different underlying pipeline implementations being created, each of which
may have different compatible scheduler names listed. The model needs to be loaded in order to
gather this information.

For example there is only one compatible scheduler for this upscaler configuration:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/sd-x2-latent-upscaler --variant fp16 --dtype float16 \
    --model-type upscaler-x2 \
    --prompts "none" \
    --image-seeds my-image.png \
    --output-size 256 \
    --scheduler help

    # Outputs:
    #
    # Compatible schedulers for "stabilityai/sd-x2-latent-upscaler" are:
    #
    #    "EulerDiscreteScheduler"

Typically however, there will be many compatible schedulers:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler help

    # Outputs:
    #
    # Compatible schedulers for "stabilityai/stable-diffusion-2" are:
    #
    #     "DDIMScheduler"
    #     "DDPMScheduler"
    #     "DEISMultistepScheduler"
    #     "DPMSolverMultistepScheduler"
    #     "DPMSolverSDEScheduler"
    #     "DPMSolverSinglestepScheduler"
    #     "EDMEulerScheduler"
    #     "EulerAncestralDiscreteScheduler"
    #     "EulerDiscreteScheduler"
    #     "HeunDiscreteScheduler"
    #     "KDPM2AncestralDiscreteScheduler"
    #     "KDPM2DiscreteScheduler"
    #     "LCMScheduler"
    #     "LMSDiscreteScheduler"
    #     "PNDMScheduler"
    #     "UniPCMultistepScheduler"


Passing ``helpargs`` to a ``--scheduler`` related option will reveal configuration arguments that
can be overridden via a URI syntax, for every possible scheduler.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler helpargs


    # Outputs (shortened for brevity...):
    #
    # Compatible schedulers for "stabilityai/stable-diffusion-2" are:
    #    ...
    #
    #    PNDMScheduler:
    #            clone-config: True
    #            num-train-timesteps: int = 1000
    #            beta-start: float = 0.0001
    #            beta-end: float = 0.02
    #            beta-schedule: str = linear
    #            trained-betas: list | None = None
    #            skip-prk-steps: bool = False
    #            set-alpha-to-one: bool = False
    #            prediction-type: str = epsilon
    #            timestep-spacing: str = leading
    #            steps-offset: int = 0
    #   ...


As an example, you may override the mentioned arguments for any scheduler in this manner:

.. code-block:: bash

    #!/usr/bin/env bash

    # Change prediction type of the scheduler to "v_prediction".
    # for some models this may be necessary, not for this model
    # this is just a syntax example

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler "PNDMScheduler;prediction-type=v_prediction"

In the case of list / array arguments such as ``trained-betas`` you may use python
literal syntax, i.e: ``[1, 2, 3]`` or CSV (tuple) ``1,2,3``.

Take note that the default values displayed by ``helpargs`` may not be how the scheduler
gets configured internally unless you manually specify scheduler config argument
with said value, this is due to the default behavior of cloning the models original
scheduler configuration into the alternate scheduler that you have specified.

You may notice that every scheduler possesses the argument ``clone-config`` with a default
value of ``True``.  This indicates that the schedulers config will be cloned from the
scheduler config that the model was originally loaded with.

Usually a diffusion model will be loaded with a pre-configured scheduler that is appropriate
for the way it was trained. And when you specify an alternate scheduler, the original configuration or
parts of it that are applicable to the alternate scheduler, are cloned into the new
schedulers config.

This allows you to have a somewhat sane configuration for the alternate scheduler without
specifying many argument overrides.

If you would rather this not occur, and to manually configure the scheduler without the interference
of the values from the original scheduler configuration, or to just use it with the default values
that are presented by ``helpargs``, you can set ``clone-config`` to ``False`` and the config cloning
behavior will be disabled.

Setting ``clone-config`` to ``False`` results in the new scheduler being initialized entirely
with the default argument values that are presented by ``helpargs``, you can then specify
overrides as needed.

These scheduler arguments and default values may also be easily viewed in the `Console UI`_ from the
``Edit -> Insert URI -> Karras Scheduler URI`` dialog, or the recipes form scheduler selection field.

Like diffusion parameter arguments, you may specify multiple scheduler URIs and they will be tried in turn,
this allows you to iterate over alternate schedulers, to produce variations that use different schedulers.

When you specify multiple schedulers in this manner they will be added to the beginning of the
output file name, in the order: ``(scheduler)_(refiner / decoder scheduler)``

.. code-block:: bash

    #!/usr/bin/env bash

    # Try these two schedulers one after another

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --schedulers EulerAncestralDiscreteScheduler KDPM2AncestralDiscreteScheduler \
    --output-size 512x512 \
    --prompts "a horse standing in a field"


    # This works for all scheduler arguments, for instance
    # this SDXL command results in 4 generation steps, trying
    # all possible combinations of schedulers provided

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --schedulers EulerAncestralDiscreteScheduler EulerDiscreteScheduler \
    --second-model-schedulers KDPM2AncestralDiscreteScheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a horse standing in a field"
