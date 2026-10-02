.. _specifying-sigmas:

Specifying Sigmas (denoising schedule)
======================================

The denoising schedule sigma values can be overridden with the options ``--sigmas`` or ``--sdxl-refiner-sigmas``

This is supported when the selected ``--scheduler`` or ``--second-model-scheduler`` supports overriding
sigma values.  Which is the case in the default scheduler for most model types.

An error will be issued if this particular operation is not supported for the model or the model and
selected scheduler.

Sigma values can be overridden by providing a CSV list of float values, or by using an expression
that acts on the existing sigmas calculated by the scheduler.

The ``--sigmas`` and ``--sdxl-refiner-sigmas`` options are combinatorial, meaning you can provide
multiple CSV lists, or multiple expressions, and each one of those will be tried in batch.

To specify a list of sigma values to try, simply use: ``--sigmas 1.0,0.8,0.6,0.4,0.2`` for example,
this CSV list is parsed as one token, so you may want to quote it depending on the situation.

To specify an expression you should use: ``--sigmas "expr: sigmas * 0.95"`` for instance,
the ``expr:`` prefix on the argument value indicates that you are using an expression.

Expressions are evaluated using ``asteval`` which is also used for some expression
parsing operations in dgenerate's shell.

In this expression environment, numpy is available through the namespace ``np`` if you
wish to use it to help with calculating a set of sigma values.

A common operation is simply scaling the sigma values. The variable ``sigmas``
in the expression environment is a numpy array, so the multiplication operator
scales the whole schedule.

For most schedulers, ``sigmas`` is the schedule from ``set_timesteps``. For
``--model-type flux``, ``flux-fill``, ``flux-kontext``, ``flux2``,
``flux2-klein-kv``, ``z-image``, ``z-image-omni``, ``qwen-image``,
``qwen-image-edit``, and ``qwen-image-layered``, ``sigmas`` is the pipeline's
default schedule: values spaced evenly from 1 down to ``1 / steps``. The model
applies its resolution-dependent shift to the expression result. Qwen-Image
layered writes that same sequence.


Here is an example of manually calculating that Flux schedule and passing it as
CSV from inside of a dgenerate config.

@EXAMPLE[@PROJECT_DIR/examples/flux/sigmas/sigmas-manual-config.dgen]

An expression scales the same curve. This is the schedule used by Flux.1,
Flux Fill, Flux Kontext, Flux.2, Z-Image, and Qwen-Image.

@EXAMPLE[@PROJECT_DIR/examples/flux/sigmas/sigmas-expression-config.dgen]

For SDXL and other models, the expression receives the scheduler's
``set_timesteps`` schedule. That is useful when the initial schedule is not a
simple line.

``--model-type ltx`` accepts the same CSV lists and ``expr:`` forms.
When the loaded scheduler has no dynamic shifting, ``sigmas`` in the expression
is the distilled 8-value table, not a schedule from ``set_timesteps``.
When the scheduler uses dynamic shifting, ``sigmas`` comes from
``set_timesteps`` using ``--inference-steps``. Passing ``--sigmas`` sets the
step count to the length of the result and uses your ``--guidance-scales``
value as written. See :ref:`video-generation`.

@EXAMPLE[@PROJECT_DIR/examples/ltx/ltx2/sigmas/sigmas-config.dgen]

