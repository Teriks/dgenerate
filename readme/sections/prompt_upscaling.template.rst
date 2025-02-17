Prompt Upscaling
================

Prompt upscaler plugins can preprocess your prompt text, and or expand the number of prompts used automatically
by the use of txt2txt LLMs or other methods.

They can be specified globally with the ``--prompt-upscaler`` related arguments of dgenerate, or
per prompt by using the ``<upscaler: ...>`` embedded prompt argument.

Prompt upscalers can be chained together sequentially, simply by specifying multiple plugin URIs.

This works even when prompt upscalers that expand your original prompt into multiple prompts.

You can see which prompt upscalers dgenerate implements via: ``dgenerate --prompt-upscaler-help``
or ``\prompt_upscaler_help`` from within a config script.

Specifying: ``dgenerate --prompt-upscaler-help NAME1 NAME2`` will return help for the named upscaler plugins.

The following is an example making use of the ``dynamicprompts`` upscaler.

@EXAMPLE[../../examples/prompt_upscaler/dynamicprompts-config.dgen]