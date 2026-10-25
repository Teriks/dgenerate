Auth Tokens
===========

dgenerate will automatically append your CivitAI token to CivitAI API links if you set the environmental variable ``CIVIT_AI_TOKEN``

For Hugging Face hub downloads setting ``HF_TOKEN`` is sufficient if you wish to avoid using ``--auth-token`` or related ``token`` URI arguments.

``--auth-token`` also applies to single-file, blob, and resolve downloads, including the config ``download()`` function. It does not replace a saved login when the option is omitted or empty.


