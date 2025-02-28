File Cache Control
==================

The base directory for files cached by dgenerate can be controlled with the environmental
variable ``DGENERATE_CACHE``, which defaults to: ``~/.cache/dgenerate``, on Windows this equates
to: ``%USERPROFILE%\.cache\dgenerate``.


Web Cache
---------

dgenerate will cache downloaded non hugging face models, downloaded ``--image-seeds`` files,
files downloaded by the ``\download`` directive, ``download`` template function, and downloaded
files used by image processors in the directory ``$DGENERATE_CACHE/web``

Files are cleared from the web cache automatically after an expiry time upon running dgenerate or
when downloading additional files, the default value is after 12 hours.

This can be controlled with the environmental variable ``DGENERATE_WEB_CACHE_EXPIRY_DELTA``.

The value of ``DGENERATE_WEB_CACHE_EXPIRY_DELTA`` is that of the named arguments of pythons
`datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_ class
seperated by semicolons.

For example: ``DGENERATE_WEB_CACHE_EXPIRY_DELTA="days=5;hours=6"``

Specifying ``"forever"`` or an empty string will disable cache expiration for every downloaded file.

spaCy Model Cache
-----------------

spaCy models that need to be downloaded by dgenerate for NLP tasks are stored under ``$DGENERATE_CACHE/spacy``.

These models cannot be stored in the python environments ``site-packages`` as is normal for spaCy.

spaCy relies on ``pip`` to install these models which are packaged as wheel files, and they cannot be installed
in degenerate's frozen Windows installer environment created by ``pyinstaller``.

Instead of being installed by ``pip``, the models are extracted into this directory by dgenerate and loaded
directly.


Hugging Face Cache
------------------

Files downloaded from huggingface by the diffusers/huggingface_hub library will be cached under
``~/.cache/huggingface/``, on Windows this equates to ``%USERPROFILE%\.cache\huggingface\``.

This is controlled by the environmental variable ``HF_HOME``

In order to specify that all large model files be stored in another location,
for example on another disk, simply set ``HF_HOME`` to a new path in your environment.

You can read more about environmental variables that affect huggingface libraries on this
`huggingface documentation page <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>`_.
