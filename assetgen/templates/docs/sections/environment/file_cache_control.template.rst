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


Checkpoint Conversion Cache
---------------------------

In order to support quantization on single-file checkpoints from sources such as CivitAI, or
in the case of quantization with LoRAs involved, dgenerate will load and then save the checkpoint
sub-modules into diffusers format to an on disk cache before reloading them with
quantization pre-processing applied.

In the case of single file checkpoints, this is just to get the checkpoint into diffusers
format so that it can be processed by the auto quantizer, when LoRAs are involved, the
LoRAs are first merged into the applicable checkpoint sub-modules at the desired scale
values before being saved in diffusers format.

For LoRAs, this process allows for improved numerical stability, LoRAs are first
merged with the applicable checkpoint sub-modules in original precision before
the quantization pre-process occurs. If you change your LoRA scale values,
this will equate to a cache miss, and the merge and save process will be repeated
for the new scale values.

These converted module checkpoints exist in the directory ``$DGENERATE_CACHE/diffusers_converted``.

They are not removed automatically, and will remain on disk until you manually delete them
similar to the huggingface cache. If you use quantization with many different LoRAs or LoRA
scale values, this directory can grow large over time.
