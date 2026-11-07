Startup Configuration
=====================

dgenerate supports automatic execution of a startup configuration file located at ``~/.dgenerate/init.dgen``
(on Windows: ``%USERPROFILE%\.dgenerate\init.dgen``).

This file is executed every time dgenerate starts up, before processing any user commands. It can be used to:

* Set environment variables using the ``\env`` directive 
* Import plugins with ``\import_plugins``
* Set up any other initialization logic

Example ``~/.dgenerate/init.dgen`` for setting environment variables:

.. code-block:: jinja

    # Cache directories
    \env DGENERATE_CACHE=/path/to/my/cache
    \env HF_HOME=/path/to/hf/cache
    
    # Authentication tokens
    \env HF_TOKEN=your_huggingface_token_here
    \env CIVIT_AI_TOKEN=your_civitai_token_here
    
    # Performance and behavior
    \env DGENERATE_TORCH_COMPILE=1
    \env DGENERATE_OFFLINE_MODE=1
    
    # Cache expiry. The default keeps model files and expires every other file after 12 hours.
    # "models;forever,images;days=1,videos;days=2,.png;hours=1,other;hours=12"
    \env DGENERATE_WEB_CACHE_EXPIRY_DELTA=models;forever,other;hours=12

    # Interrupted downloads of files kept forever. One interval, not a rule list.
    \env DGENERATE_WEB_CACHE_PARTIAL_EXPIRY_DELTA=hours=12

The ``~/.dgenerate/`` directory and a default ``init.dgen`` file are created automatically when 
dgenerate runs for the first time. The default file contains helpful comments and examples.

If there are any errors executing the ``init.dgen`` file, warnings will be logged but dgenerate
will continue to start normally.