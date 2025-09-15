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
    \env CIVITAI_TOKEN=your_civitai_token_here
    
    # Performance and behavior
    \env DGENERATE_TORCH_COMPILE=0
    \env DGENERATE_OFFLINE_MODE=1
    
    # Cache expiry control
    \env DGENERATE_WEB_CACHE_EXPIRY_DELTA=days=7

The ``~/.dgenerate/`` directory and a default ``init.dgen`` file are created automatically when 
dgenerate runs for the first time. The default file contains helpful comments and examples.

If there are any errors executing the ``init.dgen`` file, warnings will be logged but dgenerate
will continue to start normally.