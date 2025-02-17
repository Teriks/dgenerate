Writing and Running Configs
===========================

Config scripts can be read from ``stdin`` using a shell pipe or file redirection, or by
using the ``--file`` argument to specify a file to interpret.

Config scripts are processed with model caching and other optimizations, in order
to increase speed when many dgenerate invocations with different arguments are desired.

Loading the necessary libraries and bringing models into memory is quite slow, so using dgenerate
this way allows for multiple invocations using different arguments, without needing to load the
machine learning libraries and models multiple times in a row.

When a model is loaded dgenerate caches it in memory with it's creation parameters, which includes
among other things the pipeline mode (basic, img2img, inpaint), user specified UNets, VAEs, LoRAs,
Textual Inversions, and ControlNets.

If another invocation of the model occurs with creation parameters that are identical, it will be
loaded out of an in memory cache, which greatly increases the speed of the invocation.

Diffusion Pipelines, user specified UNets, VAEs, Text Encoders, Image Encoders, ControlNet,
and IP Adapter models are cached individually.

All user specifiable model objects can be reused by diffusion pipelines in certain
situations and this is taken advantage of by using an in memory cache of these objects.

In effect, the creation of a diffusion pipeline is memoized, as well as the creation of
any pipeline subcomponents when you have specified them explicitly with a URI.

A number of things effect cache hit or miss upon a dgenerate invocation, extensive information
regarding runtime caching behavior of a pipelines and other models can be observed using ``-v/--verbose``

When loading multiple different models be aware that they will all be retained in memory for
the duration of program execution, unless all models are flushed using the ``\clear_model_cache`` directive or
individually using one of:

    * ``\clear_pipeline_cache``
    * ``\clear_unet_cache``
    * ``\clear_vae_cache``
    * ``\clear_text_encoder_cache``
    * ``\clear_image_encoder_cache``
    * ``\clear_controlnet_cache``
    * ``\clear_adapter_cache``
    * ``\clear_transformer_cache``

dgenerate uses heuristics to clear the in memory cache automatically when needed, including a size estimation
of models before they enter system memory, however by default it will use system memory very aggressively
and it is not entirely impossible to run your system out of memory if you are not careful.

Basic config syntax
-------------------

The basic idea of the dgenerate config syntax is that it is a pseudo Unix shell mixed with Jinja2 templating.

The config language provides many niceties for batch processing large amounts of images
and image output in a Unix shell like environment with Jinja2 control constructs.

Shell builtins, known as directives, are prefixed with ``\``, for example: ``\print``

Environmental variables will be expanded in config scripts using both Unix and Windows CMD syntax

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # these all expand from your system environment
    # if the variable is not set, they expand to nothing

    \print $VARIABLE
    \print ${VARIABLE}
    \print %VARIABLE%

Empty lines and comments starting with ``#`` will be ignored, comments that occur at the end of lines will also be ignored.

You can create a multiline continuation using ``\`` to indicate that a line continues similar to bash.

Unlike bash, if the next line starts with ``-`` it is considered part of a continuation as well
even if ``\`` had not been used previously. This allows you to list out many Posix style shell
options starting with ``-`` without having to end every line with ``\``.

Comments can be interspersed with invocation or directive arguments
on their own line with the use of ``\`` on the last line before
comments and whitespace begin. This can be used to add documentation
above individual arguments instead of at the tail end of them.

The following is a config file example that covers the most basic syntax concepts.

@EXAMPLE[../../examples/config_syntax/basic-syntax-config.dgen]


Built in template variables and functions
-----------------------------------------

There is valuable information about the previous invocation of dgenerate that
is set in the environment and available to use via Jinja2 templating or in
the ``\setp`` directive, some of these include:

* ``{{ last_images }}`` (An iterable of un-quoted filenames which were generated)
* ``{{ last_animations }}`` (An iterable of un-quoted filenames which were generated)

There are template variables for prompts, containing the previous prompt values:

* ``{{ last_prompts }}`` (List of prompt objects with the un-quoted attributes 'positive' and 'negative')
* ``{{ last_sdxl_second_prompts }}``
* ``{{ last_second_model_prompts }}``
* ``{{ last_second_model_second_prompts }}``

Some available custom jinja2 functions/filters are:

* ``{{ first(list_of_items) }}`` (First element in a list)
* ``{{ last(list_of_items) }}`` (Last element in a list)
* ``{{ unquote('"unescape-me"') }}`` (shell unquote / split, works on strings and lists)
* ``{{ quote('escape-me') }}`` (shell quote, works on strings and lists)
* ``{{ format_prompt(prompt_object) }}`` (Format and quote one or more prompt objects with their delimiter, works on single prompts and lists)
* ``{{ format_size(size_tuple) }}`` (Format a size tuple / iterable, join with "x" character)
* ``{{ align_size('700x700', 8) }}`` (Align a size string or tuple to a specific alignment, return a formatted string by default)
* ``{{ pow2_size('700x700', 8) }}`` (Round a size string or tuple to the nearest power of 2, return a formatted string by default)
* ``{{ size_is_aligned('700x700', 8) }}`` (Check if a size string or tuple is aligned to a specific alignment, return ``True`` or ``False``)
* ``{{ size_is_pow2('700x700') }}`` (Check if a size string or tuple is a power of 2 dimension, return ``True`` or ``False``)
* ``{{ format_model_type(last_model_type) }}`` (Format a ``ModelType`` enum to a value to ``--model-type``)
* ``{{ format_dtype(last_dtype) }}`` (Format a ``DataType`` enum to a value to ``--dtype``)
* ``{{ gen_seeds(n) }}`` (Return a list of random integer seeds in the form of strings)
* ``{{ cwd() }}`` (Return the current working directory as a string)
* ``{{ download(url) }}`` (Download from a url to the web cache and return the file path)
* ``{{ have_feature(feature_name) }}`` (Check for feature and return bool, value examples: ``ncnn``)
* ``{{ platform() }}`` (Return platform.system())

The above functions which possess arguments can be used as either a function or filter IE: ``{{ "quote_me" | quote }}``

The option ``--functions-help`` and the directive ``\functions_help`` can be used to print
documentation for template functions. When the option or directive is used alone all built
in functions will be printed with their signature, specifying function names as arguments
will print documentation for those specific functions.

To receive information about Jinja2 template variables that are set after a dgenerate invocation.
You can use the ``\templates_help`` directive which is similar to the ``--templates-help`` option
except it will print out all the template variables assigned values instead of just their
names and types. This is useful for figuring out the values of template variables set after
a dgenerate invocation in a config file for debugging purposes. You can specify one or
more template variable names as arguments to ``\templates_help`` to receive help for only
the mentioned variable names.

Template variables set with the ``\set``, ``\setp``, and ``\sete`` directive will
also be mentioned in this output.

@EXAMPLE[../../examples/config_syntax/templates-help-directive-config.dgen]

The ``\templates_help`` output from the above example is:

@COMMAND_OUTPUT[{
  "command": "dgenerate --file ../../examples/config_syntax/templates-help-directive-config.dgen --inference-steps 1 --output-path ../../examples/config_syntax/output",
  "replace": {
          "(?s).*Config": "Config",
          " from.*>": ">",
          "'--inference-steps', '1'.*]": "]",
          " object at.*>": " object>",
          "\\.\\./\\.\\./examples/config_syntax/output": "output",
          "Value: [1]": "Value: [30]"
      }
}]

The following is output from ``\functions_help`` showing every implemented template function signature.

@COMMAND_OUTPUT[dgenerate --no-stdin --functions-help]

Directives, and applying templating
-----------------------------------

You can see all available config directives with the command
``dgenerate --directives-help``, providing this option with a name, or multiple
names such as: ``dgenerate --directives-help save_modules use_modules`` will print
the documentation for the specified directives. The backslash may be omitted.
This option is also available as the config directive ``\directives_help``.

Example output:

@COMMAND_OUTPUT[dgenerate --no-stdin --directives-help]

Here are examples of other available directives such as ``\set``, ``\setp``, and
``\print`` as well as some basic Jinja2 templating usage. This example also covers
the usage and purpose of ``\save_modules`` for saving and reusing pipeline modules
such as VAEs etc. outside of relying on the caching system.

@EXAMPLE[../../examples/config_syntax/directives-templating-config.dgen]

Setting template variables, in depth
------------------------------------

The directives ``\set``, ``\sete``, and ``\setp`` can be used to set the value
of template variables within a configuration.  The directive ``\unset`` can be
used to undefine template variables.

All three of the assignment directives have unique behavior.

The ``\set`` directive sets a value with templating and environmental variable expansion applied to it,
and nothing else aside from the value being striped of leading and trailing whitespace. The value that is
set to the template variables is essentially the text that you supply as the value, as is. Or the text that
the templates or environment variables in the value expand to, unmodified or parsed in any way.

This is for assigning literal text values to a template variable.

@EXAMPLE[../../examples/config_syntax/set-directive-config.dgen]

The ``\sete`` directive can be used to assign the result of shell parsing and expansion to a
template variable, the value provided will be shell parsed into tokens as if it were a line of
dgenerate config. This is useful because you can use the config languages built in shell globbing
feature to assign template variables.

@EXAMPLE[../../examples/config_syntax/sete-directive-config.dgen]

The ``\setp`` directive can be used to assign the result of evaluating a limited subset of python
expressions to a template variable.  This can be used to set a template variable to the result
of a mathematical expression, python literal value such as a list, dictionary, set, etc...
python comprehension, or python ternary statement.  In addition, all template functions
implemented by dgenerate are available for use in the evaluated expressions.

@EXAMPLE[../../examples/config_syntax/setp-directive-config.dgen]

Setting environmental variables, in depth
-----------------------------------------

The directives ``\env`` and ``\unset_env`` can be used to
manipulate multiple environmental variables at once.

The directive ``\env`` can also be used without arguments to print out
the values of all environment variables that exist in your environment
for debugging purposes.

Indirect expansion is allowed just like with ``\set``, ``\sete``, and ``\setp``.

@EXAMPLE[../../examples/config_syntax/env-directive-config.dgen]

Globbing and path manipulation
------------------------------

The entirety of pythons builtin ``glob`` and ``os.path`` module are also accessible during templating, you
can glob directories using functions from the glob module, you can also glob directory's using shell
globbing.

@EXAMPLE[../../examples/config_syntax/globbing-config.dgen]

The \\print and \\echo directive
--------------------------------

The ``\print`` and ``\echo`` directive can both be used to output text to the console.

The difference between the two directives is that ``\print`` only ever prints
the raw value with templating and environmental variable expansion applied,
similar to the behavior of ``\set``

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # the text after \print(space) will be printed verbatim

    \print I am a raw value, I have no ability to * glob

    # Print the PATH environmental variable

    \set header Path Elements:

    \print {{ header }} $PATH
    \print {{ header }} ${PATH}
    \print {{ header }} %PATH%

The ``\echo`` directive preforms shell expansion into tokens before printing, like ``\sete``,
This can be useful for debugging / displaying the results of a shell expansion.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # lets pretend "directory" is full of files

    # this prints: directory/file1 directory/file2 ...

    \echo directory/*

    # Templates and environmental variables are expanded

    # this prints: Files: directory/file1 directory/file2 ...

    \set header Files:

    \echo {{ header }} directory/*


The \\image_process directive
-----------------------------

The dgenerate sub-command ``image-process`` has a config directive implementation.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # print the help message of --sub-command image-process, this does
    # not cause the config to exit

    \image_process --help

    \set myfiles {{ quote(glob.glob('my_images/*.png')) }}

    # this will create the directory "upscaled"
    # the files will be named "upscaled/FILENAME_processed_1.png" "upscaled/FILENAME_processed_2.png" ...

    \image_process {{ myfiles }} \
    --output upscaled/
    --processors upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth


    # the last_images template variable will be set, last_animations is also usable if
    # animations were written. In the case that you have generated an animated output with frame
    # output enabled, this will contain paths to the frames

    \print {{ quote(last_images) }}

The \\exec directive
--------------------

The ``\exec`` directive can be used to run native system commands and supports bash
pipe and file redirection syntax in a platform independent manner. All file
redirection operators supported by bash are supported. This can be useful
for running other image processing utilities as subprocesses from within a
config script.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # run dgenerate as a subprocess, read a config
    # and send stdout and stderr to a file

    \exec dgenerate < my_config.dgen &> log.txt

    # chaining processes together with pipes is supported
    # this example emulates 'cat' on Windows using cmd

    \exec cmd /c "type my_config.dgen" | dgenerate &> log.txt

    # on a Unix platform you could simply use cat

    \exec cat my_config.dgen | dgenerate &> log.txt


The \\download directive
------------------------

Arbitrary files can be downloaded via the ``\download`` directive.

This directive can be used to download a file and assign its
downloaded path to a template variable.

Files can either be inserted into dgenerates web cache or
downloaded to a specific directory or absolute path.

This directive is designed with using cached files in mind,
so it will reuse existing files by default when downloading
to an explicit path.

See the directives help output for more details: ``\download --help``

If you plan to download many large models to the web cache in
this manner you may wish to adjust the global cache expiry time
so that they exist in the cache longer than the default of 12 hours.

You can see how to do this in the section `File Cache Control`_

This directive is primarily intended to download models and or other
binary file formats such as images and will raise an error if it encounters
a text mimetype. This  behavior can be overridden with the ``-t/--text`` argument.

Be weary that if you have a long-running loop in your config using
a top level jinja template, which refers to your template variable,
cache expiry may invalidate the file stored in your variable.

You can rectify this by putting the download directive inside of
your processing loop so that the file is simply re-downloaded if
it expires in the cache.

Or you may be better off using the ``download``
template function which provides this functionality
as a template function. See: `The download() template function`_


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # download a model into the web cache,
    # assign its path to the variable "path"

    \download path https://modelhost.com/somemodel.safetensors

    # download to the models folder in the current directory
    # the models folder will be created if it does not exist
    # if somemodel.safetensors already exists it will be reused
    # instead of being downloaded again

    \download path https://modelhost.com/somemodel.safetensors -o models/somemodel.safetensors

    # download into the folder without specifying a name
    # the name will be derived from the URL or content disposition
    # header from the http request, if you are not careful you may
    # end up with a file named in a way you were not expecting.
    # only use this if you know how the service you are downloading
    # from behaves in this regard

    \download path https://modelhost.com/somemodel.safetensors -o models/


    # download a model into the web cache an overwrite any cached model using -x

    \download path https://modelhost.com/somemodel.safetensors -x

    # Download to an explicit path without any cached file reuse
    # using the -x/--overwrite argument. In effect, always freshly
    # download the file

    \download path https://modelhost.com/somemodel.safetensors -o models/somemodel.safetensors -x

    \download path https://modelhost.com/somemodel.safetensors -o models/ -x


The download() template function
--------------------------------

The template function ``download`` is analogous to the ``\download`` directive

And can be used to download a file with the same behaviour and return its
path as a string, this may be easier to use inside of certain jinja flow
control constructs.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    \set my_variable {{ download('https://modelhost.com/model.safetensors') }}

    \set my_variable {{ download('https://modelhost.com/model.safetensors', output='model.safetensors') }}

    \set my_variable {{ download('https://modelhost.com/model.safetensors', output='directory/') }}

    # you can also use any template function with \setp (python expression evaluation)

    \setp my_variable download('https://modelhost.com/model.safetensors')


The signature for this template function is: ``download(url: str, output: str | None = None, overwrite: bool = False, text: bool = False) -> str``


The \\exit directive
--------------------

You can exit a config early if need be using the ``\exit`` directive

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # exit the process with return code 0, which indicates success

    \print "success"
    \exit


An explicit return code can be provided as well


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # exit the process with return code 1, which indicates an error

    \print "some error occurred"
    \exit 1


Running configs from the command line
-------------------------------------

To utilize configuration files use the ``--file`` option,
or pipe them into the command, or use file redirection:


Use the ``--file`` option

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate --file my-config.dgen


Piping or redirection in Bash:

.. code-block:: bash

    #!/usr/bin/env bash

    # Pipe
    cat my-config.dgen | dgenerate

    # Redirection
    dgenerate < my-config.dgen


Redirection in Windows CMD:

.. code-block:: bash

    dgenerate < my-config.dgen


Piping Windows Powershell:

.. code-block:: powershell

    Get-Content my-config.dgen | dgenerate


Config argument injection
-------------------------

You can inject arguments into every dgenerate invocation of a batch processing
configuration by simply specifying them. The arguments will added to the end
of the argument specification of every call.

.. code-block:: bash

    #!/usr/bin/env bash

    # Pipe
    cat my-animations-config.dgen | dgenerate --frame-start 0 --frame-end 10

    # Redirection
    dgenerate --frame-start 0 --frame-end 10 < my-animations-config.dgen


On Windows CMD:

.. code-block:: bash

    dgenerate  --frame-start 0 --frame-end 10 < my-animations-config.dgen


On Windows Powershell:

.. code-block:: powershell

    Get-Content my-animations-config.dgen | dgenerate --frame-start 0 --frame-end 10


If you need arguments injected from the command line within the config for
some other purpose such as for using with the ``\image_process`` directive
which does not automatically recieve injected arguments, use the
``injected_args``  and related ``injected_*`` template variables.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # all injected args

    \print {{ quote(injected_args) }}

    # just the injected device

    \print {{ '--device '+injected_device if injected_device else '' }}

    # was -v/--verbose injected?

    \print {{ '-v' if injected_verbose else '' }}

    # plugin module paths injected with --plugin-modules

    \print {{ quote(injected_plugin_modules) if injected_plugin_modules else '' }}

