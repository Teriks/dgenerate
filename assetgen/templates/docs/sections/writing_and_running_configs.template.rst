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
and T2I Adapter models are cached individually.

All user specifiable model objects can be reused by diffusion pipelines in certain
situations and this is taken advantage of by using an in memory cache of these objects.

In effect, the creation of a diffusion pipeline is memoized, as well as the creation of
any pipeline subcomponents when you have specified them explicitly with a URI.

A number of things effect cache hit or miss upon a dgenerate invocation, extensive information
regarding runtime caching behavior of a pipelines and other models can be observed using ``-v/--verbose``

When loading multiple different models be aware that they will all be retained in CPU side memory for
the duration of program execution, unless all models are flushed using the ``\clear_object_cache``
directive.

You can also specify which caches you want to flush individually with ``\clear_object_cache NAME1 NAME2``,
for instance ``\clear_object_cache unet`` clears all cached ``unet`` objects.

See: ``\list_object_caches`` for a list of object cache names.

To clear any models cached in VRAM, use ``\clear_device_cache DEVICE_NAME1 DEVICE_NAME2``, where ``DEVICE_NAME``
is the name of a torch device, i.g: ``cuda:0``, ``cuda:1`` etc.

dgenerate uses heuristics to clear the in memory cache automatically when needed, including a size estimation
of models before they enter system memory, however by default it will use system memory very aggressively
and it is not entirely impossible to run your system out of memory if you are not careful.

Basic config syntax
-------------------

The basic idea of the dgenerate config syntax is that it is a pseudo Unix shell mixed with Jinja2 templating.

The config language provides many niceties for batch processing large amounts of images
and image output in a Unix shell like environment with Jinja2 control constructs.

Shell builtins, known as directives, are prefixed with ``\``, for example: ``\print``

Environmental variables not inside of jinja templates will be expanded in config scripts using both Unix and Windows CMD syntax.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # these all expand from your system environment
    # if the variable is not set, they expand to nothing

    \print $VARIABLE
    \print ${VARIABLE}
    \print %VARIABLE%


To expand environmental variables inside of a jinja template construct, use the special ``env`` namespace.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # this expands from your system environment
    # if the variable is not set, it expands to
    # nothing

    \print {{ env.VARIABLE }}

    {% for i in range(0, 10) %}
        \print {{ env.VARIABLE }}
    {% endfor %}


Empty lines and comments starting with ``#`` will be ignored, comments that occur at the end of lines will also be ignored.

You can create a multiline continuation using ``\`` to indicate that a line continues similar to bash.

Unlike bash, if the next line starts with ``-`` it is considered part of a continuation as well
even if ``\`` had not been used previously. This allows you to list out many POSIX style shell
options starting with ``-`` without having to end every line with ``\``.

Comments can be interspersed with invocation or directive arguments
on their own line with the use of ``\`` on the last line before
comments and whitespace begin. This can be used to add documentation
above individual arguments instead of at the tail end of them.

The following is a config file example that covers the most basic syntax concepts.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/basic-syntax-config.dgen]


Built in template variables
---------------------------

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

To receive information about Jinja2 template variables that are set after a dgenerate invocation.
You can use the ``\templates_help`` directive which is similar to the ``--templates-help`` option
except it will print out all the template variables assigned values instead of just their
names and types. This is useful for figuring out the values of template variables set after
a dgenerate invocation in a config file for debugging purposes. You can specify one or
more template variable names as arguments to ``\templates_help`` to receive help for only
the mentioned variable names.

Template variables set with the ``\set``, ``\setp``, and ``\sete`` directive will
also be mentioned in this output.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/templates-help-directive-config.dgen]

The ``\templates_help`` output from the above example is:

@COMMAND_OUTPUT[{
  "command": "dgenerate --file @PROJECT_DIR/examples/config_syntax/templates-help-directive-config.dgen --inference-steps 1 --device cpu --output-size 8 --output-path @PROJECT_DIR/examples/config_syntax/output",
  "replace": [
          ["(?s).*?Config", "Config"],
          [" from.*?>", ">"],
          ["'--inference-steps', '1'.*?]", "]"],
          [" object at.*?>", " object>"],
          ["Value: '[^']*examples/config_syntax/output'", "Value: 'output'"],
          ["Value: \\[1\\]", "Value: [30]"],
          ["Value: \\(8, 8\\)", "Value: None"],
          ["Value: 'cpu'", {"value":"Value: None", "count":1}],
          ["Value: 'cpu'", {"value":"Value: 'cuda'", "count":1}]
      ],
  "reflags": ["multiline", "dotall"]
}]



Built in template functions
---------------------------

The option ``--functions-help`` and the directive ``\functions_help`` can be used to print
documentation for template functions. When the option or directive is used alone all built
in functions will be printed with their signature, specifying function names as arguments
will print documentation for those specific functions.

Functions with arguments can be used as either a function or filter IE: ``{{ "quote_me" | quote }}``

The dgenerate specific jinja2 functions/filters are:

@COMMAND_OUTPUT[python ../../../scripts/get_dgenerate_shell_functions.py]

In addition to the dgenerate specific jinja2 functions, some python builtins are available:

@COMMAND_OUTPUT[python ../../../scripts/get_builtin_shell_functions.py]


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

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/directives-templating-config.dgen]

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

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/set-directive-config.dgen]

The ``\sete`` directive can be used to assign the result of shell parsing and expansion to a
template variable, the value provided will be shell parsed into tokens as if it were a line of
dgenerate config. This is useful because you can use the config languages built in shell globbing
feature to assign template variables.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/sete-directive-config.dgen]

The ``\setp`` directive can be used to assign the result of evaluating a limited subset of python
expressions to a template variable.  This can be used to set a template variable to the result
of a mathematical expression, python literal value such as a list, dictionary, set, etc...
python comprehension, or python ternary statement.  In addition, all template functions
implemented by dgenerate are available for use in the evaluated expressions.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/setp-directive-config.dgen]

Setting environmental variables, in depth
-----------------------------------------

The directives ``\env`` and ``\unset_env`` can be used to
manipulate multiple environmental variables at once.

The directive ``\env`` can also be used without arguments to print out
the values of all environment variables that exist in your environment
for debugging purposes.

Indirect expansion is allowed just like with ``\set``, ``\sete``, and ``\setp``.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/env-directive-config.dgen]

Globbing and path manipulation
------------------------------

The entirety of pythons builtin ``glob`` and ``os.path`` module are also accessible during templating, you
can glob directories using functions from the glob module, you can also glob directory's using shell
globbing.

The glob modules is set to the ``glob`` template variable, and the ``os`` module is set to the
``os`` template variable, giving you access to ``os.path`` among other things.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/globbing-config.dgen]


Importing arbitrary python modules
----------------------------------

You can use the ``\import`` function to import arbitrary python modules, this supports
the ``as`` syntax as well.

In addition ``import_module`` function can be used with ``\setp`` to import the module
as well, and can also be directly used inside a template.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # Python style import with alias

    \import numpy as np

    \setp arr np.array([1, 2, 3, 4, 5]) * 2

    \print {{ arr }}


    # Set the imported module to the variable "torch"
    # Using the import_module function

    \setp torch import_module('torch')

    # Call a module function and print the result

    \print {{ torch.cuda.is_available() }}

    # With import_module, you can also do the import
    # directly in a template expression if you want

    \print {{ import_module('torch').cuda.is_available() }}

You can use this to calculate and scale linear Flux sigmas for instance.

@EXAMPLE[@PROJECT_DIR/examples/flux/sigmas/sigmas-manual-config.dgen]

Or try scaling exponential SDXL sigmas.

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion_xl/sigmas/sigmas-manual-config.dgen]

String and text escaping behavior
---------------------------------

The shell language implements unique string and text token escaping behaviors
that are tailored around the need to handle parseable URI arguments, natural
language inputs such as prompts, and URLs.

These behaviors are designed so that they do not get in the way
as much as possible when declaring prompts and URI values.

The shell parsing is not POSIX, string handling is somewhat
comparable to python for standalone string values in terms
of quote escaping.

Most if not all behaviors are covered in the example below.

@EXAMPLE[@PROJECT_DIR/examples/config_syntax/token-escaping-config.dgen]


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

The ``\echo`` directive performs shell expansion into tokens before printing, like ``\sete``,
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

The scripts / bin directory of dgenerate's python environment is prioritized in the PATH
for processes launched by the ``\exec`` directive only, so ``python`` can be used to run python scripts
using the environment dgenerate is installed into. You can also use tools such as
``accelerate``. This PATH modification only applies to ``\exec`` commands, not globally.

This can be used to leverage dgenerate's python environment for tasks such as LoRA training
as described in the section `Utilizing the Python Environment for Training`_


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # run dgenerate as a subprocess, read a config
    # and send stdout and stderr to a file

    \exec dgenerate < my_config.dgen &> log.txt

    # use python to access dgenerate's Python environment and libraries

    \exec python -c "import torch; print(torch.__version__)"

    \exec python my_script.py

    \exec accelerate launch my_training_script.py

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

Files can either be inserted into dgenerate's web cache or
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
of the argument specification of every dgenerate invocation.

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


Setting template variables from the CLI
----------------------------------------

When using the ``--file`` option or file redirection to run configuration files, you can set template
variables from the command line using the ``--set`` and ``--setp`` meta arguments.
These mirror the functionality of the ``\set`` and ``\setp`` config directives
respectively, but allow you to set variables before the configuration file is executed.

Both meta arguments use the syntax ``variable=value`` and support two usage patterns:

Multiple values per argument:

.. code-block:: bash

    #! /usr/bin/env bash

    # Set multiple template variables with --set
    dgenerate --set model=stabilityai/stable-diffusion-xl-base-1.0 prompt="a red sports car" \
              --file my-config.dgen

    # Set multiple variables using Python expressions with --setp  
    dgenerate --setp "steps=20*2" "size=[512,512]" \
              --file my-config.dgen

Multiple argument invocations (enables templating):

.. code-block:: bash

    #! /usr/bin/env bash

    # Use multiple --set calls for templating within variables
    dgenerate --set base_prompt="a red sports car" \
              --set full_prompt="{{ base_prompt }} in the rain" \
              --file my-config.dgen

    # Multiple --setp calls with sequential dependency
    dgenerate --setp base_steps=20 \
              --setp final_steps="base_steps * 2" \
              --file my-config.dgen

    # Natural mixing of --set and --setp in order
    dgenerate --set base_value=10 \
              --setp calculated="base_value * 2" \
              --set final_prompt="The result is {{ calculated }}" \
              --file my-config.dgen

Mixed approaches:

.. code-block:: bash

    #! /usr/bin/env bash

    # Combine both patterns
    dgenerate --set model=stabilityai/stable-diffusion-xl-base-1.0 device=cuda \
              --set base_prompt="a car" \
              --set full_prompt="{{ base_prompt }} in the rain" \
              --setp "steps=20*2" "size=[512,512]" \
              --file my-config.dgen

    # Works with file redirection too
    dgenerate --set model=stabilityai/stable-diffusion-xl-base-1.0 \
              --set prompt="a red sports car" \
              < my-config.dgen

    # Pipe with variables
    cat my-config.dgen | dgenerate --setp "steps=20*2" --set device=cuda

The ``--set`` meta argument works exactly like the ``\set`` directive - it performs
template expansion and environmental variable expansion on both the variable name
and value, then assigns the result as a literal string value.

The ``--setp`` meta argument works exactly like the ``\setp`` directive - it performs
template expansion and environmental variable expansion on the variable name and value,
then evaluates the value as a Python expression and assigns the result to the variable.

These meta arguments are processed before any configuration content is executed,
allowing you to provide configuration-specific values from the command line.

All ``--set`` and ``--setp`` arguments are processed in the exact order they appear
on the command line, regardless of argument type. This means you can freely mix
``--set`` and ``--setp`` arguments and each will be able to reference variables
defined in earlier arguments.

Each argument can accept multiple ``variable=value`` pairs in a single invocation,
and arguments can be used multiple times for sequential variable definition and templating.

Example configuration file using CLI-set variables:

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # Set defaults for variables not provided via CLI

    {% if model is not defined %}
        \set model stabilityai/stable-diffusion-xl-base-1.0
    {% endif %}

    {% if prompt is not defined %}
        \set prompt "a beautiful landscape"
    {% endif %}

    {% if steps is not defined %}
        \setp steps 30
    {% endif %}

    {% if size is not defined %}
        \setp size [1024, 1024]
    {% endif %}

    {% if model_type is not defined %}
        \set model_type sdxl
    {% endif %}

    # Generate the image with our variables

    {{ model }}
    --model-type {{ model_type }}
    --prompts "{{ prompt }}"
    --inference-steps {{ steps }}
    --output-size {{ size[0] }}x{{ size[1] }}
    --output-path output

Example usage:

.. code-block:: bash

    # Use all defaults
    dgenerate --file config.dgen

    # Override specific variables
    dgenerate --set prompt="a red sports car" --setp steps=50 --file config.dgen

    # Provide a different model and model type
    dgenerate --set model="black-forest-labs/FLUX.1-dev" model_type=flux --file config.dgen

The ``--set`` and ``--setp`` meta arguments can only be used from the command line
or during a popen invocation of dgenerate. They work with ``--file``, file redirection,
and piped configuration input. They cannot be used within configuration scripts themselves,
similar to other meta arguments like ``--file`` and ``--shell``.