# helper script for creating Console UI schema files
# these are used to display options that are dependent
# on the default capabilities of the dgenerate library / command
# generating a static schema for these options is much faster than
# getting the information dynamically from dgenerate from the
# Console UI code.

import inspect
import itertools
import json
import os

import diffusers

import dgenerate.arguments as _arguments
import dgenerate.batchprocess as _batchprocess
import dgenerate.imageprocessors.imageprocessorloader as _imageprocessorloader
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.promptupscalers.promptupscalerloader as _promptupscalerloader
import dgenerate.promptweighters.promptweighterloader as _promptweighterloader
import dgenerate.textprocessing as _textprocessing

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Schema for image processors and accepted arguments

with open('dgenerate/console/schemas/imageprocessors.json', 'w') as file:
    plugin_loader = _imageprocessorloader.ImageProcessorLoader()
    schema = plugin_loader.get_accepted_args_schema(include_bases=True)

    # sort by processor name, this affects json output
    schema = dict(sorted(schema.items(), key=lambda x: x[0]))

    for plugin in schema.keys():
        schema[plugin].update({'PROCESSOR_HELP': plugin_loader.get_help(
            plugin, wrap_width=100, include_bases=True)})

    json.dump(schema, file)

with open('dgenerate/console/schemas/promptupscalers.json', 'w') as file:
    plugin_loader = _promptupscalerloader.PromptUpscalerLoader()
    schema = plugin_loader.get_accepted_args_schema(include_bases=True)

    # sort by processor name, this affects json output
    schema = dict(sorted(schema.items(), key=lambda x: x[0]))

    for plugin in schema.keys():
        schema[plugin].update({'PROMPT_UPSCALER_HELP': plugin_loader.get_help(
            plugin, wrap_width=100, include_bases=True)})

    json.dump(schema, file)

with open('dgenerate/console/schemas/promptweighters.json', 'w') as file:
    plugin_loader = _promptweighterloader.PromptWeighterLoader()
    schema = plugin_loader.get_accepted_args_schema(include_bases=True)

    # sort by processor name, this affects json output
    schema = dict(sorted(schema.items(), key=lambda x: x[0]))

    for plugin in schema.keys():
        schema[plugin].update({'PROMPT_WEIGHTER_HELP': plugin_loader.get_help(
            plugin, wrap_width=100, include_bases=True)})

    json.dump(schema, file)

with open('dgenerate/console/schemas/karrasschedulers.json', 'w') as file:
    scheduler_names = sorted(
        [e.name for e in diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers] +
        ['LCMScheduler', 'FlowMatchEulerDiscreteScheduler', 'DDPMWuerstchenScheduler'])

    schema = _pipelinewrapper.get_scheduler_uri_schema([getattr(diffusers, n) for n in scheduler_names])
    json.dump(schema, file)

config_runner = _batchprocess.ConfigRunner()

with open('dgenerate/console/schemas/directives.json', 'w') as file:

    schema = dict()

    for directive in config_runner.directives.keys():
        schema['\\' + directive] = config_runner.generate_directives_help([directive])

    json.dump(schema, file)

with open('dgenerate/console/schemas/functions.json', 'w') as file:

    schema = dict()

    for function in itertools.chain(config_runner.template_functions.keys(), config_runner.builtins.keys()):
        schema[function] = config_runner.generate_functions_help([function])

    json.dump(schema, file)

with open('dgenerate/console/schemas/mediaformats.json', 'w') as file:
    schema = dict()

    schema['images-in'] = sorted(_mediainput.get_supported_image_formats())

    schema['images-out'] = sorted(_mediaoutput.get_supported_static_image_formats())

    schema['videos-in'] = sorted(_mediainput.get_supported_animation_reader_formats())

    schema['videos-out'] = sorted(_mediaoutput.get_supported_animation_writer_formats())

    json.dump(schema, file)

with open('dgenerate/console/schemas/arguments.json', 'w') as file:
    schema = dict()


    def _opt_name(a):
        if len(a.option_strings) == 1:
            return a.option_strings[0]
        if len(a.option_strings) > 1:
            return a.option_strings[1]


    for action in (a for a in _arguments._actions if a.option_strings):
        schema[_opt_name(action)] = _textprocessing.wrap_paragraphs(
            inspect.cleandoc(action.help), width=100)

    json.dump(schema, file)
