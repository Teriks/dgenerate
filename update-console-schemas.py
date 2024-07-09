# helper script for creating Console UI schema files
# these are used to display options that are dependent
# on the default capabilities of the dgenerate library / command

# generating a static schema for these options is much faster than
# getting the information dynamically from dgenerate from the
# Console UI code.

import json
import os

import dgenerate.imageprocessors.imageprocessorloader as _loader
import diffusers

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Schema for image processors and accepted arguments

with open('dgenerate/console/schemas/imageprocessors.json', 'w') as file:
    plugin_loader = _loader.ImageProcessorLoader()
    schema = plugin_loader.get_accepted_args_schema()

    # sort by processor name, this affects json output
    schema = dict(sorted(schema.items(), key=lambda x: x[0]))

    for plugin in schema.keys():
        schema[plugin].update({'PROCESSOR_HELP': plugin_loader.get_help(plugin, wrap_width=100)})

    json.dump(schema, file)


with open('dgenerate/console/schemas/karrasschedulers.json', 'w') as file:

    schema = dict()

    # sort by name, this affects json output
    schema['names'] = sorted(
        [e.name for e in diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers] +
        ['LCMScheduler'])

    json.dump(schema, file)
