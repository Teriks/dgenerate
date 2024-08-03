import sys

import dgenerate.batchprocess

# it is fairly straight forward to run a dgenerate configuration from either a string or a file


config = r"""
#! dgenerate 4.0.0

stabilityai/stable-diffusion-2 --prompts "a man walking on the moon without a space suit"

# Print all set template variables

\templates_help
"""

# inject any provided user arguments into the configuration, such as -v/--verbose or
# -d/--device for instance

# using -v/--verbose with a config running enables debugging output globally
# while the config is running.
runner = dgenerate.batchprocess.ConfigRunner(injected_args=sys.argv[1:])

# Run the config from the string above
runner.run_string(config)

# the example input file is not named 'config.dgen' because the example runner
# would pick it up and run it if it were named that :)

with open('example-input.dgen', mode='rt') as input_file:
    # run file wants a TextIO object so you need to use mode='t'
    runner.run_file(input_file)
