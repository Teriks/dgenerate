import shlex

import dgenerate

args = shlex.split(
    'stabilityai/stable-diffusion-2 --prompts "an astronaut walking on the moon" --inference-steps 30 --guidance-scales 8 --output-path astronaut')

# the command runs inside the current process
return_code = dgenerate.invoke_dgenerate(args=args)

print('Return Code:', return_code)

# run the command again, this time observe render loop events

for event in dgenerate.invoke_dgenerate_events(args=args):
    print(event)
