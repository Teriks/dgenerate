import dgenerate
import shlex

args = shlex.split(
    'stabilityai/stable-diffusion-2 --prompts "an astronaut walking on the moon" --inference-steps 30 --guidance-scales 8 --output-path astronaut')


# the command runs inside the current process
return_code = dgenerate.invoke_dgenerate(args=args)


exit(return_code)