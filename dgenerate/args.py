# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import random

from diffusers.schedulers import KarrasDiffusionSchedulers

from .mediaoutput import supported_animation_writer_formats
from .pipelinewrappers import supported_model_types, have_jax_flax
from .textprocessing import oxford_comma
from .diffusionloop import is_valid_device_string, InvalidDeviceOrdinalException

if have_jax_flax():
    from diffusers.schedulers import FlaxKarrasDiffusionSchedulers

parser = argparse.ArgumentParser(
    prog='dgenerate',
    description='Stable diffusion batch image generation tool with '
                'support for video / gif / webp animation transcoding.')

parser.add_argument('model_path', action='store',
                    help='huggingface model repository, repository slug/URI, path to folder on disk, '
                         'or path to a .cpkt or .safetensors file.')


def _from_model_type(val):
    val = val.lower()
    if val not in supported_model_types():
        raise argparse.ArgumentTypeError(
            f'Must be one of: {oxford_comma(supported_model_types(), "or")}. Unknown value: {val}')
    return val


parser.add_argument('--model-type', action='store', default='torch', type=_from_model_type,
                    help=f'Use when loading different model types. '
                         f'Currently supported: {oxford_comma(supported_model_types(), "or")}. (default: torch)')

parser.add_argument('--revision', action='store', default="main",
                    help='The model revision to use, (The git branch / tag, default is "main")')

parser.add_argument('--variant', action='store', default=None,
                    help='If specified load weights from "variant" filename, e.g. "pytorch_model.<variant>.bin". '
                         'This option is ignored if using flax.')

parser.add_argument('--vae', action='store', default=None,
                    help=f'Specify a VAE. When using torch models the syntax '
                         f'is: "AutoEncoderClass;(URL or file path)". Examples: "AutoencoderKL;vae.pt", '
                         f'"AsymmetricAutoencoderKL;vae.pt", "AutoencoderTiny;vae.pt". When using a Flax model, '
                         f'there is currently only one available encoder class: "AutoencoderKL;vae.pt". '
                         f'Hugging face URI/slugs, .pt, .pth, and .safetensors files are accepted.')

parser.add_argument('--scheduler', action='store', default=None,
                    help=f'Specify a Scheduler. torch compatible schedulers: ({", ".join(e.name for e in KarrasDiffusionSchedulers)}). ' +
                         (f'flax compatible schedulers: ({", ".join(e.name for e in FlaxKarrasDiffusionSchedulers)})' if have_jax_flax() else ''))

parser.add_argument('--safety-checker', action='store_true', default=False,
                    help=f'Enable safety checker loading, this is off by default. '
                         f'When turned on images with NSFW content detected may result in solid black output. '
                         f'Some pretrained models have settings indicating a safety checker is not to be loaded, '
                         f'in that case this option has no effect.')


def _type_device(device):
    try:
        if not is_valid_device_string(device):
            raise argparse.ArgumentTypeError(f'Must be cuda or cpu. Unknown value: {device}')
    except InvalidDeviceOrdinalException as e:
        raise argparse.ArgumentTypeError(e)

    return device


parser.add_argument('-d', '--device', action='store', default='cuda', type=_type_device,
                    help='cuda / cpu. (default: cuda). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific GPU.')


def _type_dtype(dtype):
    dtype = dtype.lower()

    if dtype not in {'float16', 'float32', 'auto'}:
        raise argparse.ArgumentTypeError('Must be float16, float32, or auto.')
    else:
        return dtype


parser.add_argument('-t', '--dtype', action='store', default='auto', type=_type_dtype,
                    help='Model precision: float16 / float32 / auto. (default: auto)')


def _type_output_size(size):
    r = size.lower().split('x')
    if len(r) < 2:
        x, y = int(r[0]), int(r[0])
    else:
        x, y = int(r[0]), int(r[1])

    if x % 8 != 0:
        raise argparse.ArgumentTypeError('Output X dimension must be divisible by 8.')

    if y % 8 != 0:
        raise argparse.ArgumentTypeError('Output Y dimension must be divisible by 8.')

    return x, y


parser.add_argument('-s', '--output-size', action='store', default=None, type=_type_output_size,
                    help='Image output size. '
                         'If an image seed is used it will be resized to this dimension with aspect ratio '
                         'maintained, width will be fixed and a new height will be calculated. If only one integer '
                         'value is provided, that is the value for both dimensions. X/Y dimension values should '
                         'be separated by "x".  (default: 512x512 when no image seeds are specified)')

parser.add_argument('-o', '--output-path', action='store', default=os.path.join(os.getcwd(), 'output'),
                    help='Output path for generated images and files. '
                         'This directory will be created if it does not exist. (default: ./output)')


def _type_prompts(prompt):
    pn = prompt.strip().split(';')
    pl = len(pn)
    if pl == 0:
        return {'prompt': ''}
    elif pl == 1:
        return {'prompt': pn[0].rstrip()}
    elif pl == 2:
        return {'prompt': pn[0].rstrip(), 'negative_prompt': pn[1].lstrip()}
    else:
        raise argparse.ArgumentTypeError(
            f'Parse error, too many values, only a prompt and optional negative prompt are accepted')


parser.add_argument('-p', '--prompts', nargs='+', action='store',
                    default=[{'prompt': ''}],
                    type=_type_prompts,
                    help='List of prompts to try, an image group is generated for each prompt, '
                         'prompt data is split by ; (semi-colon). The first value is the positive '
                         'text influence, things you want to see. The Second value is negative '
                         'influence IE. things you don\'t want to see. '
                         'Example: --prompts "shrek flying a tesla over detroit; clouds, rain, missiles". '
                         '(default: [(empty string)])')

seed_options = parser.add_mutually_exclusive_group()

seed_options.add_argument('-se', '--seeds', nargs='+', action='store', default=None,
                          type=int,
                          help='List of seeds to try, define fixed seeds to achieve deterministic output. '
                               'This argument may not be used when --gse/--gen-seeds is used. '
                               '(default: [randint(0, 99999999999999)])')

seed_options.add_argument('-gse', '--gen-seeds', action='store', default=None, type=int,
                          help='Auto generate N random seeds to try. This argument may not '
                               'be used when -se/--seeds is used.')


def _type_animation_format(val):
    val = val.lower()
    if val not in supported_animation_writer_formats():
        raise argparse.ArgumentTypeError(
            f'Must be {oxford_comma(supported_animation_writer_formats(), "or")}. Unknown value: {val}')
    return val


parser.add_argument('-af', '--animation-format', action='store', default='mp4', type=_type_animation_format,
                    help='Output format when generating an animation from an input video / gif / webp etc. '
                         f'Value must be one of: {oxford_comma(supported_animation_writer_formats(), "or")}. '
                         f'(default: mp4)')


def _type_frame_start(val):
    val = int(val)
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('-fs', '--frame-start', action='store', default=0, type=_type_frame_start,
                    help='Starting frame slice point for animated files, the specified frame will be included.')


def _type_frame_end(val):
    val = int(val)
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('-fe', '--frame-end', action='store', default=None, type=_type_frame_end,
                    help='Ending frame slice point for animated files, the specified frame will be included.')

parser.add_argument('-is', '--image-seeds', action='store', nargs='*', default=[],
                    help='List of image seeds to try when processing image seeds, these may '
                         'be URLs or file paths. Videos / GIFs / WEBP files will result in frames '
                         'being rendered as well as an animated output file being generated if more '
                         'than one frame is available in the input file. Inpainting for static images can be '
                         'achieved by specifying a black and white mask image in each image seed string using '
                         'a semicolon as the seperating character, like so: "my-seed-image.png;my-image-mask.png", '
                         'white areas of the mask indicate where generated content is to be placed in your seed '
                         'image. Output dimensions specific to the image seed can be specified by placing the '
                         'dimension at the end of the string following a semicolon like so: '
                         '"my-seed-image.png;512x512" or "my-seed-image.png;my-image-mask.png;512x512". '
                         'Inpainting masks can be downloaded for you from a URL or be a path to a file on disk.')


def _type_image_seed_strengths(val):
    val = float(val)
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


parser.add_argument('-iss', '--image-seed-strengths', action='store', nargs='*', default=[0.8],
                    type=_type_image_seed_strengths,
                    help='List of image seed strengths to try. Closer to 0 means high usage of the seed image '
                         '(less noise convolution), 1 effectively means no usage (high noise convolution). '
                         'Low values will produce something closer or more relevant to the input image, high '
                         'values will give the AI more creative freedom. (default: [0.8])')


def _type_guidance_scale(val):
    val = float(val)
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('-gs', '--guidance-scales', action='store', nargs='*', default=[5], type=_type_guidance_scale,
                    help='List of guidance scales to try. Guidance scale effects how much your '
                         'text prompt is considered. Low values draw more data from images unrelated '
                         'to text prompt. (default: [5])'
                    )


def _type_inference_steps(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


parser.add_argument('-ifs', '--inference-steps', action='store', nargs='*', default=[30], type=_type_inference_steps,
                    help='Lists of inference steps values to try. The amount of inference (denoising) steps '
                         'effects image clarity to a degree, higher values bring the image closer to what '
                         'the AI is targeting for the content of the image. Values between 30-40 '
                         'produce good results, higher values may improve image quality and or '
                         'change image content. (default: [30])')


def parse_args(args=None, namespace=None):
    args = parser.parse_args(args, namespace)

    if args.gen_seeds is not None:
        args.seeds = [random.randint(0, 99999999999999) for i in range(0, int(args.gen_seeds))]
    elif args.seeds is None:
        args.seeds = [random.randint(0, 99999999999999)]

    if args.output_size is None and len(args.image_seeds) == 0:
        args.output_size = (512, 512)

    return args
