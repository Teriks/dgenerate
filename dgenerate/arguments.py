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
import sys

import diffusers.schedulers

import dgenerate
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing


class DgenerateArguments(dgenerate.DiffusionRenderLoopConfig):
    def __init__(self):
        super().__init__()

        self.plugin_module_paths = []
        self.verbose = False


parser = argparse.ArgumentParser(
    prog='dgenerate', exit_on_error=False,
    description="""Stable diffusion batch image generation tool with 
                support for video / gif / webp animation transcoding.""")

parser.add_argument('model_path', action='store',
                    help="""huggingface model repository slug, huggingface blob link to a model file, 
                            path to folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file.""")

parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help="""Output information useful for debugging, such as pipeline call and model load parameters.""")

parser.add_argument('--version', action='version', version=f"dgenerate v{dgenerate.__version__}",
                    help="Show dgenerate's version and exit")

parser.add_argument('--plugin-modules', action='store', default=[], nargs="+", dest='plugin_module_paths',
                    metavar="PATH",
                    help="""Specify one or more plugin module folder paths (folder containing __init__.py) or 
                    python .py file paths to load as plugins. Plugin modules can currently only implement 
                    image preprocessors.""")


def _from_model_type(val):
    val = val.lower()
    if val not in _pipelinewrapper.supported_model_type_strings():
        raise argparse.ArgumentTypeError(
            f'Must be one of: {_textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), "or")}. Unknown value: {val}')
    return _pipelinewrapper.get_model_type_enum(val)


def _type_dtype(dtype):
    dtype = dtype.lower()

    if dtype not in {'float16', 'float32', 'auto'}:
        raise argparse.ArgumentTypeError('Must be float16, float32, or auto.')
    else:
        return dtype


def _type_prompts(prompt):
    try:
        return _prompt.Prompt().parse(prompt)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f'Prompt parse error: {e}')


def _type_inference_steps(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_guidance_scale(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('--model-type', action='store', default='torch', type=_from_model_type,
                    help=f"""Use when loading different model types. 
                         Currently supported: {_textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), "or")}. (default: torch)""")

parser.add_argument('--revision', action='store', default="main", metavar="BRANCH",
                    help="""The model revision to use when loading from a huggingface repository,
                         (The git branch / tag, default is "main")""")

parser.add_argument('--variant', action='store', default=None,
                    help="""If specified when loading from a huggingface repository or folder, load weights
                         from "variant" filename, e.g. "pytorch_model.<variant>.safetensors".
                         Defaults to automatic selection. This option is ignored if using flax.""")

parser.add_argument('--subfolder', action='store', default=None, dest='model_subfolder',
                    help="""Main model subfolder.
                         If specified when loading from a huggingface repository or folder,
                         load weights from the specified subfolder.""")

parser.add_argument('--auth-token', action='store', default=None, metavar="TOKEN",
                    help="""Huggingface auth token.
                         Required to download restricted repositories that have access permissions
                         granted to your huggingface account.""")

parser.add_argument('--vae', action='store', default=None, metavar="MODEL_PATH", dest='vae_path',
                    help=
                    """Specify a VAE. When using torch models the syntax is: 
                    "AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)".
                    
                    Examples: "AutoencoderKL;model=vae.pt", "AsymmetricAutoencoderKL;model=huggingface/vae",
                    "AutoencoderTiny;model=huggingface/vae". 
                    
                    When using a Flax model, there is currently only one available 
                    encoder class: "FlaxAutoencoderKL;model=huggingface/vae".
                    
                    The AutoencoderKL encoder class accepts huggingface repository slugs/blob links, 
                    .pt, .pth, .bin, .ckpt, and .safetensors files. Other encoders can only accept huggingface 
                    repository slugs/blob links, or a path to a folder on disk with the model 
                    configuration and model file(s). 
                    
                    Aside from the "model" argument, there are four other optional arguments that can be specified,
                    these include "revision", "variant", "subfolder", "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    "AutoencoderKL;model=huggingface/vae;revision=main;variant=fp16;subfolder=sub_folder;dtype=float16".
                    
                    The "revision" argument specifies the model revision to use for the VAE when loading from 
                    huggingface repository or blob link, (The git branch / tag, default is "main").
                    
                    The "variant" argument specifies the VAE model variant, if "variant" is specified when loading 
                    from a huggingface repository or folder, weights will be loaded from "variant" filename, e.g. 
                    "pytorch_model.<variant>.safetensors. "variant" defaults to automatic selection and is ignored if 
                    using flax. "variant" in the case of --vae does not default to the value of --variant to prevent
                    failures during common use cases.
                    
                    The "subfolder" argument specifies the VAE model subfolder, if specified when loading from a 
                    huggingface repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the VAE model precision, it defaults to the value of -t/--dtype
                    and should be one of: float16 / float32 / auto.
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --vae "AutoencoderKL;my_vae.safetensors", or with a 
                    dtype "AutoencoderKL;my_vae.safetensors;dtype=float16", all other loading arguments are unused 
                    in this case and may produce an error message if used.
                    
                    If you wish to load a specific weight file from a huggingface repository, use the blob link
                    loading syntax: --vae "AutoencoderKL;https://huggingface.co/UserName/repository-name/blob/main/vae_model.safetensors",
                    the revision argument may be used with this syntax.
                    """)

parser.add_argument('--vae-tiling', action='store_true', default=False,
                    help="""Enable VAE tiling (torch models only). Assists in the generation of
                    large images with lower memory overhead. The VAE will split the input tensor 
                    into tiles to compute decoding and encoding in several steps. This is 
                    useful for saving a large amount of memory and to allow processing larger images. 
                    Note that if you are using --control-nets you may still run into memory 
                    issues generating large images.""")

parser.add_argument('--vae-slicing', action='store_true', default=False,
                    help="""Enable VAE slicing (torch* models only). Assists in the generation 
                    of large images with lower memory overhead. The VAE will split the input tensor
                    in slices to compute decoding in several steps. This is useful to save some memory. 
                    Note that if you are using --control-nets you may still run into memory 
                    issues generating large images.""")

parser.add_argument('--lora', '--loras', action='store', default=None, metavar="MODEL_PATH", dest='lora_paths',
                    help=
                    """Specify a LoRA model (flax not supported). This should be a
                    huggingface repository slug, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    huggingface blob links are not supported, see "subfolder" and "weight-name" below instead.
                    
                    Optional arguments can be provided after the LoRA model specification, 
                    these include: "scale", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-name=lora.safetensors".
                    
                    The "scale" argument indicates the scale factor of the LoRA.
                    
                    The "revision" argument specifies the model revision to use for the VAE when loading from 
                    huggingface repository, (The git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the VAE model subfolder, if specified when loading from a 
                    huggingface repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when 
                    loading from a huggingface repository or folder on disk. 
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --lora "my_lora.safetensors",  or with a scale "my_lora.safetensors;scale=1.0", 
                    all other loading arguments are unused in this case and may produce an error message if used.""")

parser.add_argument('--textual-inversions', nargs='+', action='store', default=None, metavar="MODEL_PATH",
                    dest='textual_inversion_paths',
                    help=
                    """Specify one or more Textual Inversion models (flax and SDXL not supported). This should be a
                    huggingface repository slug, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files. 
                    
                    huggingface blob links are not supported, see "subfolder" and "weight-name" below instead.
                    
                    Optional arguments can be provided after the Textual Inversion model specification, 
                    these include: "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    "huggingface/ti_model;revision=main;subfolder=repo_subfolder;weight-name=lora.safetensors".
                    
                    The "revision" argument specifies the model revision to use for the Textual Inversion model
                    when loading from huggingface repository, (The git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the Textual Inversion model subfolder, if specified 
                    when loading from a huggingface repository or folder, weights from the specified subfolder.
                
                    The "weight-name" argument indicates the name of the weights file to be loaded when 
                    loading from a huggingface repository or folder on disk. 
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --textual-inversions "my_ti_model.safetensors", all other loading arguments 
                    are unused in this case and may produce an error message if used.""")

parser.add_argument('--control-nets', nargs='+', action='store', default=None, metavar="MODEL_PATH",
                    dest='control_net_paths',
                    help=
                    """Specify one or more ControlNet models. This should be a
                    huggingface repository slug / blob link, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    Optional arguments can be provided after the ControlNet model specification, for torch
                    these include: "scale", "start", "end", "revision", "variant", "subfolder", and "dtype".
                    
                    For flax: "scale", "revision", "subfolder", "dtype", "from_torch" (bool)
                    
                    They can be specified as so in any order, they are not positional:cd 
                    "huggingface/controlnet;scale=1.0;start=0.0;end=1.0;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "scale" argument specifies the scaling factor applied to the ControlNet model, 
                    the default value is 1.0.
                    
                    The "start" (only for --model-type "torch*") argument specifies at what fraction of 
                    the total inference steps to begin applying the ControlNet, defaults to 0.0, IE: the very beginning.
                    
                    The "end" (only for --model-type "torch*") argument specifies at what fraction of 
                    the total inference steps to stop applying the ControlNet, defaults to 1.0, IE: the very end.
                    
                    The "revision" argument specifies the model revision to use for the ControlNet model
                    when loading from huggingface repository, (The git branch / tag, default is "main").
                    
                    The "variant" (only for --model-type "torch*") argument specifies the ControlNet 
                    model variant, if "variant" is specified when loading from a huggingface repository or folder, 
                    weights will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant" 
                    defaults to automatic selection and is ignored if  using flax. "variant" in the case of 
                    --control-nets does not default to the value of --variant to prevent failures during common use cases.
                    
                    The "subfolder" argument specifies the ControlNet model subfolder, if specified 
                    when loading from a huggingface repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the ControlNet model precision, it defaults to the value of -t/--dtype
                    and should be one of: float16 / float32 / auto.
                    
                    The "from_torch" (only for --model-type flax) this argument specifies that the ControlNet is to be 
                    loaded and converted from a huggingface repository or file that is designed for pytorch. (Defaults to false)
                
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --control-nets "my_controlnet.safetensors" or --control-nets "my_controlnet.safetensors;scale=1.0;dtype=float16", 
                    all other loading arguments aside from "scale" and "dtype" are unused in this case and may produce
                    an error message if used ("from_torch" is available when using flax).
                    
                    If you wish to load a specific weight file from a huggingface repository, use the blob link
                    loading syntax: --control-nets 
                    "https://huggingface.co/UserName/repository-name/blob/main/controlnet.safetensors",
                    the revision argument may be used with this syntax.
                    """)

_flax_scheduler_help_part = \
    f' Flax schedulers: ({", ".join(e.name for e in diffusers.schedulers.FlaxKarrasDiffusionSchedulers)})' \
        if _pipelinewrapper.have_jax_flax() else ''

parser.add_argument('--scheduler', action='store', default=None, metavar="SCHEDULER_NAME",
                    help=
                    f'Specify a scheduler (sampler) by name. Passing "help" to this argument '
                    f'will print the compatible schedulers for a model without generating any images. '
                    f'Torch schedulers: ({", ".join(e.name for e in diffusers.schedulers.KarrasDiffusionSchedulers)}).'
                    + _flax_scheduler_help_part)

parser.add_argument('--sdxl-refiner', action='store', default=None, metavar="MODEL_PATH",
                    dest='sdxl_refiner_path',
                    help="""Stable Diffusion XL (torch-sdxl) refiner model path. This should be a
                    huggingface repository slug / blob link, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files. 
                    
                    Optional arguments can be provided after the SDXL refiner model specification, 
                    these include: "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    "huggingface/refiner_model_xl;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "revision" argument specifies the model revision to use for the Textual Inversion model
                    when loading from huggingface repository, (The git branch / tag, default is "main").
                    
                    The "variant" argument specifies the SDXL refiner model variant and defaults to the value of 
                    --variant, when "variant" is specified when loading from a huggingface repository or folder,
                    weights will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors.
                    "variant" defaults to automatic selection.
                    
                    The "subfolder" argument specifies the SDXL refiner model subfolder, if specified 
                    when loading from a huggingface repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the SDXL refiner model precision, it defaults to the value of -t/--dtype
                    and should be one of: float16 / float32 / auto.
                
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --sdxl-refiner "my_sdxl_refiner.safetensors" or --sdxl-refiner "my_sdxl_refiner.safetensors;dtype=float16", 
                    all other loading arguments aside from "dtype" are unused in this case and may produce
                    an error message if used.
                    
                    If you wish to load a specific weight file from a huggingface repository, use the blob link
                    loading syntax: --sdxl-refiner 
                    "https://huggingface.co/UserName/repository-name/blob/main/refiner_model.safetensors",
                    the revision argument may be used with this syntax.
                    """)

parser.add_argument('--sdxl-refiner-scheduler', action='store', default=None, metavar="SCHEDULER_NAME",
                    help='Specify a scheduler (sampler) by name for the SDXL refiner pass. Operates the exact'
                         'same way as --scheduler including the "help" option. Defaults to the value of --scheduler.')


def _type_micro_conditioning_size(size):
    if size is None:
        return None

    try:
        r = size.lower().split('x')
        if len(r) < 2:
            return int(r[0]), int(r[0])
        else:
            return int(r[0]), int(r[1])
    except ValueError:
        raise argparse.ArgumentTypeError('Dimensions must be integer values.')


def _type_image_coordinate(coord):
    if coord is None:
        return (0, 0)

    r = coord.split(',')

    try:
        return int(r[0]), int(r[1])
    except ValueError:
        raise argparse.ArgumentTypeError('Coordinates must be integer values.')


# SDXL Main pipeline


parser.add_argument('--sdxl-second-prompts', nargs='+', action='store', metavar="PROMPT",
                    default=None,
                    type=_type_prompts,
                    help="""List of secondary prompts to try using SDXL's secondary text encoder. 
                    By default the model is passed the primary prompt for this value, this option
                    allows you to choose a different prompt. The negative prompt component can be
                    specified with the same syntax as --prompts""")

parser.add_argument('--sdxl-aesthetic-scores', metavar="FLOAT",
                    action='store', nargs='+', default=[], type=float,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "aesthetic-score" micro-conditioning parameters.
                            Used to simulate an aesthetic score of the generated image by influencing the positive text
                            condition. Part of SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952].""")

parser.add_argument('--sdxl-crops-coords-top-left', metavar="COORD",
                    action='store', nargs='+', default=[], type=_type_image_coordinate,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-conditioning
                            parameters in the format "0,0". --sdxl-crops-coords-top-left can be used to generate an image that
                            appears to be "cropped" from the position --sdxl-crops-coords-top-left downwards. Favorable,
                            well-centered images are usually achieved by setting --sdxl-crops-coords-top-left to "0,0".
                            Part of SDXL's micro-conditioning as explained in section 2.2 of 
                            [https://huggingface.co/papers/2307.01952].""")

parser.add_argument('--sdxl-original-size', '--sdxl-original-sizes', dest='sdxl_original_sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "original-size" micro-conditioning parameters in
                            the format (WIDTHxHEIGHT). If not the same as --sdxl-target-size the image will appear to be
                            down or upsampled. --sdxl-original-size defaults to --output-size if not specified. Part of
                            SDXL\'s micro-conditioning as explained in section 2.2 of 
                            [https://huggingface.co/papers/2307.01952]""")

parser.add_argument('--sdxl-target-size', '--sdxl-target-sizes', dest='sdxl_target_sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "target-size" micro-conditioning parameters in
                            the format (WIDTHxHEIGHT). For most cases, --sdxl-target-size should be set to the desired
                            height and width of the generated image. If not specified it will default to --output-size.
                            Part of SDXL\'s micro-conditioning as explained in section 2.2 of 
                            [https://huggingface.co/papers/2307.01952]""")

parser.add_argument('--sdxl-negative-aesthetic-scores', metavar="FLOAT",
                    action='store', nargs='+', default=[], type=float,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "negative-aesthetic-score" micro-conditioning parameters.
                            Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                            Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition.""")

parser.add_argument('--sdxl-negative-original-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-conditioning parameters.
                            Negatively condition the generation process based on a specific image resolution. Part of SDXL's
                            micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                            For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208""")

parser.add_argument('--sdxl-negative-target-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-conditioning parameters.
                            To negatively condition the generation process based on a target image resolution. It should be as same
                            as the "target_size" for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]. For more information, refer to this issue thread:
                            https://github.com/huggingface/diffusers/issues/4208.""")

parser.add_argument('--sdxl-negative-crops-coords-top-left', metavar="COORD",
                    action='store', nargs='+', default=[], type=_type_image_coordinate,
                    help="""One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-conditioning
                            parameters in the format "0,0". Negatively condition the generation process based on a specific
                            crop coordinates. Part of SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]. For more information, refer
                            to this issue thread: https://github.com/huggingface/diffusers/issues/4208.""")

# SDXL Refiner pipeline

parser.add_argument('--sdxl-refiner-prompts', nargs='+', action='store',
                    metavar="PROMPT",
                    default=None,
                    type=_type_prompts,
                    help="""List of prompts to try with the SDXL refiner model, 
                    by default the refiner model gets the primary prompt, this argument 
                    overrides that with a prompt of your choosing. The negative prompt 
                    component can be specified with the same syntax as --prompts""")

parser.add_argument('--sdxl-refiner-second-prompts', nargs='+', action='store',
                    metavar="PROMPT",
                    default=None,
                    type=_type_prompts,
                    help="""List of prompts to try with the SDXL refiner models secondary 
                    text encoder, by default the refiner model gets the primary prompt passed
                    to its second text encoder, this argument overrides that with a prompt 
                    of your choosing. The negative prompt component can be specified with the 
                    same syntax as --prompts""")

parser.add_argument('--sdxl-refiner-aesthetic-scores', metavar="FLOAT",
                    action='store', nargs='+', default=[], type=float,
                    help="See: --sdxl-aesthetic-scores, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-crops-coords-top-left', metavar="COORD",
                    action='store', nargs='+', default=[], type=_type_image_coordinate,
                    help="See: --sdxl-crops-coords-top-left, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-original-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="See: --sdxl-refiner-original-sizes, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-target-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="See: --sdxl-refiner-target-sizes, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-negative-aesthetic-scores', metavar="FLOAT",
                    action='store', nargs='+', default=[], type=float,
                    help="See: --sdxl-negative-aesthetic-scores, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-negative-original-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="See: --sdxl-negative-original-sizes, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-negative-target-sizes', metavar="SIZE",
                    action='store', nargs='+', default=[], type=_type_micro_conditioning_size,
                    help="See: --sdxl-negative-target-sizes, applied to SDXL refiner pass.")

parser.add_argument('--sdxl-refiner-negative-crops-coords-top-left', metavar="COORD",
                    action='store', nargs='+', default=[], type=_type_image_coordinate,
                    help="See: --sdxl-negative-crops-coords-top-left, applied to SDXL refiner pass.")


def _type_sdxl_high_noise_fractions(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


parser.add_argument('-hnf', '--sdxl-high-noise-fractions', action='store', nargs='+', default=None,
                    metavar="FLOAT",
                    type=_type_sdxl_high_noise_fractions,
                    help="""High noise fraction for Stable Diffusion XL (torch-sdxl), this fraction of inference steps
                         will be processed by the base model, while the rest will be processed by the refiner model.
                         Multiple values to this argument will result in additional generation steps for each value.
                         In certain situations when the mixture of denoisers algorithm is not supported,
                         such as when using --control-nets and inpainting with SDXL, the inverse proportion
                         of this value IE: (1.0 - high-noise-fraction) becomes the --image-seed-strength 
                         input to the SDXL refiner. (default: [0.8])""")

parser.add_argument('-ri', '--sdxl-refiner-inference-steps', action='store', nargs='+', default=None, metavar="INT",
                    type=_type_inference_steps,
                    help="""One or more inference steps values for the SDXL refiner when in use. 
                    Override the number of inference steps used by the SDXL refiner, 
                    which defaults to the value taken from --inference-steps.""")

parser.add_argument('-rg', '--sdxl-refiner-guidance-scales', action='store', nargs='+', default=None, metavar="FLOAT",
                    type=_type_guidance_scale,
                    help="""One or more guidance scale values for the SDXL refiner when in use. 
                    Override the guidance scale value used by the SDXL refiner, 
                    which defaults to the value taken from --guidance-scales.""")

parser.add_argument('-rgr', '--sdxl-refiner-guidance-rescales', action='store', nargs='+', default=None,
                    metavar="FLOAT",
                    type=_type_guidance_scale,
                    help="""One or more guidance rescale values for the SDXL refiner when in use. 
                    Override the guidance rescale value used by the SDXL refiner,
                    which defaults to the value taken from --guidance-rescales.""")

parser.add_argument('--safety-checker', action='store_true', default=False,
                    help="""Enable safety checker loading, this is off by default.
                         When turned on images with NSFW content detected may result in solid black output.
                         Some pretrained models have settings indicating a safety checker is not to be loaded,
                         in that case this option has no effect.""")


def _type_device(device):
    try:
        if not _pipelinewrapper.is_valid_device_string(device):
            raise argparse.ArgumentTypeError(f'Must be cuda or cpu. Unknown value: {device}')
    except _pipelinewrapper.InvalidDeviceOrdinalException as e:
        raise argparse.ArgumentTypeError(e)

    return device


parser.add_argument('-d', '--device', action='store', default='cuda', type=_type_device,
                    help='cuda / cpu. (default: cuda). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific GPU.')

parser.add_argument('-t', '--dtype', action='store', default='auto', type=_type_dtype,
                    help='Model precision: float16 / float32 / auto. (default: auto)')


def _type_output_size(size):
    r = size.lower().split('x')

    try:
        if len(r) < 2:
            x, y = int(r[0]), int(r[0])
        else:
            x, y = int(r[0]), int(r[1])
    except ValueError:
        raise argparse.ArgumentTypeError('Output dimensions must be integer values.')

    if x % 8 != 0:
        raise argparse.ArgumentTypeError('Output X dimension must be divisible by 8.')

    if y % 8 != 0:
        raise argparse.ArgumentTypeError('Output Y dimension must be divisible by 8.')

    return x, y


parser.add_argument('-s', '--output-size', action='store', default=None, type=_type_output_size,
                    metavar="SIZE",
                    help="""Image output size.
                         If an image seed is used it will be resized to this dimension with aspect ratio
                         maintained, width will be fixed and a new height will be calculated. If only one integer
                         value is provided, that is the value for both dimensions. X/Y dimension values should
                         be separated by "x".  (default: 512x512 when no image seeds are specified)""")

parser.add_argument('-o', '--output-path', action='store', default=os.path.join(os.getcwd(), 'output'),
                    metavar="PATH",
                    help="""Output path for generated images and files.
                         This directory will be created if it does not exist. (default: ./output)""")

parser.add_argument('-op', '--output-prefix', action='store', default=None, type=str, metavar="PREFIX",
                    help="""Name prefix for generated images and files.
                         This prefix will be added to the beginning of every generated file,
                         followed by an underscore.""")

parser.add_argument('-ox', '--output-overwrite', action='store_true', default=False,
                    help="""Enable overwrites of files in the output directory that already exists.
                            The default behavior is not to do this, and instead append a filename suffix:
                             "_duplicate_(number)" when it is detected that the generated file name already exists.""")

parser.add_argument('-oc', '--output-configs', action='store_true', default=False,
                    help="""Write a configuration text file for every output image or animation.
                            The text file can be used reproduce that particular output image or animation by piping
                            it to dgenerate STDIN, for example "dgenerate < config.txt". These files will be written
                            to --output-directory and are affected by --output-prefix and --output-overwrite as well. 
                            The files will be named after their corresponding image or animation file. Configuration 
                            files produced for animation frame images will utilize --frame-start and --frame-end to 
                            specify the frame number.""")

parser.add_argument('-om', '--output-metadata', action='store_true', default=False,
                    help="""Write the information produced by --output-configs to the PNG metadata of each image.
                            Metadata will not be written to animated files (yet). The data is written to a 
                            PNG metadata property named DgenerateConfig and can be read using ImageMagick like so: 
                            "magick identify -format "%%[Property:DgenerateConfig] generated_file.png".""")

parser.add_argument('-p', '--prompts', nargs='+', action='store', metavar="PROMPT",
                    default=[_prompt.Prompt()],
                    type=_type_prompts,
                    help="""List of prompts to try, an image group is generated for each prompt,
                         prompt data is split by ; (semi-colon). The first value is the positive
                         text influence, things you want to see. The Second value is negative
                         influence IE. things you don't want to see.
                         Example: --prompts "shrek flying a tesla over detroit; clouds, rain, missiles".
                         (default: [(empty string)])""")

seed_options = parser.add_mutually_exclusive_group()


def _type_seeds(val):
    try:
        val = int(val)
    except ValueError as e:
        raise argparse.ArgumentTypeError('Must be an integer')

    return val


seed_options.add_argument('-se', '--seeds', nargs='+', action='store', metavar="SEED", type=_type_seeds,
                          help="""List of seeds to try, define fixed seeds to achieve deterministic output.
                               This argument may not be used when --gse/--gen-seeds is used.
                               (default: [randint(0, 99999999999999)])""")


def _type_gen_seeds(val):
    try:
        val = int(val)
    except ValueError as e:
        raise argparse.ArgumentTypeError('Must be an integer')

    return dgenerate.gen_seeds(val)


seed_options.add_argument('-gse', '--gen-seeds', action='store', type=_type_gen_seeds, metavar="COUNT",
                          dest='seeds',
                          help="""Auto generate N random seeds to try. This argument may not
                               be used when -se/--seeds is used.""")


def _type_animation_format(val):
    val = val.lower()
    if val not in _mediaoutput.supported_animation_writer_formats():
        supported = _textprocessing.oxford_comma(_mediaoutput.supported_animation_writer_formats(), "or")
        raise argparse.ArgumentTypeError(
            f'Must be {supported}. Unknown value: {val}')
    return val


parser.add_argument('-af', '--animation-format', action='store', default='mp4', type=_type_animation_format,
                    metavar="FORMAT",
                    help=f"""Output format when generating an animation from an input video / gif / webp etc.
                         Value must be one of: {_textprocessing.oxford_comma(_mediaoutput.supported_animation_writer_formats(), "or")}.
                         (default: mp4)""")


def _type_frame_start(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('-fs', '--frame-start', action='store', default=0, type=_type_frame_start, metavar="FRAME_NUMBER",
                    help='Starting frame slice point for animated files, the specified frame will be included.')


def _type_frame_end(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


parser.add_argument('-fe', '--frame-end', action='store', default=None, type=_type_frame_end, metavar="FRAME_NUMBER",
                    help='Ending frame slice point for animated files, the specified frame will be included.')

parser.add_argument('-is', '--image-seeds', action='store', nargs='+', default=[], metavar="SEED",
                    help="""List of image seeds to try when processing image seeds, these may
                         be URLs or file paths. Videos / GIFs / WEBP files will result in frames
                         being rendered as well as an animated output file being generated if more
                         than one frame is available in the input file. Inpainting for static images can be
                         achieved by specifying a black and white mask image in each image seed string using
                         a semicolon as the separating character, like so: "my-seed-image.png;my-image-mask.png",
                         white areas of the mask indicate where generated content is to be placed in your seed
                         image. Output dimensions specific to the image seed can be specified by placing the
                         dimension at the end of the string following a semicolon like so:
                         "my-seed-image.png;512x512" or "my-seed-image.png;my-image-mask.png;512x512".
                         Inpainting masks can be downloaded for you from a URL or be a path to a file on disk.
                         When using --control-nets, a singular image specification is interpreted as the control
                         guidance image. Using --control-nets with img2img or inpainting can be accomplished with
                         the syntax: "my-seed-image.png;mask=my-image-mask.png;control=my-control-image.png;resize=512x512".
                         The "mask" and "resize" arguments are optional when using --control-nets. Videos, GIFs,
                         and WEBP are also supported as inputs when using --control-nets, even for the "control"
                         argument. --image-seeds is capable of reading from 3 animated files at once or any combination
                         of animated files and images, the animated file with the least amount of frames dictates how
                         many frames are generated.
                         """)

image_seed_noise_opts = parser.add_mutually_exclusive_group()

parser.add_argument('--seed-image-preprocessors', action='store', nargs='+', default=None, metavar="PREPROCESSOR",
                    help="""Specify one or more image preprocessor actions to preform on the primary
                    image specified by --image-seeds. For example: --seed-image-preprocessors "flip" "mirror" "grayscale".
                    To obtain more information about what image preprocessors are available and how to use them, 
                    see: --image-preprocessor-help.
                    """)

parser.add_argument('--mask-image-preprocessors', action='store', nargs='+', default=None, metavar="PREPROCESSOR",
                    help="""Specify one or more image preprocessor actions to preform on the inpaint mask
                    image specified by --image-seeds. For example: --mask-image-preprocessors "invert".
                    To obtain more information about what image preprocessors are available and how to use them, 
                    see: --image-preprocessor-help.
                    """)

parser.add_argument('--control-image-preprocessors', action='store', nargs='+', default=None, metavar="PREPROCESSOR",
                    help="""Specify one or more image preprocessor actions to preform on the control
                    image specified by --image-seeds. For example: 
                    --control-image-preprocessors "canny;lower=50;upper=100". This option is ment to be used 
                    in combination with --control-nets. To obtain more information about what image 
                    preprocessors are available and how to use them, see: --image-preprocessor-help.
                    """)

parser.add_argument('--image-preprocessor-help', action='store', nargs='*', default=None, metavar="PREPROCESSOR",
                    dest=None,  # This is handled elsewhere
                    help="""Use this option alone with no model specification in order to 
                    list available image preprocessor module names. Specifying one or more module names
                    after this option will cause usage documentation for the specified modules to be printed.""")


def _type_image_seed_strengths(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


image_seed_noise_opts.add_argument('-iss', '--image-seed-strengths', action='store', nargs='+', default=None,
                                   metavar="FLOAT",
                                   type=_type_image_seed_strengths,
                                   help=f"""List of image seed strengths to try. Closer to 0 means high usage of the seed image
                         (less noise convolution), 1 effectively means no usage (high noise convolution).
                         Low values will produce something closer or more relevant to the input image, high
                         values will give the AI more creative freedom. (default: [0.8])""")


def _type_upscaler_noise_levels(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')

    return val


image_seed_noise_opts.add_argument('-uns', '--upscaler-noise-levels', action='store', nargs='+', default=None,
                                   metavar="INTEGER",
                                   type=_type_upscaler_noise_levels,
                                   help=f"""
                    List of upscaler noise levels to try when using the super resolution upscaler 
                    (torch-upscaler-x4). These values will be ignored when using (torch-upscaler-x2).
                    The higher this value the more noise is added to the image before upscaling 
                    (similar to --image-seed-strength). (default: [20])""")

parser.add_argument('-gs', '--guidance-scales', action='store', nargs='+',
                    default=[_pipelinewrapper.DEFAULT_GUIDANCE_SCALE],
                    metavar="FLOAT",
                    type=_type_guidance_scale,
                    help="""List of guidance scales to try. Guidance scale effects how much your
                         text prompt is considered. Low values draw more data from images unrelated
                         to text prompt. (default: [5])""")


def _type_image_guidance_scale(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


parser.add_argument('-igs', '--image-guidance-scales', action='store', nargs='+', default=None,
                    metavar="FLOAT",
                    type=_type_image_guidance_scale,
                    help="""Push the generated image towards the inital image when using --model-type *-pix2pix models.
                            Use in conjunction with --image-seeds, inpainting (masks) and --control-nets are not supported.
                            Image guidance scale is enabled by setting image-guidance-scale > 1. Higher image guidance scale
                            encourages generated images that are closely linked to the source image, usually at the expense
                            of lower image quality. Requires a value of at least 1. (default: [1.5])""")

parser.add_argument('-gr', '--guidance-rescales', action='store', nargs='+', default=[],
                    metavar="FLOAT",
                    type=_type_guidance_scale,
                    help="""List of guidance rescale factors to try. Proposed by [Common Diffusion Noise Schedules and 
                            Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) "guidance_scale" is defined 
                            as "φ" in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed]
                            (https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure 
                            when using zero terminal SNR. This is supported for basic text to image generation 
                            when using --model-type "torch" but not inpainting, img2img, or --control-nets. 
                            When using --model-type "torch-sdxl" it is supported for basic generation, inpainting, 
                            and img2img, unless --control-nets is specified in which case only inpainting is supported.
                            It is supported for --model-type "torch-sdxl-pix2pix" but not --model-type "torch-pix2pix".
                            (default: [0.0])""")

parser.add_argument('-ifs', '--inference-steps', action='store', nargs='+',
                    default=[_pipelinewrapper.DEFAULT_INFERENCE_STEPS],
                    type=_type_inference_steps,
                    metavar="INTEGER",
                    help="""Lists of inference steps values to try. The amount of inference (de-noising) steps
                         effects image clarity to a degree, higher values bring the image closer to what
                         the AI is targeting for the content of the image. Values between 30-40
                         produce good results, higher values may improve image quality and or
                         change image content. (default: [30])""")


def _parse_args(args=None) -> DgenerateArguments:
    args = parser.parse_args(args, namespace=DgenerateArguments())

    def args_that_start_with(s):
        return (a for a in dir(args) if a.startswith(s) and getattr(args, a))

    def args_that_end_with(s):
        return (a for a in dir(args) if a.endswith(s) and getattr(args, a))

    if args.output_size is None and not args.image_seeds:
        args.output_size = (512, 512) if not _pipelinewrapper.model_type_is_sdxl(args.model_type) else (1024, 1024)

    if not args.image_seeds and args.image_seed_strengths:
        raise argparse.ArgumentTypeError(
            'you cannot specify --image-seed-strengths without --image-seeds.')

    if not _pipelinewrapper.model_type_is_upscaler(args.model_type):
        if args.upscaler_noise_levels:
            raise argparse.ArgumentTypeError(
                'you cannot specify --upscaler-noise-levels for a '
                'non upscaler model type, see --model-type.')
    elif args.control_net_paths:
        raise argparse.ArgumentTypeError(
            'argument --control-nets is not compatible '
            'with upscaler models, see --model-type.')
    elif args.upscaler_noise_levels is None:
        args.upscaler_noise_levels = [_pipelinewrapper.DEFAULT_X4_UPSCALER_NOISE_LEVEL]

    if not _pipelinewrapper.model_type_is_pix2pix(args.model_type):
        if args.image_guidance_scales:
            raise argparse.ArgumentTypeError(
                'argument --image-guidance-scales only valid with '
                'pix2pix models, see --model-type.')
    elif args.control_net_paths:
        raise argparse.ArgumentTypeError(
            'argument --control-nets '
            'is not compatible with pix2pix models, see --model-type.')
    elif not args.image_guidance_scales:
        args.image_guidance_scales = [_pipelinewrapper.DEFAULT_IMAGE_GUIDANCE_SCALE]

    if args.control_image_preprocessors:
        if not args.image_seeds:
            raise argparse.ArgumentTypeError(f'you cannot specify --control-image-preprocessors '
                                             f'without --image-seeds.')

    if not args.image_seeds:
        invalid_args = []
        for preprocessor_args in args_that_end_with('preprocessors'):
            invalid_args.append(
                f'you cannot specify --{preprocessor_args.replace("_", "-")} '
                f'without --image-seeds.')
        if invalid_args:
            raise argparse.ArgumentTypeError('\n'.join(invalid_args))

    if not _pipelinewrapper.model_type_is_sdxl(args.model_type):
        invalid_args = []
        for sdxl_args in args_that_start_with('sdxl'):
            invalid_args.append(f'you cannot specify --{sdxl_args.replace("_", "-")} '
                                f'for a non SDXL model type, see --model-type.')
        if invalid_args:
            raise argparse.ArgumentTypeError('\n'.join(invalid_args))

        args.sdxl_high_noise_fractions = []
    else:
        if not args.sdxl_refiner_path:
            invalid_args = []
            for sdxl_args in args_that_start_with('sdxl_refiner'):
                invalid_args.append(f'you cannot specify --{sdxl_args.replace("_", "-")} '
                                    f'without --sdxl-refiner.')
            if invalid_args:
                raise argparse.ArgumentTypeError('\n'.join(invalid_args))
        else:
            if args.sdxl_high_noise_fractions is None:
                # Default value
                args.sdxl_high_noise_fractions = [_pipelinewrapper.DEFAULT_SDXL_HIGH_NOISE_FRACTION]

    if not _pipelinewrapper.model_type_is_torch(args.model_type):
        if args.vae_tiling or args.vae_slicing:
            raise argparse.ArgumentTypeError(
                'argument --vae-tiling/--vae-slicing not supported for '
                'non torch model type, see --model-type.')

    if args.scheduler == 'help' and args.sdxl_refiner_scheduler == 'help':
        raise argparse.ArgumentTypeError(
            'cannot list compatible schedulers for the main model and the SDXL refiner at '
            'the same time. Do not use the scheduler "help" option for --scheduler '
            'and --sdxl-refiner-scheduler simultaneously.')

    if args.image_seeds:
        if args.image_seed_strengths is None:
            # Default value
            args.image_seed_strengths = [_pipelinewrapper.DEFAULT_IMAGE_SEED_STRENGTH]
    else:
        args.image_seed_strengths = []

    return args


def parse_args(args, exit_on_error=True) -> DgenerateArguments:
    if not exit_on_error:
        return _parse_args(args)

    try:
        return _parse_args(args)
    except argparse.ArgumentTypeError as e:
        _messages.log(f'dgenerate: error: {e}', level=_messages.ERROR)
        sys.exit(1)
