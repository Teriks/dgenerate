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
import dgenerate
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.batchprocess.util as _b_util
import dgenerate.memoize as _memoize


def _dtype_type(plugin):
    def f(val):
        if val is None:
            return val

        lower = val.lower()

        if lower not in {'float16', 'float32'}:
            raise plugin.argument_error(f'Unsupported --dtype value: {lower}')
        return lower

    return f


class ToDiffusersSubCommand(_subcommand.SubCommand):
    """
    Convert single file diffusion model checkpoints from CivitAI and elsewhere into diffusers format (a folder on disk with configuration).

    This can be useful if you want to load a single file checkpoint with quantization.

    This conversion is done automatically by dgenerate when you load a single file checkpoint
    and request quantization post-processing to occur.

    You may also save models loaded from Hugging Face repos.

    Examples:

    dgenerate --sub-command to-diffusers "all_in_one.safetensors" --model-type sd --output model_directory

    dgenerate --sub-command to-diffusers "https://modelsite.com/all_in_one.safetensors" --model-type sdxl --output model_directory

    See: dgenerate --sub-command to-diffusers --help
    """

    NAMES = ['to-diffusers']

    def __init__(self, program_name='to-diffusers', **kwargs):
        super().__init__(**kwargs)

        self._parser = parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description='Save a loaded model to a diffusers format pretrained model folder, models can be '
                        'loaded from a single file or Hugging Face hub repository.')

        parser.add_argument('model_path',
                            help='Model path, as you would provide to dgenerate to generate images.')
        parser.add_argument('-mt', '--model-type',
                            help='Model type, as you would provide to dgenerate to generate images, '
                                 'must match the checkpoint model type.', required=False, default=dgenerate.ModelType.SD)
        parser.add_argument('-rev', '--revision',
                            help='Model revision, if loading from Hugging Face hub.', required=False, default=None)
        parser.add_argument('-sbf', '--subfolder',
                            help='Model subfolder, if loading from Hugging Face hub.', required=False, default=None)
        parser.add_argument('-t', '--dtypes',
                            help='Model dtypes to generate, this generates variants, such as "fp16", you may specify '
                                 'up to 2 values. Accepted values are: float16, and float32. By default only the 32 bit '
                                 'variant is saved if you do not specify this argument, if you want both variants you must '
                                 'specify both dtypes simultaneously.', nargs='*',
                            default=[None], type=_dtype_type(self))
        parser.add_argument('-olc', '--original-config',
                            help='Original LDM config (.yaml) file.', required=False, default=None)
        parser.add_argument('-atk', '--auth-token',
                            help='Optional Hugging Face authentication token value.', required=False, default=None)
        parser.add_argument('-o', '--output',
                            help='Output directory for the converted model, this is a folder you '
                                 'can point dgenerate at to generate images.', required=True)
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable debug output?')

        parser.add_argument('-ofm', '--offline-mode', action='store_true',
                            help="""Prevent downloads of resources that do not exist on disk already.""")

    def __call__(self) -> int:

        args = self._parser.parse_args(self.args)

        if self._parser.return_code is not None:
            return self._parser.return_code

        if len(args.dtypes) > 2:
            raise self.argument_error(
                'Too many --dtypes values specified, you may only specify up to 2.')

        for dtype in args.dtypes:
            dtype = _pipelinewrapper.get_data_type_enum(dtype) if dtype else None

            try:
                if args.verbose:
                    dgenerate.messages.push_level(dgenerate.messages.DEBUG)

                with _memoize.disable_memoization_context():
                    pipe = _pipelinewrapper.create_diffusion_pipeline(
                        model_path=args.model_path,
                        model_type=_pipelinewrapper.get_model_type_enum(args.model_type),
                        revision=args.revision,
                        subfolder=args.subfolder,
                        variant='fp16' if dtype == dgenerate.DataType.FLOAT16 else None,
                        dtype=dtype if dtype is not None else dgenerate.DataType.AUTO,
                        auth_token=args.auth_token,
                        missing_submodules_ok=True,
                        local_files_only=self.local_files_only or args.offline_mode
                    )
            except (_pipelinewrapper.InvalidModelFileError,
                    _pipelinewrapper.InvalidModelUriError,
                    _pipelinewrapper.InvalidSchedulerNameError,
                    _pipelinewrapper.UnsupportedPipelineConfigError,
                    dgenerate.ModelNotFoundError,
                    dgenerate.NonHFDownloadError) as e:
                raise self.argument_error(f'Failed to save pretrained model: {e}') from e
            finally:
                if args.verbose:
                    dgenerate.messages.pop_level()

            pipe.pipeline.save_pretrained(args.output)

        return 0
