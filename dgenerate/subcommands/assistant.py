import dgenerate.subcommands.subcommand as _subcommand


class AssistantSubCommand(_subcommand.SubCommand):
    """
    Write a dgenerate config from a plain language request.

    A local Qwen model running through xllamacpp writes the config, guided by
    the closest example configs and documentation, retrieved from an index of
    the dgenerate examples and docs that ships with dgenerate. The config is
    then checked with dgenerate's own config runner and argument parser without
    loading any diffusion models, and the model is asked to fix anything
    dgenerate rejects.

    Requires the xllamacpp extra (pip install dgenerate[xllamacpp]).

    The chat and embedding models are downloaded from Hugging Face on first use.
    ``--embed-model`` selects the embedding model. The models offered in Generate
    Config have an index packaged with dgenerate. Any other Qwen3-Embedding model
    builds an index the first time it is used.

    File paths in the request are treated as input files, relative paths are
    relative to the current directory and are rewritten relative to --output.

    Examples:

    dgenerate --sub-command assistant "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    dgenerate --sub-command assistant -o fox.dgen "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    See: dgenerate --sub-command assistant --help
    """

    NAMES = ['assistant']

    def __init__(self, program_name='assistant', **kwargs):
        super().__init__(**kwargs)
        self._program_name = program_name

    def __call__(self) -> int:
        import dgenerate.assistant.cli as _cli
        return _cli.main(self.args, prog=self._program_name, local_files_only=self.local_files_only)
