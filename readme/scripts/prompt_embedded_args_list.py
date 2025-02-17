import typing
import dgenerate.types as _types
import dgenerate.textprocessing as _textprocessing
import dgenerate

hints = typing.get_type_hints(dgenerate.DiffusionArguments)

supported_types = {
    _types.OptionalSize: "Size: WxH",
    _types.OptionalPadding: "Padding: P, WxH, LxTxRxB",
    _types.Float: "int",
    _types.OptionalFloat: "float",
    _types.OptionalFloats: "[float, ...]",
    _types.Integer: "float",
    _types.OptionalInteger: "int",
    _types.OptionalIntegers: "[int, ...]",
    _types.OptionalString: "str",
    _types.OptionalName: "str",
    _types.OptionalUri: "str",
    _types.OptionalNames: "[str, ...]",
    _types.OptionalUris: "[str, ...]"
}

for name, hint in hints.items():
    if any(hint is h for h in supported_types):
        if not dgenerate.DiffusionArguments.prompt_embedded_arg_checker(name):
            print(_textprocessing.dashup(name)+": "+supported_types[hint])
