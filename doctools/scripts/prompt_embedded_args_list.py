import typing
import dgenerate
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

hints = typing.get_type_hints(dgenerate.DiffusionArguments)

supported_types = {
    _types.Boolean: "bool",
    _types.OptionalBoolean: "bool",
    _types.Size: "Size: WxH",
    _types.Sizes: "[Size: WxH, ...]",
    _types.OptionalSize: "Size: WxH",
    _types.OptionalSizes: "[Size: WxH, ...]",
    _types.Padding: "Padding: P, WxH, LxTxRxB",
    _types.Paddings: "[Padding: P, WxH, LxTxRxB, ...]",
    _types.OptionalPadding: "Padding: P, WxH, LxTxRxB",
    _types.OptionalPaddings: "[Padding: P, WxH, LxTxRxB, ...]",
    _types.Integer: "int",
    _types.Integers: "[int, ...]",
    _types.OptionalInteger: "int",
    _types.OptionalIntegers: "[int, ...]",
    _types.Float: "float",
    _types.Floats: "[float, ...]",
    _types.OptionalFloat: "float",
    _types.OptionalFloats: "[float, ...]",
    _types.String: "str",
    _types.OptionalString: "str",
    _types.Name: "str",
    _types.Names: "[str, ...]",
    _types.OptionalName: "str",
    _types.OptionalNames: "[str, ...]",
    _types.Uri: "str",
    _types.Uris: "[str, ...]",
    _types.OptionalUri: "str",
    _types.OptionalUris: "[str, ...]",
}

for name, hint in hints.items():
    if any(hint is h for h in supported_types):
        if not dgenerate.DiffusionArguments.prompt_embedded_arg_checker(name, None):
            print(_textprocessing.dashup(name) + ": " + supported_types[hint])
