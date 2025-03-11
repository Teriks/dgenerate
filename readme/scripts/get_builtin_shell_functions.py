import dgenerate.batchprocess as _b

runner = _b.ConfigRunner()

print(runner.generate_functions_help(runner.template_functions.keys()))
