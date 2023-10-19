import dgenerate.hfutil as _hfutil
import dgenerate.messages as _messages
import dgenerate.batchprocess as _bp
import dgenerate.arguments as _args
import glob

_messages.LEVEL = _messages.DEBUG



runner = _bp.create_config_runner()

def my_invoker(args):
    argo = _args.parse_args(args, throw=True)

    print(_hfutil.estimate_model_memory_use(argo.model_path,
                                            variant=argo.variant,
                                            subfolder=argo.model_subfolder,
                                            revision=argo.revision) / 1024 ** 3)

    return 0

runner.invoker = my_invoker

for i in glob.glob('examples/**/*config.txt', recursive=True):
    if 'flax' not in i:
        with open(i, mode='rt') as f:
            runner.run_file(f)

