Very basic environment for testing flax and nix features on Windows, this should also work on *nix platforms.

To start a shell use: ``python run.py``

To run a command in a fresh environment use for example:

```
python run.py "python3 examples/run.py --short-animations --subprocess-only &> examples/examples-docker.log"

python run.py "python3 run_tests.py --clean --examples"
```

Take note that the initial working directory of the environment is the top level directory of the dgenerate project.

And that it is a linux environment, therefore the python executable is named python3.