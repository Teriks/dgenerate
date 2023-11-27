import unittest
import subprocess

runner = unittest.TextTestRunner()

if runner.run(unittest.defaultTestLoader.discover("tests", pattern='*_test.py')).wasSuccessful():
    print('unit tests passed, running examples..')
    subprocess.run(
        'python examples/run.py --device cuda:1 --offline-mode --short-animations --output-configs --output-metadata -v > examples/examples.log 2>&1', shell=True)
else:
    exit(1)