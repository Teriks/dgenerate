import argparse
import os
import subprocess
import unittest

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-c', '--clean', default=False, action='store_true')
parser.add_argument('-e', '--examples', default=False, action='store_true')

args = parser.parse_args()

runner = unittest.TextTestRunner()

if runner.run(unittest.defaultTestLoader.discover("tests", pattern='*_test.py')).wasSuccessful():

    if not args.examples:
        exit(0)

    print('unit tests passed, running examples..')

    if args.clean:
        os.chdir('examples')
        print('running: git clean -f -d in examples folder...')
        subprocess.run('git clean -f -d', shell=True)
        os.chdir('..')

    run_string = f'python examples/run.py --device {args.device} --short-animations --output-configs --output-metadata -v > examples/examples.log 2>&1'
    print('running:', run_string)
    subprocess.run(run_string, shell=True)
else:
    exit(1)
