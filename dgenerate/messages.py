import sys

from .textprocessing import underline

LEVEL = 0

INFO = 0
WARNING = 1
ERROR = 2
DEBUG = 3


def log(*args, **kwargs):
    level = kwargs.get('level', INFO)
    underline_me = kwargs.get('underline', False)
    file = sys.stdout
    if level != INFO and LEVEL == INFO:
        if level != ERROR and level != WARNING:
            return
        else:
            file = sys.stderr

    if underline_me:
        print(underline(' '.join(str(a) for a in args)), file=file)
    else:
        print(' '.join(str(a) for a in args), file=file)
