"""
Locate Tcl/Tk script directories for PyInstaller.

Python 3.13+ virtual environments on Windows often cannot initialize Tcl
while PyInstaller is analyzing the program, so the tkinter hook never copies
the script library. The frozen runtime hook then raises FileNotFoundError
for ``_tcl_data``.
"""

import glob
import os
import sys


def tcl_tk_datas():
    """
    Return ``(source, dest)`` pairs for Tcl/Tk directories under the base
    interpreter prefix.

    Versioned ``tcl8.6`` / ``tk8.6`` directories are stored as ``_tcl_data``
    and ``_tk_data``, which is where PyInstaller's tkinter runtime hook
    looks. The unversioned ``tcl8`` modules directory keeps its name.
    """
    base = getattr(sys, 'base_prefix', sys.prefix)
    roots = [
        os.path.join(base, 'tcl'),
        os.path.join(base, 'lib'),
        os.path.join(base, 'Lib'),
    ]
    datas = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        matches = glob.glob(os.path.join(root, 'tcl*')) + glob.glob(os.path.join(root, 'tk*'))
        for path in matches:
            if not os.path.isdir(path):
                continue
            name = os.path.basename(path)
            if name.startswith('tk') and '.' in name:
                dest = '_tk_data'
            elif name.startswith('tcl') and '.' in name:
                dest = '_tcl_data'
            elif name.startswith('tcl') and '.' not in name:
                dest = name
            else:
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            datas.append((path, dest))
    return datas


def tcl_tk_environ(environ=None):
    """
    Return an environment with ``TCL_LIBRARY`` and ``TK_LIBRARY`` set when
    those directories exist and the variables are not already set.
    """
    env = dict(os.environ if environ is None else environ)
    for source, dest in tcl_tk_datas():
        if dest == '_tcl_data' and not env.get('TCL_LIBRARY'):
            env['TCL_LIBRARY'] = source
        elif dest == '_tk_data' and not env.get('TK_LIBRARY'):
            env['TK_LIBRARY'] = source
    return env
