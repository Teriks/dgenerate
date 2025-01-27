# updates dgenerates help output text in the readme

import os
import re
import subprocess
import dgenerate.textprocessing

# Change to the parent directory of the script location
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

os.environ['COLUMNS'] = "110"

# Replace content between headers
pattern = r'(?s)Help Output\n=+[\s\S]*?Windows Install\n=+'

help_txt = subprocess.run(['dgenerate', '--no-stdin', '--help'],
                      capture_output=True, text=True, shell=True).stdout.strip()

# replace ANSI escape codes
help_txt = re.sub(r'\x1b\[[0-9;]*m', '', help_txt)

replacement = f"Help Output\n===========\n\n.. code-block:: text\n\n" \
              f"{dgenerate.textprocessing.indent_text(help_txt, '    ', '    ')}" \
              f"\n\nWindows Install\n==============="

with open('README.rst', 'rt') as readme:
    text = readme.read()
    new_content = re.sub(pattern, replacement, text)

with open('README.rst', 'wt') as readme:
    readme.write(new_content)
