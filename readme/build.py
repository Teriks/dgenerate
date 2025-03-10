#! /usr/bin/env python3
# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import json
import os
import re
import subprocess
import sys
from collections import defaultdict

COMMAND_CACHE = dict()

proj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

with open(os.path.join(proj_dir, 'dgenerate', '__init__.py')) as _f:
    VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', _f.read(), re.MULTILINE).group(1)


def get_git_revision():
    try:
        # Try to get the current tag name
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--exact-match'],
            stderr=subprocess.DEVNULL).strip().decode()
        return tag  # Return the tag name if on a tag
    except subprocess.CalledProcessError:
        try:
            # If not on a tag, return the branch name
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).strip().decode()
            return branch
        except subprocess.CalledProcessError:
            return 'main'  # If both commands fail, return 'main'
    except FileNotFoundError:
        return 'main'  # If git is not installed, return 'main'


REVISION = get_git_revision()


def find_and_condense_links(rst_content):
    print('Condensing links...')

    link_pattern = re.compile(r'`([^`]+) <(https?://[^>]+)>`_')
    link_counts = defaultdict(int)

    # Count occurrences of each (title, URL) pair
    for title, url in link_pattern.findall(rst_content):
        link_counts[(title, url)] += 1

    # Identify links that appear more than once
    duplicates = {key: count for key, count in link_counts.items() if count > 1}
    if not duplicates:
        print("No duplicate links found.")
        return rst_content

    ref_map = {}
    reference_definitions = []

    for (title, url), count in duplicates.items():
        ref_name = title.replace(' ', '_')
        if ref_name in ref_map:
            ref_name += f"_{len(ref_map)}"
        ref_map[url] = ref_name
        reference_definitions.append(f".. _{ref_name}: {url}")

    def link_replacer(match):
        title, url = match.groups()
        if url in ref_map:
            return f"`{title} <{ref_map[url]}_>`_"
        return match.group(0)

    new_text = link_pattern.sub(link_replacer, rst_content)

    print(f'Condensed {len(reference_definitions)} links.')
    return '\n'.join(reference_definitions) + '\n\n' + new_text


def execute_command(base_dir, settings):
    cache_key = str(settings)
    if cache_key in COMMAND_CACHE:
        return COMMAND_CACHE[cache_key]

    if isinstance(settings, dict):
        command = settings.get('command')
        replacements = settings.get('replace', None)
        columns = settings.get('columns', None)
    else:
        command = settings
        replacements = None
        columns = None

    print("running:", command)
    try:

        env = os.environ.copy()
        if columns is not None:
            env['COLUMNS'] = str(columns)

        result = subprocess.run(
            command,
            cwd=base_dir,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            env=env
        )

        # remove ANSI codes
        result = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        if replacements is not None:
            for replacements in replacements.items():
                result = re.sub(replacements[0], replacements[1], result, flags=re.MULTILINE)

        COMMAND_CACHE[cache_key] = result
        return result
    except subprocess.CalledProcessError as e:
        print(f'Warning, error executing: {command}', file=sys.stderr)
        return f"Error executing command '{command}': {e}"


def extract_headings(rst_content, skip: list | set = None, from_index: int | str = None, to_index: int | str = None):
    if skip is None:
        skip = set()

    lines = rst_content.splitlines()
    headings = []
    prev_line = ""

    for i, line in enumerate(lines):
        if re.match(r'^[=]+$', line) and prev_line.strip() and prev_line.strip() not in skip:
            headings.append((prev_line, 1))  # Top-level heading
        elif re.match(r'^[-]+$', line) and prev_line.strip() and prev_line.strip() not in skip:
            headings.append((prev_line, 2))  # Second-level heading
        prev_line = line

    if to_index is None:
        to_index = len(headings)

    if isinstance(to_index, str):
        for idx, heading in enumerate(headings):
            if heading[0] == to_index:
                to_index = idx
                break

    if from_index is None:
        from_index = 0

    if isinstance(from_index, str):
        for idx, heading in enumerate(headings):
            if heading[0] == from_index:
                from_index = idx
                break

    return headings[from_index:to_index]


def generate_toc(headings, indent: int | None = None, max_depth: int | None = None):
    toc_lines = []

    if indent is not None:
        indent = (indent * ' ')
    else:
        indent = ''

    l2_indent = indent + (' ' * 4)

    for title, level in headings:
        if max_depth is not None and level > max_depth:
            continue
        ref = f'`{title}`_'
        if level == 1:
            toc_lines.append(f"{indent}* {ref}")
        else:
            toc_lines.append(f"{l2_indent}* {ref}")

    return "\n".join(toc_lines)


def replace_directives(rst_content, directives):
    for match in reversed(list(re.finditer(r'@([a-zA-Z_][a-zA-Z0-9_]+)(\[)?', rst_content, flags=re.DOTALL))):

        directive = match.group(1)

        start = match.end()

        last_parsed_json = None
        if rst_content[start - 1] == '[':
            if rst_content[start] == '{':
                end = start
                while end <= len(rst_content):
                    try:
                        last_parsed_json = json.loads(rst_content[start:end])
                        break
                    except json.JSONDecodeError:
                        end += 1

                content_end = end
                while rst_content[content_end - 1] != ']':
                    content_end += 1
            else:
                end = start + rst_content[start:].index(']')
                content_end = end + 1

            arguments = rst_content[start:end] if last_parsed_json is None else last_parsed_json

            if directive not in directives:
                continue

            rst_content = rst_content[:match.start()] + directives[directive](arguments) + rst_content[content_end:]
        else:
            if directive not in directives:
                continue

            rst_content = rst_content[:match.start()] + directives[directive]() + rst_content[match.end():]

    return rst_content


def render_code_block(code_type, block_content):
    return f".. code-block:: {code_type}\n\n    " + block_content.replace('\n', '\n    ')


def render_templates(rst_content, filename=None):
    base_dir = os.path.dirname(filename) if filename else '.'

    if filename:
        filename = os.path.relpath(filename, os.getcwd())

    def command_replacer(command):
        command_output = execute_command(base_dir, command).strip()
        if isinstance(command, dict):
            block = command.get('block', True)
        else:
            block = True

        if block:
            return render_code_block('text', command_output)
        else:
            return command_output.strip()

    def include_replacer(include_path):
        if not isinstance(include_path, str):
            raise ValueError("INCLUDE only accepts a string argument")

        if not base_dir:
            raise ValueError("Base directory must be provided for includes.")

        full_include_path = os.path.relpath(os.path.join(base_dir, include_path), os.getcwd())

        if not os.path.exists(full_include_path):
            print(f"Warning: Included file '{full_include_path}' not found.", file=sys.stderr)
            return f".. WARNING: Missing included file: {include_path} .."
        elif filename:
            print(f'"{filename}" is including:', full_include_path)
        else:
            print(f'including:', full_include_path)

        with open(full_include_path, 'rt') as include_file:
            included_content = include_file.read().strip()

        return render_templates(included_content, filename=full_include_path)

    def example_replacer(include_path):
        if not isinstance(include_path, str):
            raise ValueError("EXAMPLE only accepts a string argument")

        if not base_dir:
            raise ValueError("Base directory must be provided for includes.")

        full_include_path = os.path.relpath(os.path.join(base_dir, include_path), os.getcwd())

        if not os.path.exists(full_include_path):
            print(f"Warning: Example file '{full_include_path}' not found.", file=sys.stderr)
            return f".. WARNING: Missing example file: {include_path} .."
        elif filename:
            print(f'"{filename}" is including example:', full_include_path)
        else:
            print(f'including example:', full_include_path)

        with open(full_include_path, 'rt') as include_file:
            included_content = include_file.read().strip()

        return render_code_block('jinja', included_content)

    rst_content = replace_directives(rst_content, {
        'INCLUDE': include_replacer,
        'EXAMPLE': example_replacer,
        'COMMAND_OUTPUT': command_replacer,
        'VERSION': lambda: VERSION,
        'REVISION': lambda: REVISION
    })

    def toc_replacer(arguments):
        indent = arguments.get('indent')
        max_depth = arguments.get('max_depth')
        if indent is not None:
            arguments.pop('indent')
        if max_depth is not None:
            arguments.pop('max_depth')

        return generate_toc(extract_headings(rst_content, **arguments), indent=indent, max_depth=max_depth)

    rst_content = replace_directives(rst_content, {
        'TOC': toc_replacer
    })

    return rst_content


command_cache_path = os.path.join(proj_dir, 'readme', 'command.cache.json')

if '--no-cache' not in sys.argv:
    if os.path.exists(command_cache_path):
        with open(command_cache_path, 'r') as cache:
            COMMAND_CACHE = json.load(cache)

# default command output width
os.environ['COLUMNS'] = '110'

input_file = os.path.join(proj_dir, 'readme', 'readme.template.rst')
output_file = os.path.join(proj_dir, 'README.rst')
with open(input_file, 'r') as file:
    content = file.read()

print('Rendering templates...')
updated_content = render_templates(content, filename=input_file)
print('Templates rendered.')

with open(output_file, 'w') as file:
    file.write(find_and_condense_links(updated_content))

with open(command_cache_path, 'w') as cache:
    json.dump(COMMAND_CACHE, cache)
