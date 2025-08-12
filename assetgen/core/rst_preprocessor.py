#!/usr/bin/env python3
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
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Set, Union, Tuple


class RSTPreprocessor:
    """A reusable RST preprocessor with directive support."""

    def __init__(self,
                 command_cache: Optional[Dict[str, str]] = None,
                 command_cache_path: Optional[Path] = None,
                 verbose: bool = True):
        """
        Initialize the RSTPreprocessor.

        :param command_cache: Pre-existing command cache dictionary
        :type command_cache: Optional[Dict[str, str]]
        :param command_cache_path: Path to save/load command cache. If provided, caching is automatically enabled.
        :type command_cache_path: Optional[Path]
        :param verbose: Whether to print status messages
        :type verbose: bool
        """
        self._command_cache = command_cache or {}
        self._command_cache_path = command_cache_path
        self.verbose = verbose
        self._custom_directives = {}

        # Load cache if path is provided and caching is enabled
        if self._command_cache_path and self._command_cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        """Load command cache from file."""
        try:
            with open(self._command_cache_path, 'r') as cache:
                self._command_cache = json.load(cache)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"Warning: Could not parse cache file: {self._command_cache_path}", file=sys.stderr)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error loading cache file: {e}", file=sys.stderr)

    def _save_cache(self):
        """Save command cache to file."""
        if self._command_cache_path:
            try:
                with open(self._command_cache_path, 'w') as cache:
                    json.dump(self._command_cache, cache)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error saving cache file: {e}", file=sys.stderr)

    def register_directive(self, name: str, handler: Callable):
        """
        Register a custom directive handler.

        :param name: The directive name (without @ prefix)
        :type name: str
        :param handler: A callable that takes arguments and returns replacement text
        :type handler: Callable
        """
        self._custom_directives[name] = handler

    def register_directives(self, directives: Dict[str, Callable]):
        """
        Register multiple custom directive handlers.

        :param directives: Dictionary mapping directive names to handlers
        :type directives: Dict[str, Callable]
        """
        self._custom_directives.update(directives)

    def clear_cache(self, patterns: Optional[List[str]] = None, regex_patterns: Optional[List[str]] = None):
        """
        Clear command cache, optionally filtering by patterns.

        :param patterns: List of substring patterns to match
        :type patterns: Optional[List[str]]
        :param regex_patterns: List of regex patterns to match
        :type regex_patterns: Optional[List[str]]
        """
        if not patterns and not regex_patterns:
            self._command_cache = {}
        else:
            def should_keep_command(key):
                if patterns and any(pattern in key for pattern in patterns):
                    return False
                if regex_patterns:
                    try:
                        return not any(re.search(pattern, key) for pattern in regex_patterns)
                    except re.error as e:
                        if self.verbose:
                            print(f"Warning: Invalid regex pattern: {e}", file=sys.stderr)
                        return True
                return True

            self._command_cache = {k: v for k, v in self._command_cache.items() if should_keep_command(k)}

    def _execute_command(self, base_dir: Path, settings: Union[str, Dict[str, Any]]) -> str:
        """
        Execute a command and optionally cache the result.

        :param base_dir: Base directory for command execution
        :type base_dir: Path
        :param settings: Command settings (string or dict)
        :type settings: Union[str, Dict[str, Any]]
        :return: Command output
        :rtype: str
        """
        cache_key = str(settings)

        if self._command_cache_path and cache_key in self._command_cache:
            return self._command_cache[cache_key]

        if isinstance(settings, dict):
            command = settings.get('command')
            replacements = settings.get('replace', None)
            columns = settings.get('columns', None)
            g_reflags = settings.get('reflags', [])
            g_count = settings.get('count', 0)
        else:
            command = settings
            replacements = None
            columns = None
            g_reflags = []
            g_count = 0

        if self.verbose:
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

            # Remove ANSI codes
            result = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)

            if replacements is not None:
                g_flags = 0
                for flag in g_reflags:
                    g_flags |= getattr(re, flag.upper())

                for pattern, replacement in replacements:
                    if isinstance(replacement, dict):
                        value = replacement.get('value', '')
                        l_reflags = replacement.get('reflags', [])
                        l_count = replacement.get('count', 0)
                    else:
                        value = replacement
                        l_reflags = []
                        l_count = 0

                    l_flags = 0
                    for flag in l_reflags:
                        l_flags |= getattr(re, flag.upper())

                    result = re.sub(
                        pattern,
                        value, result,
                        count=l_count if l_count > 0 else g_count,
                        flags=l_flags if l_flags > 0 else g_flags
                    )

            if self._command_cache_path:
                self._command_cache[cache_key] = result

            return result

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"Command failed: {e}", file=sys.stderr)
            return f"Command failed: {e}"

    def _extract_headings(self, rst_content: str, skip: Optional[Union[List[str], Set[str]]] = None,
                          from_index: Optional[Union[int, str]] = None,
                          to_index: Optional[Union[int, str]] = None) -> List[Tuple[str, int]]:
        """
        Extract headings from RST content.

        :param rst_content: RST content to parse
        :type rst_content: str
        :param skip: List or set of heading titles to skip
        :type skip: Optional[Union[List[str], Set[str]]]
        :param from_index: Start index (inclusive)
        :type from_index: Optional[Union[int, str]]
        :param to_index: End index (exclusive)
        :type to_index: Optional[Union[int, str]]
        :return: List of (heading_title, level) tuples
        :rtype: List[Tuple[str, int]]
        """
        if skip is None:
            skip = set()
        elif isinstance(skip, list):
            skip = set(skip)

        # Define heading characters and their hierarchy
        heading_chars = ['=', '-', '`', ':', "'", '"', '~', '^', '_', '*', '+', '#', '<', '>']
        heading_levels = {}
        current_level = 0

        lines = rst_content.split('\n')
        headings = []

        for i, line in enumerate(lines):
            if i == 0:
                continue  # Skip first line
            
            # Check if this line could be a heading underline
            if line.strip() and all(c == line.strip()[0] for c in line.strip()) and line.strip()[0] in heading_chars:
                prev_line = lines[i-1].strip()
                if prev_line and len(prev_line) <= len(line.strip()):
                    # This is a heading
                    heading_char = line.strip()[0]
                    
                    # Assign level based on first occurrence
                    if heading_char not in heading_levels:
                        heading_levels[heading_char] = current_level
                        current_level += 1
                    
                    level = heading_levels[heading_char]
                    
                    if prev_line not in skip:
                        headings.append((prev_line, level))

        # Apply index filtering
        if from_index is not None or to_index is not None:
            start_idx = 0
            end_idx = len(headings)
            
            if from_index is not None:
                if isinstance(from_index, str):
                    # Find by heading title
                    for idx, (title, _) in enumerate(headings):
                        if title == from_index:
                            start_idx = idx
                            break
                else:
                    start_idx = from_index
            
            if to_index is not None:
                if isinstance(to_index, str):
                    # Find by heading title
                    for idx, (title, _) in enumerate(headings):
                        if title == to_index:
                            end_idx = idx
                            break
                else:
                    end_idx = to_index
            
            headings = headings[start_idx:end_idx]

        return headings

    def _generate_toc(self, headings: List[Tuple[str, int]], indent: Optional[int] = None,
                      max_depth: Optional[int] = None) -> str:
        """
        Generate a table of contents from headings.

        :param headings: List of (heading_title, level) tuples
        :type headings: List[Tuple[str, int]]
        :param indent: Indentation level (spaces)
        :type indent: Optional[int]
        :param max_depth: Maximum depth to include
        :type max_depth: Optional[int]
        :return: Generated TOC
        :rtype: str
        """
        if not headings:
            return ""

        if indent is None:
            indent = 3

        toc_lines = []
        min_level = min(level for _, level in headings)

        for title, level in headings:
            adjusted_level = level - min_level
            if max_depth is not None and adjusted_level >= max_depth:
                continue
            
            spaces = ' ' * (adjusted_level * indent)
            toc_lines.append(f"{spaces}* `{title}`_")

        return '\n'.join(toc_lines)

    def _find_and_condense_links(self, rst_content: str) -> str:
        """
        Find and condense duplicate RST links like the original system.

        :param rst_content: RST content to process
        :type rst_content: str
        :return: Processed content with condensed links
        :rtype: str
        """
        if self.verbose:
            print('Condensing links...')

        # Pattern to match RST links with URLs: `title <url>`_
        link_pattern = re.compile(r'`([^`]+) <(https?://[^>]+)>`_')
        
        # Count occurrences of each (title, URL) pair
        from collections import defaultdict
        link_counts = defaultdict(int)
        
        for title, url in link_pattern.findall(rst_content):
            link_counts[(title, url)] += 1

        # Identify links that appear more than once
        duplicates = {key: count for key, count in link_counts.items() if count > 1}
        if not duplicates:
            if self.verbose:
                print("No duplicate links found.")
            return rst_content

        # Create reference definitions and replacement map
        ref_map = {}
        reference_definitions = []

        for (title, url), count in duplicates.items():
            ref_name = title.replace(' ', '_')
            if ref_name in ref_map.values():
                ref_name += f"_{len(ref_map)}"
            ref_map[url] = ref_name
            reference_definitions.append(f".. _{ref_name}: {url}")

        def link_replacer(match):
            title, url = match.groups()
            if url in ref_map:
                return f"`{title} <{ref_map[url]}_>`_"
            return match.group(0)

        new_text = link_pattern.sub(link_replacer, rst_content)

        if self.verbose:
            print(f'Condensed {len(reference_definitions)} links.')
        
        return '\n'.join(reference_definitions) + '\n\n' + new_text

    def _render_code_block(self, code_type: str, block_content: str) -> str:
        """
        Render a code block with proper formatting.

        :param code_type: Type of code block
        :type code_type: str
        :param block_content: Content of the code block
        :type block_content: str
        :return: Formatted code block
        :rtype: str
        """
        lines = [f".. code-block:: {code_type}", ""]
        for line in block_content.split('\n'):
            lines.append(f"    {line}")
        return '\n'.join(lines)

    def _replace_directives(self, rst_content: str, directives: Dict[str, Callable]) -> str:
        """
        Replace custom directives in RST content using the original parsing logic.

        :param rst_content: RST content to process
        :type rst_content: str
        :param directives: Dictionary of directive handlers
        :type directives: Dict[str, Callable]
        :return: Processed content
        :rtype: str
        """
        # Define directive processing order - simple replacement directives first
        simple_directives = ['VERSION', 'REVISION', 'PROJECT_DIR']
        
        # Process simple directives first (these don't take arguments)
        for directive_name in simple_directives:
            if directive_name in directives:
                handler = directives[directive_name]
                # Pattern for directives without arguments: @DIRECTIVE_NAME
                pattern = rf'@{directive_name}(?!\[)'
                
                def make_simple_replacer(h, name):
                    def replace_simple_directive(match):
                        try:
                            return h()
                        except Exception as e:
                            if self.verbose:
                                print(f"Error in directive {name}: {e}", file=sys.stderr)
                            return f"[Error in {name}: {e}]"
                    return replace_simple_directive
                
                rst_content = re.sub(pattern, make_simple_replacer(handler, directive_name), rst_content)
        
        # Now process all directives with arguments using the original parsing logic
        # This handles both simple and complex directives with proper JSON parsing
        for match in reversed(list(re.finditer(r'@([a-zA-Z_][a-zA-Z0-9_]+)(\[)?', rst_content, flags=re.DOTALL))):
            directive = match.group(1)
            
            if directive not in directives:
                continue
                
            start = match.end()
            
            last_parsed_json = None
            if len(match.groups()) > 1 and match.group(2) == '[':  # Has opening bracket
                if start < len(rst_content) and rst_content[start] == '{':
                    # Try to parse JSON incrementally
                    end = start
                    while end <= len(rst_content):
                        try:
                            last_parsed_json = json.loads(rst_content[start:end])
                            break
                        except json.JSONDecodeError:
                            end += 1
                    
                    # Find the closing bracket after the JSON
                    content_end = end
                    while content_end < len(rst_content) and rst_content[content_end] != ']':
                        content_end += 1
                    content_end += 1  # Include the closing bracket
                else:
                    # Not JSON, find the closing bracket
                    try:
                        bracket_end = start + rst_content[start:].index(']')
                        content_end = bracket_end + 1
                        end = bracket_end
                    except ValueError:
                        # No closing bracket found, skip this directive
                        continue
                
                # Get the arguments
                arguments = rst_content[start:end] if last_parsed_json is None else last_parsed_json
                
                try:
                    replacement = directives[directive](arguments)
                    rst_content = rst_content[:match.start()] + replacement + rst_content[content_end:]
                except Exception as e:
                    if self.verbose:
                        print(f"Error in directive {directive}: {e}", file=sys.stderr)
                    replacement = f"[Error in {directive}: {e}]"
                    rst_content = rst_content[:match.start()] + replacement + rst_content[content_end:]
            else:
                # No arguments, simple directive
                if directive in simple_directives:
                    # Already processed above
                    continue
                    
                try:
                    replacement = directives[directive]()
                    rst_content = rst_content[:match.start()] + replacement + rst_content[match.end():]
                except Exception as e:
                    if self.verbose:
                        print(f"Error in directive {directive}: {e}", file=sys.stderr)
                    replacement = f"[Error in {directive}: {e}]"
                    rst_content = rst_content[:match.start()] + replacement + rst_content[match.end():]

        return rst_content

    def _get_builtin_directives(self, base_dir: Path, filename: Optional[Path] = None) -> Dict[str, Callable]:
        """
        Get built-in directive handlers.

        :param base_dir: Base directory for file operations
        :type base_dir: Path
        :param filename: Current filename being processed
        :type filename: Optional[Path]
        :return: Dictionary of directive handlers
        :rtype: Dict[str, Callable]
        """
        def command_replacer(command):
            """Replace @COMMAND_OUTPUT[command] with command output."""
            command_output = self._execute_command(base_dir, command).strip()
            if isinstance(command, dict):
                block = command.get('block', True)
            else:
                block = True

            if block:
                return self._render_code_block('text', command_output)
            else:
                return command_output.strip()

        def include_replacer(include_path):
            """Replace @INCLUDE[path] with file contents."""
            if not isinstance(include_path, str):
                raise ValueError("INCLUDE only accepts a string argument")

            if not base_dir:
                raise ValueError("Base directory must be provided for includes.")

            try:
                # Handle both relative and absolute paths
                if Path(include_path).is_absolute():
                    file_path = Path(include_path)
                else:
                    file_path = base_dir / include_path
                
                if not file_path.exists():
                    if self.verbose:
                        print(f"Warning: Included file '{file_path}' not found.", file=sys.stderr)
                    return f".. WARNING: Missing included file: {include_path} .."

                if filename and self.verbose:
                    rel_filename = Path(filename).name
                    try:
                        rel_include_path = str(file_path.relative_to(Path.cwd()))
                    except ValueError:
                        # File is outside current working directory, use absolute path
                        rel_include_path = str(file_path)
                    print(f'"{rel_filename}" is including: {rel_include_path}')
                elif self.verbose:
                    print(f'including: {file_path}')
                
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                
                # Process the included content recursively
                return self._render_templates(content, filename=str(file_path))
            except Exception as e:
                if self.verbose:
                    print(f"Error including file {include_path}: {e}", file=sys.stderr)
                return f"[Error including {include_path}: {e}]"

        def example_replacer(include_path):
            """Replace @EXAMPLE[path] with code block."""
            if not isinstance(include_path, str):
                raise ValueError("EXAMPLE only accepts a string argument")

            if not base_dir:
                raise ValueError("Base directory must be provided for includes.")

            try:
                # Handle both relative and absolute paths
                if Path(include_path).is_absolute():
                    file_path = Path(include_path)
                else:
                    file_path = base_dir / include_path
                
                if not file_path.exists():
                    if self.verbose:
                        print(f"Warning: Example file '{file_path}' not found.", file=sys.stderr)
                    return f".. WARNING: Missing example file: {include_path} .."

                if filename and self.verbose:
                    rel_filename = Path(filename).name
                    try:
                        rel_include_path = str(file_path.relative_to(Path.cwd()))
                    except ValueError:
                        # File is outside current working directory, use absolute path
                        rel_include_path = str(file_path)
                    print(f'"{rel_filename}" is including example: {rel_include_path}')
                elif self.verbose:
                    print(f'including example: {file_path}')
                
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                
                # Determine code type from extension (default to jinja like original)
                extension = file_path.suffix.lower()
                code_type_map = {
                    '.py': 'python',
                    '.js': 'javascript',
                    '.ts': 'typescript',
                    '.sh': 'bash',
                    '.bat': 'batch',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',
                    '.json': 'json',
                    '.xml': 'xml',
                    '.html': 'html',
                    '.css': 'css',
                    '.rst': 'rst',
                    '.md': 'markdown',
                    '.txt': 'text',
                    '.dgen': 'jinja'
                }
                
                code_type = code_type_map.get(extension, 'text')
                return self._render_code_block(code_type, content)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading example {include_path}: {e}", file=sys.stderr)
                return f"[Error loading example {include_path}: {e}]"

        return {
            'COMMAND_OUTPUT': command_replacer,
            'INCLUDE': include_replacer,
            'EXAMPLE': example_replacer
        }

    def _render_templates(self, rst_content: str, filename: Optional[str] = None) -> str:
        """
        Render templates by replacing directives.

        :param rst_content: RST content to process
        :type rst_content: str
        :param filename: Current filename being processed
        :type filename: Optional[str]
        :return: Processed content
        :rtype: str
        """
        # Determine base directory
        if filename:
            base_dir = Path(filename).parent
        else:
            base_dir = Path.cwd()

        # First, apply built-in directives and custom directives
        all_directives = self._get_builtin_directives(base_dir, filename)
        all_directives.update(self._custom_directives)

        rst_content = self._replace_directives(rst_content, all_directives)

        # Then handle TOC directive which needs access to the processed content
        def toc_replacer(arguments):
            # Parse TOC arguments (simple implementation)
            args = {}
            if arguments:
                # This is a simplified parser - in reality you'd want more robust parsing
                pass
            
            return self._generate_toc(self._extract_headings(rst_content, **args))

        rst_content = self._replace_directives(rst_content, {'TOC': toc_replacer})

        return rst_content

    def render_file(self, input_file: Path, output_file: Path):
        """
        Render a template file.

        :param input_file: Path to the input template file
        :type input_file: Path
        :param output_file: Path to the output file
        :type output_file: Path
        """
        with open(input_file, 'r') as file:
            content = file.read()

        if self.verbose:
            print(f'Rendering template {input_file} to {output_file}...')

        updated_content = self._render_templates(content, filename=str(input_file))

        if self.verbose:
            print('Template rendered.')

        updated_content = self._find_and_condense_links(updated_content)

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as file:
            file.write(updated_content)

        self._save_cache() 