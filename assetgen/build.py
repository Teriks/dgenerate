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

import argparse
import os
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

from assetgen.core.rst_preprocessor import RSTPreprocessor
from assetgen.builders.readme_builder import ReadmeBuilder
from assetgen.builders.docs_builder import DocsBuilder
from assetgen.builders.console_schema_builder import ConsoleSchemaBuilder
from assetgen.builders.helsinki_nlp_translation_map_builder import HelsinkiNLPTranslationMapBuilder
from assetgen.builders.hf_configs_builder import HfConfigsBuilder


def get_git_revision():
    """
    Get the current git revision (tag or branch name).

    :return: Tag name if on a tag, branch name if on a branch, or 'main' if git is not available
    :rtype: str
    """
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


def setup_argument_parser():
    """
    Set up and return the argument parser.

    :return: Configured argument parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Build documentation and README files from templates'
    )

    parser.add_argument(
        '--target', 
        choices=['readme', 'docs', 'console-schemas', 'helsinki-nlp-translation-map', 'hf-configs', 'all'],
        default='all',
        help='What to build (default: all)'
    )

    parser.add_argument(
        '--skip',
        choices=['readme', 'docs', 'console-schemas', 'helsinki-nlp-translation-map', 'hf-configs'],
        nargs='*',
        metavar='TARGET',
        help='Skip specific targets when building (can specify multiple targets)'
    )

    parser.add_argument(
        '--no-command-cache', nargs='*', metavar='COMMAND_PATTERN',
        help='Disable command cache (RST Templating). If patterns are provided, '
             'remove matching commands from cache'
    )

    parser.add_argument(
        '--no-command-cache-regex', nargs='*', metavar='REGEX_PATTERN',
        help='Disable command cache using regex patterns (RST Templating). '
             'If patterns are provided, remove matching commands from cache'
    )

    return parser


def should_build_target(target_name, args):
    """
    Check if a target should be built based on the --target and --skip arguments.
    
    :param target_name: Name of the target to check
    :type target_name: str
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: True if the target should be built, False otherwise
    :rtype: bool
    """
    # Check if target is selected by --target argument
    target_selected = args.target == 'all' or args.target == target_name
    
    # Check if target is explicitly skipped
    target_skipped = args.skip and target_name in args.skip
    
    return target_selected and not target_skipped


def main():
    """Main entry point for the build script."""

    # Get project paths and info
    script_path = Path(__file__).absolute().parent
    project_dir = (script_path / '..').resolve()

    # Load setup module to get VERSION
    setup = SourceFileLoader(
        'setup_as_library', str(project_dir / 'setup.py')).load_module()

    VERSION = setup.VERSION
    REVISION = get_git_revision()

    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Determine if caching should be disabled completely
    disable_cache_completely = (
            (args.no_command_cache is not None and not args.no_command_cache) or
            (args.no_command_cache_regex is not None and not args.no_command_cache_regex)
    )

    # Determine cache file path only if caching is enabled
    if disable_cache_completely:
        command_cache_path = None
    else:
        command_cache_path = project_dir / 'assetgen' / 'cache' / 'command.cache.json'
        command_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Create template builder - always verbose
    preprocessor = RSTPreprocessor(
        command_cache_path=command_cache_path,
        verbose=True  # Always verbose
    )

    # Handle cache filtering (if caching is enabled but patterns are provided)
    if command_cache_path and (args.no_command_cache or args.no_command_cache_regex):
        preprocessor.clear_cache(patterns=args.no_command_cache, regex_patterns=args.no_command_cache_regex)

    # Register project-specific directives
    preprocessor.register_directives({
        'VERSION': lambda: VERSION,
        'REVISION': lambda: REVISION,
        'PROJECT_DIR': lambda: project_dir.as_posix()
    })

    # Set default command output width
    os.environ['COLUMNS'] = '110'

    # Build targets
    if should_build_target('readme', args):
        print("Building README...")
        readme_builder = ReadmeBuilder(preprocessor, project_dir)
        readme_builder.build()
    else:
        print("Skipping README build")
        
    if should_build_target('docs', args):
        print("Building docs...")
        docs_builder = DocsBuilder(preprocessor, project_dir)
        docs_builder.build()
    else:
        print("Skipping docs build")
    
    if should_build_target('console-schemas', args):
        print("Building console schemas...")
        schema_builder = ConsoleSchemaBuilder(project_dir)
        schema_builder.build()
    else:
        print("Skipping console schemas build")
    
    if should_build_target('helsinki-nlp-translation-map', args):
        print("Building Helsinki NLP translation map...")
        translation_map_builder = HelsinkiNLPTranslationMapBuilder(project_dir)
        translation_map_builder.build()
    else:
        print("Skipping Helsinki NLP translation map build")
    
    if should_build_target('hf-configs', args):
        print("Building HF configs...")
        hf_configs_builder = HfConfigsBuilder(project_dir)
        hf_configs_builder.build()
    else:
        print("Skipping HF configs build")

    print("Build completed successfully!")


if __name__ == '__main__':
    main() 