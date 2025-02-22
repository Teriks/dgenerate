import os
import sys
import ast
import inspect
import textwrap
import dgenerate.pygments
import dgenerate.promptupscalers
import sphinx.highlighting
from importlib.machinery import SourceFileLoader


def extract_constants_and_docs(module):
    file_path = inspect.getsourcefile(module)

    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    constants = {}

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            else:
                continue

            if isinstance(target, ast.Name) and target.id.isupper():
                name = target.id
                value = ast.unparse(node.value).strip() if node.value else "None"

                constants[name] = (value, None)

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pending_docstring = inspect.cleandoc(node.value.value)

            if constants:
                last_constant = list(constants.keys())[-1]
                constants[last_constant] = (constants[last_constant][0], pending_docstring)

    return constants


def convert_constants_to_rst(module, current_module, output_file):
    rst_output = [f".. py:currentmodule:: {current_module}"]

    constants_docs = extract_constants_and_docs(module)

    for name, (value, docstring) in constants_docs.items():
        rst_output.append(f"\n.. data:: {name}")
        rst_output.append(f"    :annotation: = {value}")
        if docstring:
            formatted_doc = textwrap.indent(docstring, "    ")
            rst_output.append('\n' + formatted_doc)

    with open(output_file, 'w') as file:
        file.write("\n".join(rst_output))


sys.path.insert(0, os.path.abspath('..'))

__setup = SourceFileLoader('setup_as_library', '../setup.py').load_module()

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dgenerate'
copyright = '2023, Teriks'
author = 'Teriks'
release = __setup.VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

autodoc_member_order = 'groupwise'

html_theme_options = {'navigation_depth': 4}

sphinx.highlighting.lexers['jinja'] = dgenerate.pygments.DgenerateLexer()

# generate doc files for constants that sphinx cannot parse

convert_constants_to_rst(
    dgenerate.pipelinewrapper.constants,
    'dgenerate.pipelinewrapper.constants',
    'pipelinewrapper_constants.rst'
)

convert_constants_to_rst(
    dgenerate.promptupscalers.constants,
    'dgenerate.promptupscalers.constants',
    'promptupscalers_constants.rst'
)

convert_constants_to_rst(
    dgenerate.promptweighters.constants,
    'dgenerate.promptweighters.constants',
    'promptweighters_constants.rst'
)

convert_constants_to_rst(
    dgenerate.imageprocessors.constants,
    'dgenerate.imageprocessors.constants',
    'imageprocessors_constants.rst'
)
