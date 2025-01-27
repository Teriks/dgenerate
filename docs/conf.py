import os
import sys
from importlib.machinery import SourceFileLoader

import sphinx.highlighting

import dgenerate.pygments

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
