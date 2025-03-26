# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0,os.path.abspath('../src/echo21/'))
sys.path.append('../src/echo21/')


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'echo21'
copyright = '2025, Shikhar Mittal'
author = 'Shikhar Mittal'
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
#autoapi_dirs = ["../src"]
extensions = ['sphinx.ext.autodoc','sphinx.ext.mathjax','sphinx.ext.autosummary']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



def run_apidoc(_):
	from sphinx.ext.apidoc import main
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	cur_dir = os.path.abspath(os.path.dirname(__file__))
	module = os.path.join(cur_dir,"..","src/echo21")
	main(['-e', '-o', cur_dir, module, '--force'])

def setup(app):
	app.connect('builder-inited', run_apidoc)