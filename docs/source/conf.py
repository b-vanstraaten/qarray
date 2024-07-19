import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'Qarray'
author = 'Barnaby van Straaten'
release = '1.3.0'

# Add any Sphinx extension module names here, as strings.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

# Paths that contain templates.
templates_path = ['_templates']

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files.
html_static_path = ['_static']
