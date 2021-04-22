# -*- coding: utf-8 -*-
#
# project documentation build configuration file, created by
# sphinx-quickstart on Sat Feb 22 09:35:49 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Get version number from qpimage._version file
import os.path as op
import sys

# include parent directory
pdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.insert(0, pdir)

sys.path.append(op.abspath('extensions'))

exec(open(op.join(pdir, "nrefocus/_version.py")).read())
release = version  # noqa: F821

# http://www.sphinx-doc.org/en/stable/ext/autodoc.html#confval-autodoc_member_order
# Order class attributes and functions in separate blocks
# http://www.sphinx-doc.org/en/stable/ext/autodoc.html#confval-autodoc_member_order
# Order class attributes and functions in separate blocks
autodoc_member_order = 'groupwise'
autoclass_content = 'both'

# Display link to GitHub repo instead of doc on rtfd
rst_prolog = """
:github_url: https://github.com/RI-imaging/nrefocus
"""

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
extensions = ['sphinx.ext.intersphinx',
              'sphinx.ext.autosummary',
              'sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'fancy_include',
              'sphinxcontrib.bibtex',
              ]

# specify bibtex files (required for sphinxcontrib.bibtex>=2.0)
bibtex_bibfiles = ['nrefocus.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
projectname = "nrefocus"
projectdescription = 'numerical focusing of complex wave fields'
project = projectname
year = "2015"
authors = "Paul Müller"
copyright = year + ", " + authors

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Output file base name for HTML help builder.
htmlhelp_basename = projectname+'doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', projectname+'.tex', projectname+' Documentation',
     authors, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', projectname, projectname+' Documentation',
     authors, 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', projectname, projectname+u' Documentation',
     authors, projectname,
     projectdescription,
     'Numeric'),
]


# -----------------------------------------------------------------------------
# intersphinx
# -----------------------------------------------------------------------------
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ('https://docs.python.org/', None),
    "numpy": ('http://docs.scipy.org/doc/numpy', None),
    "scipy": ('https://docs.scipy.org/doc/scipy/reference/', None),
    }
