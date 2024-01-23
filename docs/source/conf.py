# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

project = 'usask_arg_example'
copyright = '2024, USask-ARG'
author = 'USask-ARG'
github_url = ''

from importlib.metadata import version as get_version

release: str = get_version(project)
# for example take major/minor
version: str = ".".join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'myst_nb',
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    #"sphinx_copybutton",
    "sphinx_design",
    #"sphinx_examples",
    #"sphinx_tabs.tabs",
    #"sphinx_thebe",
    #"sphinx_togglebutton",
    #"sphinxcontrib.bibtex",
    #"sphinxext.opengraph",
    # For the kitchen sink
    "sphinx.ext.todo",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_docstring_signature = True
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'member_order': 'groupwise'
}
autoclass_content = 'both'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_css_files = ["locals.css"]

html_theme_options = {
    "github_url": github_url,
    "repository_url": github_url,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

nb_execution_timeout = 300


myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "xarray-tutorial": ("https://tutorial.xarray.dev/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None)
    # "opt_einsum": ("https://dgasmith.github.io/opt_einsum/", None),
}
