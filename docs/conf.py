# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from docutils import nodes
from docutils.parsers.rst import roles

sys.path.insert(0, os.path.abspath(".."))

project = "Gxm"
copyright = "2025, huterguier"
author = "huterguier"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/gxm.png"
html_favicon = "_static/favicon.png"

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # for type hints
    "myst_nb",
]

autosummary_generate = True
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

myst_enable_extensions = ["dollarmath"]


def string_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.raw(
        "",
        f'<code class="docutils highlight-default literal notranslate"><span class="highlight"><span class="s2">"{text}"</span></span></code>',
        format="html",
    )
    return [node], []


roles.register_canonical_role("string", string_role)
