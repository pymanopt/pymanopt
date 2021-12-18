import datetime
import string

import sphinxcontrib.katex as katex

import pymanopt

# Package information
project = "Pymanopt"
author = "Jamie Townsend, Niklas Koep, Sebastian Weichwald"
copyright = f"2016-{datetime.date.today().year}, {author}"
release = version = pymanopt.__version__

# Build settings
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]
master_doc = "index"
language = None

# Output options
html_theme = "sphinx_rtd_theme"
html_logo = "logo.png"
html_show_sphinx = False
html_baseurl = "pymanopt.org"
htmlhelp_basename = "pymanoptdoc"
html_last_updated_fmt = ""

# autodoc
autodoc_default_options = {
    "member-order": "bysource",
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# nbsphinx
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        :raw-html:`<a href="https://github.com/pymanopt/pymanopt/blob/master/{{
        docname }}"><img alt="Open on GitHub"
        src="https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub"
        style="vertical-align:text-bottom"></a>`
"""

# katex
katex_version = "0.15.1"
katex_css_path = (
    f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.css"
)
katex_js_path = (
    f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.js"
)
katex_autorender_path = (
    f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/contrib/"
    "auto-render.min.js"
)
latex_macros = r"""
    \def \manM   {\mathcal{M}}
    \def \R      {\mathbb{R}}
    \def \Id     {\mathrm{Id}}
    \def \set    #1{\{#1\}}
    \def \inner  #2{\langle #1, #2 \rangle}
    \def \opt    #1{#1^\star}
    \def \sphere {\mathcal{S}}
"""
# Generate macros for boldface letters.
latex_macros += "\n".join(
    [
        r"\def \vm{letter} {{\mathbf{{{letter}}}}}".format(letter=letter)
        for letter in string.ascii_lowercase + string.ascii_uppercase
    ]
)
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = (
    "macros: {"
    + katex_macros
    + "\n"
    + r'"\\argmin":'
    + r'"\\mathop{\\operatorname{argmin}}\\limits"'
    + "}"
)
print("Defined KaTeX macros:\n{katex_macros}")
latex_elements = {"preamble": latex_macros}
