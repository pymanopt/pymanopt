import datetime
import string

import sphinxcontrib.katex as katex

import pymanopt


def setup(app):
    def config_inited(app, config):
        doc_version = config.doc_version
        if doc_version in ["latest", "stable"]:
            config.version = (
                config.release
            ) = f"{doc_version} ({config.version})"
        config.html_context["doc_version"] = doc_version
        config.html_context["doc_versions"] = (
            config.doc_versions.split(",") or []
        )
        print(f"Generating documentation for {config.version}")

    app.add_config_value(
        "doc_version", default=pymanopt.__version__, rebuild="html", types=str
    )
    app.add_config_value("doc_versions", default="", rebuild="html", types=str)
    app.connect("config-inited", config_inited)


# Package information
project = "Pymanopt"
author = "Jamie Townsend, Niklas Koep, Sebastian Weichwald"
copyright = f"2016-{datetime.date.today().year}, {author}"
version = release = pymanopt.__version__


# Build settings
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]
master_doc = "index"
language = None

# Output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_style = "css/style.css"
html_logo = "logo.png"
html_show_sphinx = False
html_baseurl = "pymanopt.org"
htmlhelp_basename = "pymanoptdoc"
html_last_updated_fmt = ""

# Doc version sidebar
templates_path = ["_templates"]

# autodoc
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_default_options = {
    "member-order": "bysource",
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

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
# TODO(nkoep): Move this macro generation to its own module.
latex_macros = r"""
    \def \manM    {\mathcal{M}}
    \def \R       {\mathbb{R}}
    \def \C       {\mathbb{C}}
    \def \O       {\mathrm{O}}
    \def \SO      {\mathrm{SO}}
    \def \U       {\mathrm{U}}
    \def \E       {\mathcal{E}}
    \def \Skew    {\mathrm{Skew}}
    \def \St      {\mathrm{St}}
    \def \Id      {\mathrm{Id}}
    \def \set     #1{\{#1\}}
    \def \inner   #2{\langle #1, #2 \rangle}
    \def \opt     #1{#1^\star}
    \def \sphere  {\mathcal{S}}
    \def \transp  #1{#1^\top}
    \def \adj     #1{#1^*}
    \def \conj    #1{\overline{#1}}
    \def \norm    #1{\|#1\|}
    \def \abs     #1{|#1|}
    \def \parens  #1{\left(#1\right)}
    \def \tangent #1{\mathrm{T}_{#1}}
    \def \Re      {\mathfrak{Re}}
"""
# Generate macros for boldface letters.
latex_macros += "\n".join(
    [
        r"\def \vm{letter} {{\mathbf{{{letter}}}}}".format(letter=letter)
        for letter in string.ascii_lowercase + string.ascii_uppercase
    ]
    + [r"\def \vmOmega {\mathbf{\Omega}}"]
)
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = (
    "macros: {"
    + katex_macros
    + "\n"
    + r'"\\argmin":'
    + r'"\\mathop{\\operatorname{argmin}}\\limits"'
    + ",\n"
    + r'"\\arccosh":'
    + r'"\\operatorname{arccosh}"'
    + ",\n"
    + r'"\\dist":'
    + r'"\\operatorname{dist}"'
    + ",\n"
    + r'"\\tr":'
    + r'"\\operatorname{tr}"'
    + "}"
)
print(f"Defined KaTeX macros:\n{katex_options}")
latex_elements = {"preamble": latex_macros}
