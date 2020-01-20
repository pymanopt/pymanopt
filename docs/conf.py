import os
import runpy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
version = runpy.run_path(
    os.path.abspath(os.path.join(BASE_DIR, "..", "pymanopt", "_version.py")))

# Package information
project = "Pymanopt"
author = "Jamie Townsend, Niklas Koep, Sebastian Weichwald"
copyright = "2016-2020, {:s}".format(author)
release = version = version["__version__"]

# Build settings
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode"
]
source_suffix = ".rst"
master_doc = "index"
language = None
exclude_patterns = ["build", "*.egg*"]

# Output options
html_theme = "sphinx_rtd_theme"
html_show_sphinx = False
html_baseurl = "pymanopt.org"
htmlhelp_basename = "pymanoptdoc"
