# Pymanopt TODO List

- Development
  - Integrate flake8-docstrings (add 'docstring-convention = numpy' in [flake8]
    section of .flake8 file)
  - Improve test coverage

- Solvers:
  - NelderMead: implement finite-differences hvp approximation to use
    TrustRegions solver to compute the Karcher mean

- Manifolds:
  - Stiefel: simplify expressions of exponential map if possible

- Miscellaneous:
  - Always use "".format rather than "%s" % "bla"
  - Drop python 3.5 support so we can switch to f-strings
  - Deploy multiple versions of the documentation, e.g.
    https://github.com/jdillard/continuous-sphinx/blob/master/_themes/sphinx_rtd_theme/versions.html
  - Change color scheme of sphinx, e.g.
    https://spinningup.openai.com/en/latest/index.html
