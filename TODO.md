# Pymanopt TODO List

- Development
  - Integrate `flake8-import-order` into travis config
  - Integrate flake8-docstrings (add 'docstring-convention = numpy' in [flake8]
    section of .flake8 file)

- **Improve test coverage**

- Manifolds:
  - Stiefel [@118](./pymanopt/manifolds/stiefel.py#L118): simplify expressions if possible

- Solvers
  - nelder_mead [@31](./pymanopt/solvers/nelder_mead.py#L31): need to decide what to do about the TR iterations
  - Solvers cast to np.array before returning

- Tools:
  - autodiff theano and tensorflow: FixedRankEmbedded compatibility
  - autodiff autograd: move type checking outside of compiled function
  - autodiff/_theano [@82](./pymanopt/tools/autodiff/_theano.py#L82): fix theano's no Rop fallback for the product manifold/investigate whether this no Rop fallback is really necessary
  - autodiff/_autograd: fix product manifold when one or more of the manifolds is FixedRankEmbedded

- Misc:
  - always use "".format rather than "%s" % "bla"
  - set up pre-commit hook to run flake8 tests (also add this to contrib.md)
  - drop python 3.5 support so we can switch to f-strings
