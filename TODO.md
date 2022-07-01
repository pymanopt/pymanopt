# TODO/Roadmap

## 2.1.x:
  - attrs
  - Add 'check_hessian' function
  - Refactor optimizer implementations
  - Add complex manifolds #125, #170
  - Add L-BFGS and other quasi-Newton optimizers
  - Add patience parameter to terminate optimization if cost does not improve
    anymore #114
  - Add callback mechanism to allow for custom termination criteria #133

## 3.0.x:
  - Raise exception if dimension of manifold is 0
  - Add pep8-naming (requires breaking public API to fix all errors)
  - Make FixedRankEmbedded manifold compatible with autodiff backends
    (add weingarten map to support euclidean_to_riemannian_hessian)
  - Refactor TrustRegions implementation and update parameter names
  - Rewrite core/manifolds
    * in JAX with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
  - Revist 'reuse_line_searcher' and 'self._line_searcher' vs.
    'self.line_searcher' instance attributes
  - Rename 'orth_value' to 'restart_threshold'
  - Revisit checking docstrings with darglint if the package is more mature
