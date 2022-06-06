# TODO/Roadmap

## 1.2.x:
  - Add 'check_hessian' function
  - Refactor optimizer implementations
  - Add complex manifolds #125, #170
  - Add JAX backend #115
  - Add L-BFGS and other quasi-Newton optimizers
  - Add patience parameter to terminate optimization if cost does not improve
    anymore #114
  - Add callback mechanism to allow for custom termination criteria
  - Add support for complex manifolds to autodiff backends

## 2.0.x:
  - Raise an exception if dimension of 'SphereSubspaceIntersection' manifold is
    0
  - Add pep8-naming (requires breaking public API to fix all errors)
  - Make FixedRankEmbedded manifold compatible with autodiff backends
    (add weingarten map to support euclidean_to_riemannian_hessian)
  - Refactor TrustRegions implementation and update parameter names
  - Rewrite core/manifolds
    * in JAX with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
  - Revist "reuse_line_searcher" and 'self._line_searcher' vs.
    'self.line_searcher'
  - Rename "orth_value" to "restart_threshold"
  - Revisit checking docstrings with darglint if the package is more mature
