# TODO/Roadmap

## 2.x:
  - Add callback mechanism to allow for custom termination criteria #133
  - Add patience parameter to terminate optimization if cost does not improve
    anymore #114
  - Add complex manifolds #125
  - Add L-BFGS and other quasi-Newton optimizers

## 3.x:
  - Refactor optimizer implementations
  - Refactor TrustRegions implementation and update parameter names
  - Raise exception if dimension of manifold is 0
  - Rename 'orth_value' to 'restart_threshold'
  - Revist 'reuse_line_searcher' and 'self._line_searcher' vs.
    'self.line_searcher' instance attributes
  - Add pep8-naming (requires breaking public API to fix all errors)

## 4.x:
  - Make FixedRankEmbedded manifold compatible with autodiff backends
    (add weingarten map to support euclidean_to_riemannian_hessian)
  - Rewrite core/manifolds
    * in JAX with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
