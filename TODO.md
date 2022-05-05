# TODO/Roadmap

## 1.0:
  - Clean up all docstrings of manifolds

## 1.0.x:
  - Use weingarten map for oblique manifold, Stiefel and fixed rank matrices
  - Change "beta_rule" of CG optimizer to internal enum representation

## 1.1.x:
  - For Riemannian submanifolds of Euclidean space, it is acceptable to
    transport simply by orthogonal projection of the tangent vector translated
    in the ambient space. For this, 'RiemannianSubmanifold' would require a
    generic 'tangent_to_ambient' method. See 'FixedRankEmbedded'.
  - Add default implementation for `to_tangent_space`?
  - attrs
  - Refactor optimizer implementations
  - Add complex manifolds #125
  - Add jax backend #115
  - Add L-BFGS and other quasi-Newton optimizers
  - Use forward-over-reverse mode hvps where possible
  - Add patience parameter to terminate optimization if cost does not improve
    anymore #114
  - Add constant step size line search method
  - Add callback mechanism to allow for custom termination criteria

## 2.0.x:
  - Refactor TrustRegions implementation and update parameter names
  - Rewrite core/manifolds
    * in jax with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
  - Revist "reuse_line_searcher"
