## TODO/Roadmap

### Issues

### 1.0:
  - Return 'Result' object from optimizer's 'run' method
  - Renaming:
    # Long forms
    * rand -> random_point
    * randvec -> random_vector
    # Undecided
    * grad, hess -> riemannian_gradient, riemannian_hvp
    * egrad, ehess -> euclidean_gradient, euclidean_hvp
    * egrad2rgrad -> euclidean_to_riemannian_gradient
    * ehess2rhess -> euclidean_to_riemannian_hvp

### 1.0.x:
  - Use weingarten map for oblique manifold and Stiefel
  - Change "beta_rule" of CG optimizer to internal enum representation

### 1.1.x:
  - For Riemannian submanifolds of Euclidean space, it is acceptable to
    transport simply by orthogonal projection of the tangent vector translated
    in the ambient space. For this, 'EuclideanEmbeddedSubmanifold' would
    require a generic 'tangent_to_ambient' method. See 'FixedRankEmbedded'.
  - Add default implementation for tangent?
  - attrs
  - Refactor optimizer implementations
  - Add complex manifolds #125
  - Add jax backend #115
  - Add L-BFGS and other quasi-Newton optimizers
  - Use forward-over-reverse mode hvps where possible
  - Add patience parameter to terminate optimization if
    cost does not improve anymore #114
  - Add constant step size line search method
  - Add callback mechanism to allow for custom termination criteria

### 2.0.x:
  - Refactor TrustRegions implementation and update parameter names
  - Rewrite core/manifolds
    * in jax with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
  - Revist "reuse_line_searcher"
