## TODO/Roadmap

### 1.0.x:
  - #160
  - attrs
  - Unify argument names across manifolds
  - Return namedtupleS from solver's 'solve' method
  - Clean up solver argument names and implementations
  - Renaming:
    * transp -> vector_transport
    * rand -> random_point
    * randvec -> random_vector
    * retr -> retraction
    * proj -> projection
    * tangent -> to_tangent_space
    * zerovec -> zero_vector
    * dist -> distance
    * exp -> exponential
    * log -> logarithm
    # Undecided
    * grad, hess -> riemannian_gradient, riemannian_hvp
    * egrad, ehess -> euclidean_gradient, euclidean_hvp
    * egrad2rgrad -> euclidean_to_riemannian_gradient
    * ehess2rhess -> euclidean_to_riemannian_hvp

### 1.1.x:
  - Add complex manifolds #125
  - Add jax backend #115
  - Add L-BFGS (and other quasi-Newton) solver
  - Use forward-over-reverse mode hvps where possible

### 2.0.x:
  - Rewrite core/manifolds
    * in jax with jit support, or
    * using a backend abstraction as in geomstats (potentially shared with
      geomstats)
