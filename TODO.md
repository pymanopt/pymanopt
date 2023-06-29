# TODO/Roadmap

## 2.x

- Enable "I" flake8 warnings
- Add callback mechanism to allow for custom termination criteria #133
- Add complex manifolds #125
- Add L-BFGS and other quasi-Newton optimizers

## 3.x

- Refactor TrustRegions implementation and update parameter names
- Disallow 0-dimensional manifolds (see sphere-subspace intersection)
- Rename `orth_value` to `restart_threshold`
- Revist `reuse_line_searcher` and `self._line_searcher` vs.
  `self.line_searcher` instance attributes
- Add pep8-naming (requires breaking public API to fix all errors)

## 4.x

- Make `FixedRankEmbedded` manifold compatible with autodiff backends
  (add `weingarten` map to support `euclidean_to_riemannian_hessian`)
- Rewrite core/manifolds
  - in JAX with jit support, or
  - using a backend abstraction as in `geomstats` (potentially shared with
    `geomstats`)
