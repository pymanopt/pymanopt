# pymanopt todo list

- tools

    - autodiff/_autograd
        - Need to change to hessian_vector_product, but not working yet, due to an issue with autograd. Issue has been fixed but they haven't merged the fix into master branch yet, apparently will happen very soon (see [here](https://github.com/HIPS/autograd/issues/86)). Have implemented our own fix for now, should be removed and replaced with theirs at some point in future.

- manifolds

    - implement product manifold

    - stiefel
        - [todo @121 (simplify expressions)](./pymanopt/manifolds/stiefel.py#L121) Not sure if this is really possible though...

    - Implement tests for manifolds

- solvers

    - implement information logging during optimisation

    - nelder_mead
        [todo @37](./pymanopt/solvers/nelder_mead.py#L37) need to decide what to do about the TR iterations

    - trust_regions
        - [todo @97 (implement value checks)](./pymanopt/solvers/trust_regions.py#L97)

    - Implement tests for solvers

- autodiff
    - add tensorflow support
