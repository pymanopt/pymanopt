# pymanopt todo list

- **implement tests**

- manifolds

    - Stiefel [@118](./pymanopt/manifolds/stiefel.py#L118): simplify expressions if possible

- solvers

    - nelder_mead [@32](./pymanopt/solvers/nelder_mead.py#L32): need to decide what to do about the TR iterations

- tools

    - autodiff/_theano [@84](./pymanopt/tools/autodiff/_theano.py#L84): fix theano's no Rop fallback for the product manifold/investigate whether this no Rop fallback is really necessary
