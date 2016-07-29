# pymanopt todo list

- **implement tests**

- manifolds

    - Stiefel [@118](./pymanopt/manifolds/stiefel.py#L118): simplify expressions if possible

- solvers

    - nelder_mead [@31](./pymanopt/solvers/nelder_mead.py#L31): need to decide what to do about the TR iterations
    - solvers cast to np.array before returning

- tools

    - autodiff theano and tensorflow: FixedRankEmbedded compatibility
    - autodiff autograd: move type checking outside of compiled function
    - autodiff/_theano [@82](./pymanopt/tools/autodiff/_theano.py#L82): fix theano's no Rop fallback for the product manifold/investigate whether this no Rop fallback is really necessary
