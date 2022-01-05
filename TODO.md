# Pymanopt TODO List

- **Improve test coverage**

- Manifolds:
  - Stiefel [@118](./pymanopt/manifolds/stiefel.py#L118): simplify expressions if possible

- Solvers
  - nelder_mead [@31](./pymanopt/solvers/nelder_mead.py#L31): need to decide what to do about the TR iterations
  - Solvers cast to np.array before returning
