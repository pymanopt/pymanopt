import jax.numpy as jnp

import pymanopt.numerics.core as nx


@nx.abs.register
def _(array: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(array)


@nx.allclose.register
def _(array_a: jnp.ndarray, array_b: jnp.ndarray) -> bool:
    return jnp.allclose(array_a, array_b)


@nx.exp.register
def _(array: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(array)


@nx.tensordot.register
def _(
    array_a: jnp.ndarray, array_b: jnp.ndarray, *, axes: int = 2
) -> jnp.ndarray:
    return jnp.tensordot(array_a, array_b, axes=axes)
