import tensorflow as tf
import tensorflow.experimental.numpy as tnp

import pymanopt.numerics.core as nx


@nx.abs.register
def _(array: tf.Tensor) -> tf.Tensor:
    return tnp.abs(array)


@nx.allclose.register
def _(array_a: tf.Tensor, array_b: tf.Tensor) -> bool:
    return tnp.allclose(array_a, array_b)


@nx.exp.register
def _(array: tf.Tensor) -> tf.Tensor:
    return tnp.exp(array)


@nx.tensordot.register
def _(array_a: tf.Tensor, array_b: tf.Tensor, *, axes: int = 2) -> tf.Tensor:
    return tnp.tensordot(array_a, array_b, axes=axes)
