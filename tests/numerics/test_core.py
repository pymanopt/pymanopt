import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([-4, 2]), np.array([4, 2])),
        (jnp.array([-4, 2]), jnp.array([4, 2])),
        (torch.Tensor([-4, 2]), torch.Tensor([4, 2])),
        (tf.constant([-4, 2]), tf.constant([4, 2])),
    ],
)
def test_abs(argument, expected_output):
    output = nx.abs(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (np.array([4, 2]), np.array([4, 2]), True),
        (np.array([4, 2]), np.array([2, 4]), False),
        (jnp.array([4, 2]), jnp.array([4, 2]), True),
        (jnp.array([4, 2]), jnp.array([2, 4]), False),
        (torch.Tensor([4, 2]), torch.Tensor([4, 2]), True),
        (torch.Tensor([4, 2]), torch.Tensor([2, 4]), False),
        (tf.constant([4, 2]), tf.constant([4, 2]), True),
        (tf.constant([4, 2]), tf.constant([2, 4]), False),
    ],
)
def test_allclose(argument_a, argument_b, expected_output):
    assert nx.allclose(argument_a, argument_b) == expected_output


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.log(np.array([4, 2])), np.array([4, 2])),
        (jnp.log(jnp.array([4, 2])), jnp.array([4, 2])),
        (torch.log(torch.Tensor([4, 2])), torch.Tensor([4, 2])),
        (tf.math.log(tf.constant([4.0, 2.0])), tf.constant([4.0, 2.0])),
    ],
)
def test_exp(argument, expected_output):
    output = nx.exp(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (np.array([-4, 2]), np.array([1, 3]), 2),
        (jnp.array([-4, 2]), jnp.array([1, 3]), 2),
        (torch.Tensor([-4, 2]), torch.Tensor([1, 3]), 2),
        (tf.constant([-4, 2]), tf.constant([1, 3]), 2),
    ],
)
def test_tensordot(argument_a, argument_b, expected_output):
    output = nx.tensordot(argument_a, argument_b, axes=argument_a.ndim)
    assert nx.allclose(output, expected_output)
