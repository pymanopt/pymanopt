import typing

import torch

import pymanopt.numerics.core as nx


@nx.abs.register
def _(array: torch.Tensor) -> torch.Tensor:
    return torch.abs(array)


@nx.all.register
def _(array: torch.Tensor) -> torch.Tensor:
    return torch.all(array)


@nx.allclose.register
def _(
    array_a: torch.Tensor, array_b: typing.Union[torch.Tensor, float, int]
) -> bool:
    # PyTorch does not automatically coerce values to tensors.
    if isinstance(array_b, (float, int)):
        array_b = torch.Tensor([array_b])
    return torch.allclose(array_a, array_b)


@nx.exp.register
def _(array: torch.Tensor) -> torch.Tensor:
    return torch.exp(array)


@nx.tensordot.register
def _(
    array_a: torch.Tensor, array_b: torch.Tensor, *, axes: int = 2
) -> torch.Tensor:
    return torch.tensordot(array_a, array_b, dims=axes)


@nx.tanh.register
def _(array: torch.Tensor) -> torch.Tensor:
    return torch.tanh(array)
