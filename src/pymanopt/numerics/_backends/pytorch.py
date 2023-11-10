import torch
from typing import Sequence

import pymanopt.numerics as nx

tensor_like = torch.Tensor


@nx.abs.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.abs(tensor)


@nx.all.register
def _(tensor: tensor_like) -> bool:
    return torch.all(tensor)


@nx.allclose.register
def _(tensor_a: tensor_like, tensor_b: tensor_like) -> bool:
    if type(tensor_b) is not torch.Tensor:
        tensor_b = torch.tensor(tensor_b).to(tensor_a)
    return torch.allclose(tensor_a, tensor_b)


@nx.any.register
def _(tensor: tensor_like) -> bool:
    return torch.any(tensor)


@nx.arccos.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.acos(tensor)


@nx.arccosh.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.acosh(tensor)


@nx.arctan.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.atan(tensor)


@nx.arctanh.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.atanh(tensor)


@nx.argmin.register
def _(tensor: tensor_like) -> int:
    return torch.argmin(tensor)


@nx.array.register
def _(tensor: tensor_like) -> torch.Tensor:
    return tensor


@nx.assert_allclose.register
def _(tensor_a: tensor_like, tensor_b: tensor_like) -> None:
    tensor_a = tensor_a.detach().cpu().numpy()
    if isinstance(tensor_b, torch.Tensor):
        tensor_b = tensor_b.detach().cpu().numpy()
    nx.assert_allclose(tensor_a, tensor_b)


@nx.assert_almost_equal.register
def _(tensor_a: tensor_like, tensor_b: tensor_like) -> None:
    tensor_a = tensor_a.detach().cpu().numpy()
    if isinstance(tensor_b, torch.Tensor):
        tensor_b = tensor_b.detach().cpu().numpy()
    nx.assert_almost_equal(tensor_a, tensor_b)


@nx.assert_array_almost_equal.register
def _(tensor_a: tensor_like, tensor_b: tensor_like) -> None:
    tensor_a = tensor_a.detach().cpu().numpy()
    if isinstance(tensor_b, torch.Tensor):
        tensor_b = tensor_b.detach().cpu().numpy()
    nx.assert_array_almost_equal(tensor_a, tensor_b)


@nx.block.register
def _(tensors: Sequence[tensor_like]) -> torch.Tensor:
    return torch.cat(tensors)


@nx.conjugate.register
def _(tensor: tensor_like) -> tensor_like:
    return torch.conj(tensor)


@nx.cos.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.cos(tensor)


@nx.diag.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.diag(tensor)


@nx.diagonal.register
def _(tensor: tensor_like, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
    return torch.diagonal(tensor, dim1=axis1, dim2=axis2)


@nx.exp.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.exp(tensor)


@nx.expand_dims.register
def _(
    tensor: tensor_like,
    axis: int | tuple[int, ...] = None
) -> torch.Tensor:
    if isinstance(axis, int):
        axis = [axis]

    # Normalize axis values for negative indices
    positive_axis = list()
    for ax in axis:
        if ax >= 0:
            positive_axis.append(ax)
    negative_axis = list()
    for ax in axis:
        if ax < 0:
            negative_axis.append(ax)

    # Sort the axis list
    positive_axis.sort()
    negative_axis.sort()
    negative_axis = negative_axis[::-1]

    for i in range(len(positive_axis)):
        tensor = torch.unsqueeze(tensor, dim=positive_axis[i])

    for i in range(len(negative_axis)):
        dim = tensor.ndim + negative_axis[i] + 1
        tensor = torch.unsqueeze(tensor, dim=dim)

    return tensor


@nx.hstack.register
def _(tensors: Sequence[tensor_like]) -> torch.Tensor:
    return torch.hstack(tensors)


@nx.iscomplexobj.register
def _(tensor: tensor_like) -> bool:
    return torch.is_complex(tensor)


@nx.isnan.register
def _(tensor: tensor_like) -> bool:
    return torch.isnan(tensor)


@nx.isrealobj.register
def _(tensor: tensor_like) -> bool:
    return torch.isreal(tensor).all()


@nx.log.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.log(tensor)


@nx.linalg.cholesky.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.linalg.cholesky(tensor)


@nx.linalg.det.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.det(tensor)


@nx.linalg.eigh.register
def _(tensor: tensor_like) -> tuple[torch.Tensor, torch.Tensor]:
    eigval, eigvec = torch.linalg.eigh(tensor)
    return eigval, eigvec


@nx.linalg.expm.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.matrix_exp(tensor)


@nx.linalg.inv.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.inverse(tensor)


@nx.linalg.logm.register
def _(tensor: tensor_like) -> torch.Tensor:
    # logm is not implemented in PyTorch
    # see: https://github.com/pytorch/pytorch/issues/9983
    # hence we use the SciPy implementation
    numpy_tensor = tensor.detach().numpy()
    return torch.tensor(nx.linalg.logm(numpy_tensor))


@nx.linalg.matrix_rank.register
def _(tensor: tensor_like) -> int:
    return int(torch.linalg.matrix_rank(tensor))


@nx.linalg.norm.register
def _(
    tensor: tensor_like,
    *args: tuple,
    **kwargs: dict,
) -> float:
    return torch.linalg.norm(tensor, *args, **kwargs)


@nx.linalg.qr.register
def _(tensor: tensor_like) -> tuple[torch.Tensor, torch.Tensor]:
    q, r = torch.linalg.qr(tensor)
    return q, r


@nx.linalg.solve.register
def _(tensor_a: tensor_like, tensor_b: tensor_like) -> torch.Tensor:
    return torch.linalg.inv(tensor_a) @ tensor_b


@nx.linalg.solve_continuous_lyapunov.register
def _(tensor_a: tensor_like, tensor_q: tensor_like) -> torch.Tensor:
    # solve_continuous_lyapunov is not implemented in PyTorch
    # we use the SciPy implementation
    numpy_a = tensor_a.detach().numpy()
    numpy_q = tensor_q.detach().numpy()
    return torch.tensor(nx.linalg.solve_continuous_lyapunov(
        numpy_a, numpy_q))


@nx.linalg.svd.register
def _(
    tensor: tensor_like,
    *args: tuple,
    **kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, V = torch.linalg.svd(tensor, *args, **kwargs)
    return U, S, V


@nx.prod.register
def _(tensor: tensor_like) -> float:
    return torch.prod(tensor)


@nx.real.register
def _(tensor: tensor_like) -> tensor_like:
    return torch.real(tensor)


@nx.sin.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.sin(tensor)


@nx.sinc.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.sinc(tensor)


@nx.sort.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.sort(tensor).values


@nx.spacing.register
def _(tensor: tensor_like) -> float:
    return torch.spacing(tensor)


@nx.sqrt.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.sqrt(tensor)


@nx.squeeze.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.squeeze(tensor)


@nx.sum.register
def _(
    tensor: tensor_like,
    *args: tuple,
    **kwargs: dict
) -> torch.Tensor:
    return torch.sum(tensor, *args, **kwargs)


@nx.tanh.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.tanh(tensor)


@nx.tensordot.register
def _(
    tensor_a: tensor_like,
    tensor_b: tensor_like,
    *args: tuple,
    **kwargs: dict
) -> torch.Tensor:
    if 'axes' in kwargs:
        axes = kwargs.pop('axes')
        kwargs['dims'] = axes
    return torch.tensordot(tensor_a, tensor_b, *args, **kwargs)


@nx.tile.register
def _(tensor: tensor_like, reps: int | tuple[int, ...]) -> torch.Tensor:
    return torch.tile(tensor, reps)


@nx.trace.register
def _(
    tensor: tensor_like,
    *args: tuple,
    **kwargs: dict
) -> float:
    return torch.trace(tensor, *args, **kwargs)


@nx.transpose.register
def _(tensor: tensor_like, axes: tuple[int, ...] | None = None) -> torch.Tensor:
    if axes is None:
        return tensor
    return torch.transpose(tensor, dim0=axes[-2], dim1=axes[-1])


@nx.vstack.register
def _(tensors: Sequence[tensor_like]) -> torch.Tensor:
    return torch.vstack(tensors)


@nx.where.register
def _(tensor: tensor_like) -> torch.Tensor:
    return torch.as_tensor(tensor).nonzero()


@nx.zeros_like.register
def _(array: tensor_like) -> torch.Tensor:
    return torch.zeros_like(array)
