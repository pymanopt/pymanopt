import functools
from numbers import Number
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy.linalg
import torch
from torch import autograd

from pymanopt.backends.backend import Backend, TupleOrList
from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


def elementary_math_function(
    f: Callable[["PytorchBackend", torch.Tensor], torch.Tensor],
) -> Callable[
    ["PytorchBackend", Union[torch.Tensor, float, complex]],
    Union[torch.Tensor, float, complex],
]:
    def inner(
        self, x: Union[torch.Tensor, float, complex]
    ) -> Union[torch.Tensor, float, complex]:
        if isinstance(x, torch.Tensor):
            return f(self, x)
        else:
            return f(self, self.array(x)).item()

    inner.__doc__ = f.__doc__
    inner.__name__ = f.__name__
    return inner


class PytorchBackend(Backend):
    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    array_t = torch.Tensor  # type: ignore
    _dtype: torch.dtype

    def __init__(self, dtype=torch.float64):
        assert dtype in {
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        }
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_dtype_real(self):
        return self.dtype in {torch.float32, torch.float64}

    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return torch.tensor([1.0]).dtype

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return torch.tensor([1j]).dtype

    def to_real_backend(self) -> "PytorchBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == torch.complex64:
            return PytorchBackend(dtype=torch.float32)
        elif self.dtype == torch.complex128:
            return PytorchBackend(dtype=torch.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "PytorchBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == torch.float32:
            return PytorchBackend(dtype=torch.complex64)
        elif self.dtype == torch.float64:
            return PytorchBackend(dtype=torch.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def _complex_to_real_dtype(
        self, complex_dtype: torch.dtype
    ) -> torch.dtype:
        if complex_dtype == torch.complex64:
            return torch.float32
        elif complex_dtype == torch.complex128:
            return torch.float64
        else:
            raise ValueError(f"Provided dtype {complex_dtype} is not complex.")

    def __repr__(self):
        return f"PytorchBackend(dtype={self.dtype})"

    ##############################################################################
    # Autodiff methods
    ##############################################################################
    def prepare_function(self, function):
        @functools.wraps(function)
        def wrapper(*args):
            return function(*args)

        return wrapper

    def _sanitize_gradient(self, tensor):
        if tensor.grad is None:
            return torch.zeros_like(tensor)
        return tensor.grad

    def _sanitize_gradients(self, tensors):
        return list(map(self._sanitize_gradient, tensors))

    def generate_gradient_operator(self, function, num_arguments):
        def gradient(*args):
            arguments = [arg.requires_grad_() for arg in args]
            function(*arguments).backward(retain_graph=True)
            return self._sanitize_gradients(arguments)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    def generate_hessian_operator(self, function, num_arguments):
        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(args)
            arguments = [argument.requires_grad_() for argument in arguments]
            gradients = autograd.grad(
                function(*arguments),
                arguments,
                create_graph=True,
                allow_unused=True,
                retain_graph=True,
            )
            dot_product: torch.Tensor = torch.tensor(0.0)
            for gradient, vector in zip(gradients, vectors):
                dot_product += torch.tensordot(
                    gradient.conj(), vector, dims=gradient.ndim
                ).real
            dot_product.backward(retain_graph=True)
            return self._sanitize_gradients(arguments)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product
            )
        return hessian_vector_product

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @elementary_math_function
    def abs(self, array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    def all(self, array: torch.Tensor) -> bool:
        return bool(torch.all(array).item())

    def allclose(
        self,
        array_a: torch.Tensor,
        array_b: torch.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> bool:
        if not isinstance(array_a, torch.Tensor):
            array_a = self.array(array_a)
        if not isinstance(array_b, torch.Tensor):
            array_b = self.array(array_b)
        if array_a.numel() == 1 and array_b.numel() != 1:
            array_a = torch.full_like(array_b, array_a.item())
        elif array_b.numel() == 1 and array_a.numel() != 1:
            array_b = torch.full_like(array_a, array_b.item())
        elif array_a.shape != array_b.shape:
            raise ValueError(
                f"Arrays with shapes {array_a.shape} and {array_b.shape} "
                "cannot be compared."
            )
        return torch.allclose(array_a, array_b, rtol=rtol, atol=atol)

    def any(self, array: torch.Tensor) -> bool:
        return bool(torch.any(array).item())

    def arange(
        self,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        if stop is None:
            return torch.arange(start)
        if step is None:
            return torch.arange(start, stop)
        return torch.arange(start, stop, step)

    @elementary_math_function
    def arccos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccos(array)

    @elementary_math_function
    def arccosh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccosh(array)

    @elementary_math_function
    def arctan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctan(array)

    @elementary_math_function
    def arctanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctanh(array)

    def argmin(self, array: torch.Tensor):
        return torch.argmin(array)

    def argsort(self, array: torch.Tensor):
        return torch.argsort(array)

    def array(self, array: Any) -> torch.Tensor:  # type: ignore
        if self.is_dtype_real and (
            (
                isinstance(array, torch.Tensor)
                and self.iscomplexobj(array)
                and not self.allclose(torch.imag(array), 0.0)
            )
            or (
                isinstance(array, np.ndarray)
                and np.iscomplexobj(array)
                and not np.allclose(np.imag(array), 0.0, atol=1e-5)
            )
        ):
            raise ValueError(
                "Complex Tensors should not be converted to real tensors"
            )
        return torch.as_tensor(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: torch.Tensor,
        array_b: torch.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        def max_abs(x):
            return torch.max(torch.abs(x))

        array_a, array_b = self.array(array_a), self.array(array_b)
        assert self.allclose(array_a, array_b, rtol, atol), (
            "Arrays are not almost equal.\n"
            f"Max absolute difference: {max_abs(array_a - array_b)}"
            f" (atol={atol})\n"
            "Max relative difference: "
            f"{max_abs(array_a - array_b) / max_abs(array_b)}"
            f" (rtol={rtol})"
        )

    def concatenate(
        self, arrays: TupleOrList[torch.Tensor], axis: int = 0
    ) -> torch.Tensor:
        return torch.cat(arrays, axis)

    @elementary_math_function
    def conjugate(self, array: torch.Tensor) -> torch.Tensor:
        return torch.conj(array)

    @elementary_math_function
    def cos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    def diag(self, array: torch.Tensor) -> torch.Tensor:
        return torch.diag(array)

    def diagonal(
        self, array: torch.Tensor, axis1: int, axis2: int
    ) -> torch.Tensor:
        return torch.diagonal(array, dim1=axis1, dim2=axis2)

    def eps(self) -> float:
        return torch.finfo(self.dtype).eps

    @elementary_math_function
    def exp(self, array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    def expand_dims(self, array: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.unsqueeze(array, dim=axis)
        # if isinstance(axis, int):
        #     axis = [axis]
        #
        # # Normalize axis values for negative indices
        # positive_axis = list()
        # for ax in axis:
        #     if ax >= 0:
        #         positive_axis.append(ax)
        # negative_axis = list()
        # for ax in axis:
        #     if ax < 0:
        #         negative_axis.append(ax)
        #
        # # Sort the axis list
        # positive_axis.sort()
        # negative_axis.sort()
        # negative_axis = negative_axis[::-1]
        #
        # for i in range(len(positive_axis)):
        #     tensor = torch.unsqueeze(tensor, dim=positive_axis[i])
        #
        # for i in range(len(negative_axis)):
        #     dim = tensor.ndim + negative_axis[i] + 1
        #     tensor = torch.unsqueeze(tensor, dim=dim)
        #
        # return tensor

    def eye(self, size: int) -> torch.Tensor:
        return torch.eye(size, dtype=self.dtype)

    def hstack(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        return torch.hstack(arrays)

    def imag(self, array: torch.Tensor) -> torch.Tensor:
        return (
            torch.imag(array)
            if self.iscomplexobj(array)
            else self.to_real_backend().zeros_like(array)
        )

    def iscomplexobj(self, array: torch.Tensor) -> bool:
        return array.dtype in {torch.complex64, torch.complex128}

    @elementary_math_function
    def isnan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    def isrealobj(self, array: torch.Tensor) -> bool:
        return array.dtype in {torch.float32, torch.float64}

    def linalg_cholesky(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(array)

    def linalg_det(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(array)

    def linalg_eigh(
        self, array: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: torch.Tensor, array_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if array_y is None:
            return torch.linalg.eigvalsh(array_x)
        else:
            return torch.from_numpy(
                np.vectorize(
                    pyfunc=scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
                )(
                    array_x.cpu().detach().numpy(),
                    array_y.cpu().detach().numpy(),
                )
            )

    def linalg_expm(
        self, array: torch.Tensor, symmetric: bool = False
    ) -> torch.Tensor:
        if not symmetric:
            return torch.matrix_exp(array)

        w, v = torch.linalg.eigh(array)
        w = torch.unsqueeze(torch.exp(w), dim=-1)
        expmA = v @ (w * self.conjugate_transpose(v))
        if torch.is_complex(array):
            return torch.real(expmA)
        return expmA

    def linalg_inv(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(array)

    def linalg_matrix_rank(self, array: torch.Tensor) -> int:
        return int(torch.linalg.matrix_rank(array))

    def linalg_logm(
        self, array: torch.Tensor, positive_definite: bool = False
    ) -> torch.Tensor:
        if not positive_definite:
            # logm is not implemented in PyTorch
            # see: https://github.com/pytorch/pytorch/issues/9983
            # hence we use the SciPy implementation
            logm_function = np.vectorize(
                scipy.linalg.logm, signature="(m,m)->(m,m)"
            )
            return self.array(logm_function(array.cpu().detach().numpy()))

        w, v = torch.linalg.eigh(array)
        w = torch.unsqueeze(torch.log(w), dim=-1)
        logmA = v @ (w * self.conjugate_transpose(v))
        if self.isrealobj(array):
            return torch.real(logmA)
        return logmA

    def linalg_norm(
        self,
        array: torch.Tensor,
        ord: Union[int, Literal["fro"], None] = None,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        norm = torch.linalg.norm(array, ord=ord, axis=axis, keepdim=keepdims)
        if axis is None:
            return norm.item()
        return norm

    def linalg_qr(
        self, array: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, r = torch.linalg.qr(array)
        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = torch.diagonal(r, dim1=-2, dim2=-1).clone()
        s[s == 0] = 1
        s = s / torch.abs(s)
        s = torch.unsqueeze(s, dim=-1)
        # normalize q and r to have either 1 or unit-modulus on the diagonal of r
        q = q * torch.transpose(s, -2, -1)
        r = r * torch.conj(s)
        return q, r

    def linalg_solve(
        self, array_a: torch.Tensor, array_b: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: torch.Tensor, array_q: torch.Tensor
    ) -> torch.Tensor:
        # solve_continuous_lyapunov is not implemented in PyTorch so we use the
        # SciPy implementation
        return torch.tensor(
            scipy.linalg.solve_continuous_lyapunov(
                array_a.cpu().detach().numpy(), array_q.cpu().detach().numpy()
            )
        )

    def linalg_svd(
        self,
        array: torch.Tensor,
        full_matrices: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(array, full_matrices=full_matrices)

    def linalg_svdvals(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(array)

    @elementary_math_function
    def log(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @elementary_math_function
    def log10(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log10(array)

    def logspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        return torch.logspace(start, stop, num, dtype=self.dtype)

    def ndim(self, array: torch.Tensor) -> int:
        return array.ndim

    def ones(self, shape: TupleOrList[int]) -> torch.Tensor:
        return torch.ones(shape, dtype=self.dtype)

    def ones_bool(self, shape: TupleOrList[int]) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.bool)

    def polyfit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        deg: int = 1,
        full: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert x.ndim == y.ndim == 1
        x = torch.stack([x**i for i in range(deg + 1)], dim=-1)
        p = torch.linalg.lstsq(x, y).solution
        res = torch.sum((y - x @ p) ** 2)
        return p, res if full else p

    def polyval(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == p.ndim == 1
        return torch.stack([x**i for i in range(p.shape[0])], dim=-1) @ p

    def prod(self, array: torch.Tensor) -> float:
        return torch.prod(array).item()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> torch.Tensor:
        # pre-process the size
        if isinstance(size, int):
            new_size = (size,)
        elif size is None:
            new_size = (1,)
        else:
            new_size = size
        # sample
        if self.is_dtype_real:
            samples = torch.normal(
                mean=loc, std=scale, size=new_size, dtype=self.dtype
            )
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            samples = torch.normal(
                mean=loc, std=scale, size=new_size, dtype=real_dtype
            ) + 1j * torch.normal(
                mean=loc, std=scale, size=new_size, dtype=real_dtype
            )
        # post-process
        return samples.item() if size is None else samples

    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> Union[torch.Tensor, Number]:
        # pre-process the size
        if isinstance(size, int):
            new_size = (size,)
        elif size is None:
            new_size = (1,)
        else:
            new_size = size
        # elif not instance
        if self.is_dtype_real:
            samples = torch.rand(new_size, dtype=self.dtype)
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            samples = torch.rand(new_size, dtype=real_dtype) + 1j * torch.rand(
                new_size, dtype=real_dtype
            )
        # post-process
        return samples.item() if size is None else samples

    @elementary_math_function
    def real(self, array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    def reshape(
        self, array: torch.Tensor, newshape: TupleOrList[int]
    ) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @elementary_math_function
    def sin(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @elementary_math_function
    def sinc(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sinc(array)

    def sort(
        self, array: torch.Tensor, descending: bool = False
    ) -> torch.Tensor:
        return torch.sort(array, descending=descending).values

    @elementary_math_function
    def spacing(self, array: torch.Tensor) -> torch.Tensor:
        # spacing is not implemented in PyTorch so we use the NumPy implementation
        return self.array(np.spacing(array.cpu().detach().numpy()))

    @elementary_math_function
    def sqrt(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    def squeeze(self, array: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(array)

    def stack(
        self, arrays: TupleOrList[torch.Tensor], axis: int = 0
    ) -> torch.Tensor:
        return torch.stack(tuple(arrays), dim=axis)

    def sum(
        self,
        array: torch.Tensor,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return torch.sum(array, dim=axis, keepdim=keepdims)

    @elementary_math_function
    def tan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tan(array)

    @elementary_math_function
    def tanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    def tensordot(
        self, a: torch.Tensor, b: torch.Tensor, axes: int = 2
    ) -> torch.Tensor:
        return torch.tensordot(a, b, dims=axes)

    def tile(
        self, array: torch.Tensor, reps: Union[int, TupleOrList[int]]
    ) -> torch.Tensor:
        return torch.tile(array, [reps] if isinstance(reps, int) else reps)

    def trace(self, array: torch.Tensor) -> Union[Number, torch.Tensor]:
        return (
            torch.trace(array)
            if array.ndim == 2
            else torch.einsum("ijj->i", array)
        )

    def transpose(self, array: torch.Tensor) -> torch.Tensor:
        return torch.transpose(array, -2, -1)

    def triu(self, array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.triu(array, k)

    def vstack(
        self, arrays: Union[tuple[torch.Tensor], list[torch.Tensor]]
    ) -> torch.Tensor:
        return torch.vstack(arrays)

    def where(
        self,
        condition: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x is None and y is None:
            return torch.where(condition)
        elif x is not None and y is not None:
            return torch.where(condition, x, y)
        else:
            raise ValueError(
                f"Both x and y have to be specified but are respectively {x} and {y}"
            )

    def zeros(self, shape: list[int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype)

    def zeros_bool(self, shape: TupleOrList[int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bool)

    def zeros_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(array, dtype=self.dtype)
