from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import torch

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


def elementary_math_function(
    f: Callable[[torch.Tensor], Union[torch.Tensor, Number]],
) -> Callable[[Union[torch.Tensor, Number]], Union[torch.Tensor, Number]]:
    def inner(
        self, x: Union[torch.Tensor, Number]
    ) -> Union[torch.Tensor, Number]:
        if isinstance(x, Number):
            return f(self, self.array(x)).item()
        else:
            return f(self, x)

    inner.__doc__ = f.__doc__
    inner.__name__ = f.__name__
    return inner


class PytorchNumericsBackend(NumericsBackend):
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

    @property
    def DEFAULT_REAL_DTYPE(self):
        return torch.tensor([1.0]).dtype

    @property
    def DEFAULT_COMPLEX_DTYPE(self):
        return torch.tensor([1j]).dtype

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
        return f"PytorchNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @elementary_math_function
    def abs(self, array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    def all(self, array: torch.Tensor) -> bool:
        return bool(torch.all(torch.tensor(array, dtype=bool)).item())

    def allclose(
        self,
        array_a: torch.Tensor,
        array_b: torch.Tensor,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> bool:
        return torch.allclose(array_a, array_b, rtol=rtol, atol=atol)

    def any(self, array: torch.Tensor) -> bool:
        return bool(torch.any(torch.tensor(array, dtype=bool)).item())

    def arange(self, *args: int) -> torch.Tensor:
        return torch.arange(*args)

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

    def array(self, array: array_t) -> torch.Tensor:  # type: ignore
        return torch.as_tensor(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: torch.Tensor,
        array_b: torch.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        torch.testing.assert_close(
            self.array(array_a),
            self.array(array_b),
            rtol=rtol,
            atol=atol,
        )

    def block(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        # TODO: implement actual block (where wr could give
        # arbitrarily nested lists of arrays)
        return torch.cat(arrays)

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

    def eps(self) -> torch.Tensor:
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

    def iscomlexobj(self, array: torch.Tensor) -> bool:
        return torch.is_complex(array)

    @elementary_math_function
    def isnan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    def isrealobj(self, array: torch.Tensor) -> bool:
        return bool(torch.all(torch.isreal(array)).item())

    def linalg_cholesky(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(array)

    def linalg_det(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(array)

    def linalg_eigh(
        self, array: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: torch.Tensor, array_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if array_y is None:
            return torch.linalg.eigvalsh(array_x)
        else:
            return torch.from_numpy(
                scipy.linalg.eigvalsh(
                    array_x.cpu().detach().numy(),
                    array_y.cpu().detach().numpy(),
                )
            )

    def linalg_expm(
        self, array: torch.Tensor, symmetric: bool = False
    ) -> torch.Tensor:
        if not symmetric:
            return torch.matrix_exp(array)

        w, v = torch.linalg.eigh(array)
        w = torch.expand_dims(torch.exp(w), dim=-1)
        expmA = v @ (w * torch.conj(v))
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
            return self.array(scipy.linalg.logm(array.cpu().detach().numpy()))

        w, v = torch.linalg.eigh(array)
        w = torch.expand_dims(torch.log(w), dim=-1)
        logmA = v @ (w * torch.conjugate(v))
        if torch.is_complex(array):
            return torch.real(logmA)
        return logmA

    def linalg_norm(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.linalg.norm(array, *args, **kwargs)

    def linalg_qr(
        self, array: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(array, *args, **kwargs)

    @elementary_math_function
    def log(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    def logspace(self, *args: int) -> torch.Tensor:
        return torch.logspace(*args, dtype=self.dtype)

    def ndim(self, array: torch.Tensor) -> int:
        return array.ndim

    def ones(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.ones(shape, dtype=self.dtype)

    def prod(self, array: torch.Tensor) -> float:
        return torch.prod(array).item()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, Sequence[int]] = 1,
    ) -> torch.Tensor:
        if not isinstance(size, Sequence):
            size = (size,)
        else:
            size = tuple(size)
        if self.is_dtype_real:
            return torch.normal(
                mean=loc, std=scale, size=size, dtype=self.dtype
            )
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            return torch.normal(
                mean=loc, std=scale, size=size, dtype=real_dtype
            ) + 1j * torch.normal(
                mean=loc, std=scale, size=size, dtype=real_dtype
            )

    def random_randn(self, *dims: int) -> torch.Tensor:
        if self.is_dtype_real:
            return torch.randn(dims, dtype=self.dtype)
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            return torch.randn(dims, dtype=real_dtype) + 1j * torch.randn(
                dims, dtype=real_dtype
            )

    def random_uniform(self, size: int) -> torch.Tensor:
        if self.is_dtype_real:
            return torch.rand(size, dtype=self.dtype)
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            return torch.rand(size, dtype=real_dtype) + 1j * torch.rand(
                size, dtype=real_dtype
            )

    @elementary_math_function
    def real(self, array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    def reshape(
        self, array: torch.Tensor, newshape: Sequence[int]
    ) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @elementary_math_function
    def sin(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @elementary_math_function
    def sinc(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sinc(array)

    def sort(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sort(array).values

    def spacing(self, array: torch.Tensor) -> torch.Tensor:
        # spacing is not implemented in PyTorch so we use the NumPy implementation
        return self.array(np.spacing(array.cpu().detach().numpy()))

    @elementary_math_function
    def sqrt(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    def squeeze(self, array: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(array)

    def sum(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.sum(array, *args, **kwargs)

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
        self, array: torch.Tensor, reps: int | Sequence[int]
    ) -> torch.Tensor:
        return torch.tile(array, [reps] if isinstance(reps, int) else reps)

    def trace(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.trace(array, *args, **kwargs)

    def transpose(self, array: torch.Tensor) -> torch.Tensor:
        return torch.transpose(array, -2, -1)

    def triu_indices(self, n: int, k: int = 0) -> torch.Tensor:
        return torch.triu_indices(n, k)

    def vstack(
        self, arrays: Tuple[torch.Tensor] | list[torch.Tensor]
    ) -> torch.Tensor:
        return torch.vstack(arrays)

    def where(
        self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(condition, x, y)

    def zeros(self, shape: list[int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype)

    def zeros_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(array, dtype=self.dtype)
