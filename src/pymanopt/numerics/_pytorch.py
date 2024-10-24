from numbers import Number
from typing import Callable, Optional, Union, override

import numpy as np
import scipy.linalg
import torch

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend, TupleOrList


def elementary_math_function(
    f: Callable[["PytorchNumericsBackend", torch.Tensor], torch.Tensor],
) -> Callable[
    ["PytorchNumericsBackend", Union[torch.Tensor, Number]],
    Union[torch.Tensor, Number],
]:
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
    @override
    def dtype(self):
        return self._dtype

    @property
    @override
    def is_dtype_real(self):
        return self.dtype in {torch.float32, torch.float64}

    @override
    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return torch.tensor([1.0]).dtype

    @override
    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return torch.tensor([1j]).dtype

    @override
    def to_real_backend(self) -> "PytorchNumericsBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == torch.complex64:
            return PytorchNumericsBackend(dtype=torch.float32)
        elif self.dtype == torch.complex128:
            return PytorchNumericsBackend(dtype=torch.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    @override
    def to_complex_backend(self) -> "PytorchNumericsBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == torch.float32:
            return PytorchNumericsBackend(dtype=torch.complex64)
        elif self.dtype == torch.float64:
            return PytorchNumericsBackend(dtype=torch.complex128)
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

    @override
    def __repr__(self):
        return f"PytorchNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @elementary_math_function
    @override
    def abs(self, array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    @override
    def all(self, array: torch.Tensor) -> bool:
        return bool(torch.all(torch.tensor(array, dtype=bool)).item())

    @override
    def allclose(
        self,
        array_a: torch.Tensor,
        array_b: torch.Tensor,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> bool:
        return torch.allclose(array_a, array_b, rtol=rtol, atol=atol)

    @override
    def any(self, array: torch.Tensor) -> bool:
        return bool(torch.any(torch.tensor(array, dtype=bool)).item())

    @override
    def arange(self, *args: int) -> torch.Tensor:
        return torch.arange(*args)

    @elementary_math_function
    @override
    def arccos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccos(array)

    @elementary_math_function
    @override
    def arccosh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccosh(array)

    @elementary_math_function
    @override
    def arctan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctan(array)

    @elementary_math_function
    @override
    def arctanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctanh(array)

    @override
    def argmin(self, array: torch.Tensor):
        return torch.argmin(array)

    @override
    def argsort(self, array: torch.Tensor):
        return torch.argsort(array)

    @override
    def array(self, array: array_t) -> torch.Tensor:  # type: ignore
        return torch.as_tensor(array, dtype=self.dtype)

    @override
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

    @override
    def block(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        # TODO: implement actual block (where wr could give
        # arbitrarily nested lists of arrays)
        return torch.cat(arrays)

    def concatenate(
        self, arrays: TupleOrList[torch.Tensor], axis: int = 0
    ) -> torch.Tensor:
        return torch.cat(arrays, axis)

    @elementary_math_function
    @override
    def conjugate(self, array: torch.Tensor) -> torch.Tensor:
        return torch.conj(array)

    @elementary_math_function
    @override
    def cos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    @override
    def diag(self, array: torch.Tensor) -> torch.Tensor:
        return torch.diag(array)

    @override
    def diagonal(
        self, array: torch.Tensor, axis1: int, axis2: int
    ) -> torch.Tensor:
        return torch.diagonal(array, dim1=axis1, dim2=axis2)

    @override
    def eps(self) -> torch.Tensor:
        return torch.finfo(self.dtype).eps

    @elementary_math_function
    @override
    def exp(self, array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @override
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

    @override
    def eye(self, size: int) -> torch.Tensor:
        return torch.eye(size, dtype=self.dtype)

    @override
    def hstack(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        return torch.hstack(arrays)

    @override
    def iscomplexobj(self, array: torch.Tensor) -> bool:
        return array.dtype in {torch.complex64, torch.complex128}

    @elementary_math_function
    @override
    def isnan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    @override
    def isrealobj(self, array: torch.Tensor) -> bool:
        return array.dtype in {torch.float32, torch.float64}

    @override
    def linalg_cholesky(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(array)

    @override
    def linalg_det(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(array)

    @override
    def linalg_eigh(
        self, array: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(array)

    @override
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

    @override
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

    @override
    def linalg_inv(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(array)

    @override
    def linalg_matrix_rank(self, array: torch.Tensor) -> int:
        return int(torch.linalg.matrix_rank(array))

    @override
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

    @override
    def linalg_norm(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.linalg.norm(array, *args, **kwargs)

    @override
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

    @override
    def linalg_solve(
        self, array_a: torch.Tensor, array_b: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.solve(array_a, array_b)

    @override
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

    @override
    def linalg_svd(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(array, *args, **kwargs)

    @elementary_math_function
    @override
    def log(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @override
    def logspace(self, *args: int) -> torch.Tensor:
        return torch.logspace(*args, dtype=self.dtype)

    @override
    def ndim(self, array: torch.Tensor) -> int:
        return array.ndim

    @override
    def ones(self, shape: TupleOrList[int]) -> torch.Tensor:
        return torch.ones(shape, dtype=self.dtype)

    @override
    def prod(self, array: torch.Tensor) -> float:
        return torch.prod(array).item()

    @override
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> torch.Tensor:
        # pre-process the size
        if size is None:
            new_size = (1,)
        elif not isinstance(size, TupleOrList):
            new_size = (size,)
        else:
            new_size = tuple(size)
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

    @override
    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> Union[torch.Tensor, Number]:
        # pre-process the size
        if size is None:
            new_size = (1,)
        elif not isinstance(size, TupleOrList):
            new_size = (size,)
        else:
            new_size = tuple(size)
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
    @override
    def real(self, array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    @override
    def reshape(
        self, array: torch.Tensor, newshape: TupleOrList[int]
    ) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @elementary_math_function
    @override
    def sin(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @elementary_math_function
    @override
    def sinc(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sinc(array)

    @override
    def sort(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sort(array).values

    @override
    def spacing(self, array: torch.Tensor) -> torch.Tensor:
        # spacing is not implemented in PyTorch so we use the NumPy implementation
        return self.array(np.spacing(array.cpu().detach().numpy()))

    @elementary_math_function
    @override
    def sqrt(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    @override
    def squeeze(self, array: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(array)

    @override
    def stack(
        self, arrays: TupleOrList[torch.Tensor], axis: int = 0
    ) -> torch.Tensor:
        return torch.stack(tuple(arrays), dim=axis)

    @override
    def sum(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.sum(array, *args, **kwargs)

    @elementary_math_function
    @override
    def tan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tan(array)

    @elementary_math_function
    @override
    def tanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    @override
    def tensordot(
        self, a: torch.Tensor, b: torch.Tensor, axes: int = 2
    ) -> torch.Tensor:
        return torch.tensordot(a, b, dims=axes)

    @override
    def tile(
        self, array: torch.Tensor, reps: int | TupleOrList[int]
    ) -> torch.Tensor:
        return torch.tile(array, [reps] if isinstance(reps, int) else reps)

    @override
    def trace(self, array: torch.Tensor) -> Union[Number, torch.Tensor]:
        return (
            torch.trace(array)
            if array.ndim == 2
            else torch.einsum("ijj->i", array)
        )

    @override
    def transpose(self, array: torch.Tensor) -> torch.Tensor:
        return torch.transpose(array, -2, -1)

    @override
    def triu(self, array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.triu(array, k)

    @override
    def vstack(
        self, arrays: tuple[torch.Tensor] | list[torch.Tensor]
    ) -> torch.Tensor:
        return torch.vstack(arrays)

    @override
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

    @override
    def zeros(self, shape: list[int]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype)

    @override
    def zeros_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(array, dtype=self.dtype)
