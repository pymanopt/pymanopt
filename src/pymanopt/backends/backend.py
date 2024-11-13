from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Literal, Optional, Protocol, TypeVar, Union

import numpy as np
import scipy.special


__all__ = ["TupleOrList", "Backend", "DummyBackendSingleton"]

T = TypeVar("T")
TupleOrList = Union[list[T], tuple[T, ...]]


def not_implemented(function):
    @wraps(function)
    def inner(self, *args, **kwargs):
        raise NotImplementedError(
            f"Function '{function.__name__}' not implemented for backend {self}"
        )

    return inner


class Backend(ABC):
    """Abstract base class defining the interface for autodiff backends."""

    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    _dtype: type

    class array_t(Protocol):
        """Array type used for static type checks.

        It is supposed to represent either a numpy array, or a jax array, or a
        torch tensor, or a tensorflow tensor.
        """

        IndexType = Union[int, slice, None]

        def __getitem__(
            self, idx: Union[IndexType, tuple[IndexType, ...]]
        ) -> "Backend.array_t":
            ...

        def __len__(self) -> int:
            ...

        def __neg__(self) -> "Backend.array_t":
            ...

        def __add__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __mul__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __rmul__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __matmul__(self, other: "Backend.array_t") -> "Backend.array_t":
            ...

        def __div__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __truediv__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __sub__(self, other: "Backend.array_t") -> "Backend.array_t":
            ...

        @property
        def shape(self) -> tuple[int, ...]:
            ...

        def __eq__(self, other: "Union[int, float, Backend.array_t]") -> bool:
            ...

        def __leq__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __geq__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __lt__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

        def __gt__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t":
            ...

    @property
    @abstractmethod
    def dtype(self) -> type:
        ...

    @property
    @abstractmethod
    def is_dtype_real(self) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def DEFAULT_REAL_DTYPE() -> type:
        ...

    @staticmethod
    @abstractmethod
    def DEFAULT_COMPLEX_DTYPE() -> type:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def __eq__(self, other):
        return repr(self) == repr(other)

    @abstractmethod
    def to_complex_backend(self) -> "Backend":
        ...

    @abstractmethod
    def to_real_backend(self) -> "Backend":
        ...

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    @abstractmethod
    def prepare_function(self, function) -> Callable:
        """Prepares a callable to be used with the backend.

        Args:
            function: A callable.

        Returns:
            A Python callable accepting and a ``numpy.ndarray`` and returning a
            scalar.
        """

    @abstractmethod
    def generate_gradient_operator(self, function, num_arguments) -> Callable:
        """Creates a function to compute gradients of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable of the gradient of `function` accepting arguments
            according to the signature defined by `arguments`.
        """

    @abstractmethod
    def generate_hessian_operator(self, function, num_arguments) -> Callable:
        """Creates a function to compute Hessian-vector products of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable evaluating the Hessian-vector product of
            ``function`` accepting arguments according to the signature defined
            by ``arguments``.
            The returned callable accepts a point of evaluation as a sequence
            of length ``num_arguments``, as well as a vector of the same shape
            that is right-multiplied to the Hessian.
        """

    ##############################################################################
    # Numerics methods
    ##############################################################################

    @not_implemented
    def abs(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def all(self, array: array_t) -> bool:
        ...

    @not_implemented
    def allclose(
        self,
        array_a: array_t,
        array_b: array_t,
        rtol: float,
        atol: float,
    ) -> bool:
        ...

    @not_implemented
    def any(self, array: array_t) -> Optional[bool]:
        ...

    @not_implemented
    def arange(
        self,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> array_t:
        ...

    @not_implemented
    def arccos(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def arccosh(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def arctan(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def arctanh(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def argmin(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def argsort(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def array(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def assert_allclose(
        self,
        array_a: array_t,
        array_b: array_t,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        ...

    # TODO: remove this method
    def assert_almost_equal(self, array_a: array_t, array_b: array_t) -> None:
        self.assert_allclose(array_a, array_b)

    # TODO: remove this method
    def assert_array_almost_equal(
        self, array_a: array_t, array_b: array_t
    ) -> None:
        self.assert_allclose(array_a, array_b)

    @not_implemented
    def assert_equal(self, array_a: array_t, array_b: array_t) -> None:
        ...

    @not_implemented
    def concatenate(
        self,
        arrays: TupleOrList[array_t],
        axis: int = 0,
    ) -> array_t:
        ...

    @not_implemented
    def conjugate(self, array: array_t) -> array_t:
        ...

    def conjugate_transpose(self, array: array_t) -> array_t:
        return self.conjugate(self.transpose(array))

    @not_implemented
    def cos(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def diag(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def diagonal(
        self,
        array: array_t,
        axis1: int,
        axis2: int,
    ) -> array_t:
        ...

    @not_implemented
    def eps(self) -> float:
        ...

    @not_implemented
    def exp(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def expand_dims(self, array: array_t, axis: int) -> array_t:
        ...

    @not_implemented
    def eye(self, size: int) -> array_t:
        ...

    @not_implemented
    def hstack(self, arrays: TupleOrList[array_t]) -> array_t:
        ...

    def herm(self, array: array_t) -> array_t:
        return 0.5 * (array + self.conjugate_transpose(array))

    @not_implemented
    def imag(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def iscomplexobj(self, array: array_t) -> bool:
        ...

    @not_implemented
    def isnan(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def isrealobj(self, array: array_t) -> bool:
        ...

    @not_implemented
    def linalg_cholesky(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def linalg_det(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def linalg_eigh(self, array: array_t) -> tuple[array_t, array_t]:
        ...

    @not_implemented
    def linalg_eigvalsh(
        self,
        array_x: array_t,
        array_y: Optional[array_t] = None,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_expm(
        self,
        array: array_t,
        symmetric: bool = False,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_inv(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def linalg_matrix_rank(self, array: array_t) -> int:
        ...

    @not_implemented
    def linalg_logm(
        self,
        array: array_t,
        positive_definite: bool = False,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_norm(
        self,
        array: array_t,
        ord: Union[int, Literal["fro"], None] = None,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_qr(self, array: array_t) -> tuple[array_t, array_t]:
        ...

    @not_implemented
    def linalg_solve(
        self,
        array_a: array_t,
        array_b: array_t,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_solve_continuous_lyapunov(
        self,
        array_a: array_t,
        array_q: array_t,
    ) -> array_t:
        ...

    @not_implemented
    def linalg_svd(
        self,
        array: array_t,
        full_matrices: bool = True,
    ) -> tuple[array_t, array_t, array_t]:
        ...

    @not_implemented
    def linalg_svdvals(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def log(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def log10(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def logical_not(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def logspace(self, start: float, stop: float, num: int) -> array_t:
        ...

    def matvec(self, A: array_t, x: array_t) -> array_t:
        return A @ x

    def matmul(self, A: array_t, B: array_t) -> array_t:
        return A @ B

    def multieye(self, k: int, n: int) -> array_t:
        return self.tile(self.eye(n), (k, 1, 1))

    @not_implemented
    def ndim(self, array: array_t) -> int:
        ...

    newaxis = None

    @not_implemented
    def ones(self, shape: TupleOrList[int]) -> array_t:
        ...

    @not_implemented
    def ones_bool(self, shape: TupleOrList[int]) -> array_t:
        ...

    pi = np.pi

    @not_implemented
    def polyfit(
        self, x: array_t, y: array_t, deg: int = 1, full: bool = False
    ) -> Union[array_t, tuple[array_t, array_t]]:
        ...

    @not_implemented
    def polyval(self, p: array_t, x: array_t) -> array_t:
        ...

    @not_implemented
    def prod(self, array: array_t) -> float:
        ...

    @not_implemented
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> array_t:
        ...

    def random_randn(self, *dims: int) -> array_t:
        return self.random_normal(loc=0.0, scale=1.0, size=dims)

    @not_implemented
    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> array_t:
        ...

    @not_implemented
    def real(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def reshape(
        self,
        array: array_t,
        newshape: TupleOrList[int],
    ) -> array_t:
        ...

    # TODO: seterr
    # def seterr(all=None):
    #     np.seterr(all=all)
    #     if all == 'raise':
    #         try:
    #             import torch
    #             torch.autograd.set_detect_anomaly(True)
    #         except ImportError:
    #             ...
    #         try:
    #             import jax
    #             jax.config.update("jax_debug_nans", True)
    #         except ImportError:
    #             ...
    #         try:
    #             import tensorflow as tf
    #             tf.debugging.enable_check_numerics()
    #         except ImportError:
    #             ...

    @not_implemented
    def sin(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def sinc(self, array: array_t) -> array_t:
        ...

    def skew(self, array: array_t) -> array_t:
        return 0.5 * (array - self.transpose(array))

    def skewh(self, array: array_t) -> array_t:
        return 0.5 * (array - self.conjugate_transpose(array))

    @not_implemented
    def sort(
        self,
        array: array_t,
        descending: bool = False,
    ) -> array_t:
        ...

    def special_comb(self, n: int, k: int):
        return scipy.special.comb(n, k)

    @not_implemented
    def sqrt(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def squeeze(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def stack(
        self,
        arrays: TupleOrList[array_t],
        axis: int = 0,
    ) -> array_t:
        ...

    @not_implemented
    def sum(
        self,
        array: array_t,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> array_t:
        ...

    def sym(self, array: array_t) -> array_t:
        return 0.5 * (array + self.transpose(array))

    @not_implemented
    def tan(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def tanh(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def tensordot(
        self,
        a: array_t,
        b: array_t,
        axes: int = 2,
    ) -> array_t:
        ...

    @not_implemented
    def tile(
        self,
        array: array_t,
        reps: Union[int, TupleOrList[int]],
    ) -> array_t:
        ...

    @not_implemented
    def trace(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def transpose(self, array: array_t) -> array_t:
        ...

    @not_implemented
    def triu(self, array: array_t, k: int = 0) -> array_t:  # type:ignore
        ...

    @not_implemented
    def vstack(self, arrays: TupleOrList[array_t]) -> array_t:
        ...

    @not_implemented
    def where(
        self,
        condition: array_t,
        x: Optional[array_t],
        y: Optional[array_t],
    ) -> Union[array_t, tuple[array_t, ...]]:
        ...

    @not_implemented
    def zeros(self, shape: TupleOrList[int]) -> array_t:
        ...

    @not_implemented
    def zeros_bool(self, shape: TupleOrList[int]) -> array_t:
        ...

    @not_implemented
    def zeros_like(self, array: array_t) -> array_t:
        ...


class DummyBackend(Backend):
    """Dummy implementation of NumericsBackend, only used as a default in manifolds.

    This class should not be used directly.
    """

    @property
    def dtype(self):
        ...

    @property
    def is_dtype_real(self):
        ...

    @staticmethod
    def DEFAULT_REAL_DTYPE():
        ...

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        ...

    def __repr__(self):
        return "DummyBackend"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def to_complex_backend(self) -> "DummyBackend":
        return self

    def to_real_backend(self) -> "DummyBackend":
        return self

    def prepare_function(self, function):
        return super().prepare_function(function)

    def generate_gradient_operator(self, function, num_arguments):
        return super().generate_gradient_operator(function, num_arguments)

    def generate_hessian_operator(self, function, num_arguments):
        return super().generate_hessian_operator(function, num_arguments)


DummyBackendSingleton = DummyBackend()
