import time
from copy import deepcopy

import numpy as np
import scipy

from pymanopt.manifolds import HermitianPositiveDefinite, SpecialHermitianPositiveDefinite, SymmetricPositiveDefinite
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer

class FrankWolfe(Optimizer):
    """Riemannian Frank Wolfe method.

    Generalization of the Reimmanian Frank-Wolfe method 
    described in [Weber, Sra, Math. Prog., 2022](@cite WeberSra:2022).

    We aim to solve the following constrained optimization problem:
    ..math::    
        \begin{equation}\label{eq:problem_statement}
            \min _{x \in \mathcal{X} \subseteq \mathcal{M}} \phi(x)
        \end{equation

    where :math:`\mathcal{M}` is a Riemannian manifold and 
    :math:`\phi` is a smooth function on :math:`\mathcal{M}`.

    The Frank-Wolfe method is an iterative algorithm that performs
    a line search in the direction of the negative gradient of :math:`\phi`
    to find a new iterate :math:`x_{k+1}`. The algorithm is initialized
    with an initial point :math:`x_0 \in \mathcal{M}` and iterates until
    a stopping criterion is satisfied.
    
    Args:
        sub_problem: A function that solves the subproblem of the Frank-Wolfe
            method. If no value is provided and the manifold is Positive Definite
            matrices then the step direction is computed using the closed-form
            solution from Theorem 4.1 in [Weber, Sra, Math. Prog., 2022](@cite WeberSra:2022).

    """

    def __init__(
        self,
        sub_problem=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sub_problem = sub_problem

    def sgnplus(self, D):
        Dsigns =  -1 * np.sign(D)
        Dsigns[Dsigns == -1] = 0
        return Dsigns


    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def step_direction(self, objective, manifold, gradient, grad, cost, X, L, U):
        sqrtX = scipy.linalg.sqrtm(X)
        D, Q = np.linalg.eigh(grad)

        Qstar = np.matrix(Q).T.conj()
        Xinv = np.linalg.inv(sqrtX)

        Lcap = Qstar @ L @ Q
        Ucap = Qstar @ U @ Q
        A = Ucap - Lcap
        P = np.linalg.cholesky(A)
        Pstar = P.T.conj()

        return Q @ (Pstar @ self.sgnplus(np.diag(D)) @ P + Lcap) @ Qstar

    def run(
        self, problem, L, U, *args, initial_point=None,
    ) -> OptimizerResult:
        """Run FW method.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient
        sub_problem = self.sub_problem
        if sub_problem is None and isinstance(problem.manifold, (HermitianPositiveDefinite, SymmetricPositiveDefinite, SpecialHermitianPositiveDefinite)):
            sub_problem = self.step_direction
        elif sub_problem is None:
            RuntimeError("No subproblem provided and manifold is not positive definite.")

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()


        # Initialize iteration counter and timer.
        iteration = 0
        step_size = 1.0
        line_searcher = AdaptiveLineSearcher()
        start_time = time.time()
        self._initialize_log()

        while True:

            cost = objective(x)
            # print(type(x))
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)

            Z = sub_problem(objective, manifold, gradient, grad, cost, x, L, U)

            column_printer.print_row([iteration, cost, gradient_norm])

            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
            )

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                gradient_norm=gradient_norm,
                iteration=iteration,
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break
            
            Xinv = np.linalg.inv(x)
            Xinvsqrt = scipy.linalg.sqrtm(Xinv)
            
            # step_size, newx = line_searcher.search(objective, manifold, x, Z, cost, manifold.norm(x, Z))
            newx = scipy.linalg.sqrtm(x) @ scipy.linalg.fractional_matrix_power(Xinvsqrt @ Z @ Xinvsqrt, step_size) @ scipy.linalg.sqrtm(x)
            # newx = x + step_size * (Z - x)
            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx, newgrad)

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            gradient_norm = newgradient_norm
            iteration += 1
            step_size = 2 / (iteration + 2)

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )
