import abc
import typing

import attrs


@attrs.define
class LineSearcher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search(self, objective, manifold, x, d, f0, df0):
        """Function to perform line search.

        Args:
            objective: Objective function to optimize.
            manifold: The manifold to optimize over.
            x: Starting point on the manifold.
            d: Tangent vector at ``x``, i.e., a descent direction.
            df0: Directional derivative at ``x`` along ``d``.

        Returns:
            A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
            the vector retracted to reach the suggested iterate ``newx`` from
            ``x``.
        """


@attrs.define
class BackTrackingLineSearcher(LineSearcher):
    """Back-tracking line-search algorithm."""

    contraction_factor: float = 0.5
    optimism: float = 2.0
    sufficient_decrease: float = 1e-4
    max_iterations: int = 25
    initial_step_size: float = 1.0

    _oldf0: typing.Optional[float] = attrs.field(init=False, default=None)

    def search(self, objective, manifold, x, d, f0, df0):
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)

        if self._oldf0 is not None:
            # Pick initial step size based on where we were last time.
            alpha = 2 * (f0 - self._oldf0) / df0
            # Look a little further
            alpha *= self.optimism
        else:
            alpha = self.initial_step_size / norm_d
        alpha = float(alpha)

        # Make the chosen step and compute the cost there.
        newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied.
        while (
            newf > f0 + self.sufficient_decrease * alpha * df0
            and step_count <= self.max_iterations
        ):

            # Reduce the step size
            alpha = self.contraction_factor * alpha

            # and look closer down the line
            newx = manifold.retraction(x, alpha * d)
            newf = objective(newx)

            step_count = step_count + 1

        # If we got here without obtaining a decrease, we reject the step.
        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        self._oldf0 = f0

        return step_size, newx


@attrs.define
class AdaptiveLineSearcher(LineSearcher):
    """Adaptive line-search algorithm."""

    contraction_factor: float = 0.5
    sufficient_decrease: float = 0.5
    max_iterations: int = 10
    initial_step_size: float = 1.0

    _oldalpha: typing.Optional[float] = attrs.field(init=False, default=None)

    def search(self, objective, manifold, x, d, f0, df0):
        norm_d = manifold.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self.initial_step_size / norm_d
        alpha = float(alpha)

        newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)
        cost_evaluations = 1

        while (
            newf > f0 + self.sufficient_decrease * alpha * df0
            and cost_evaluations <= self.max_iterations
        ):
            # Reduce the step size.
            alpha *= self.contraction_factor

            # Look closer down the line.
            newx = manifold.retraction(x, alpha * d)
            newf = objective(newx)

            cost_evaluations += 1

        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about step_size but about alpha. This is
        # the reason why this line search is not invariant under rescaling of
        # the search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        return step_size, newx
