"""
Line-search based on linesearch.m and linesearch_adaptive.m in the manopt
MATLAB package.
"""
import numpy as np


class LineSearch(object):
    def __init__(self):
        """Initialise line search default parameters"""
        # TODO: allow user to initiate these
        self.contraction_factor = 0.5
        self.optimism = 1 / 0.5
        self.suff_decr = 1e-4
        self.max_steps = 25
        self.initial_stepsize = 1

        self._oldf0 = None

    def search(self, objective, manifold, x, d, f0, df0):
        """
        Function to perform backtracking line-search.
        Arguments:
            - objective
                objective function to optimise
            - manifold
                manifold to optimise over
            - x
                starting point on the manifold
            - d
                tangent vector at x (descent direction)
            - df0
                directional derivative at x along d
        Returns:
            - stepsize
                norm of the vector retracted to reach newx from x
            - newx
                next iterate suggested by the line-search
        """
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)

        if self._oldf0 is not None:
            # Pick initial step size based on where we were last time.
            alpha = 2 * (f0 - self._oldf0) / df0
            # Look a little further
            alpha *= self.optimism
        else:
            alpha = self.initial_stepsize / norm_d

        # Make the chosen step and compute the cost there.
        newx = manifold.retr(x, alpha * d)
        newf = objective(newx)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        while (newf > f0 + self.suff_decr * alpha * df0 and
               step_count <= self.max_steps):

            # Reduce the step size
            alpha = self.contraction_factor * alpha

            # and look closer down the line
            newx = manifold.retr(x, alpha * d)
            newf = objective(newx)

            step_count = step_count + 1

        # If we got here without obtaining a decrease, we reject the step.
        if newf > f0:
            alpha = 0
            newx = x

        stepsize = alpha * norm_d

        self._oldf0 = f0

        return stepsize, newx

class LineSearchAdaptive(object):
    def __init__(self):
        self._contraction_factor = 0.5
        self._suff_decr = 0.5
        self._max_steps = 10
        self._initial_stepsize = 1

        self._oldalpha = None

    def search(self, objective, man, x, d, f0, df0):
        norm_d = man.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_stepsize / norm_d

        newx = man.retr(x, alpha * d)
        newf = objective(newx)
        cost_evaluations = 1

        while (newf > f0 + self._suff_decr * alpha * df0 and
               cost_evaluations <= self._max_steps):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            newx = man.retr(x, alpha * d)
            newf = objective(newx)

            cost_evaluations += 1

        if newf > f0:
            alpha = 0
            newx = x

        stepsize = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about stepsize but about alpha. This is the
        # reason why this line search is not invariant under rescaling of the
        # search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        return stepsize, newx
