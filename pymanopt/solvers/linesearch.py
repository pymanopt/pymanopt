"""
Line-search based on linesearch.m in the manopt MATLAB package.
"""
import numpy as np

class Linesearch(object):
    def __init__(self):
        """Initialise linesearch default parameters"""
        # TODO: allow user to initiate these
        self.contraction_factor = .5
        self.optimism = 1/.5
        self.suff_decr = 1e-4
        self.max_steps = 25
        self.initial_stepsize = 1

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

        alpha = self.initial_stepsize/norm_d

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

        return stepsize, newx
