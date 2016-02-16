import time


class Solver(object):
    def __init__(self, maxtime=1000, maxiter=1000, mingradnorm=1e-6,
                 minstepsize=1e-10, maxcostevals=5000):
        """
        Generic solver base class.
        Variable attributes (defaults in brackets):
            - maxtime (1000)
                Max time (in seconds) to run.
            - maxiter (1000)
                Max number of iterations to run.
            - mingradnorm (1e-6)
                Terminate if the norm of the gradient is below this.
            - minstepsize (1e-10)
                Terminate if linesearch returns a vector whose norm is below
                this.
            - maxcostevals (5000)
                Maximum number of allowed cost evaluations
        """
        self._maxtime = maxtime
        self._maxiter = maxiter
        self._mingradnorm = mingradnorm
        self._minstepsize = minstepsize
        self._maxcostevals = maxcostevals

    def _check_stopping_criterion(self, time0, iter=-1, gradnorm=float('inf'),
                                  stepsize=float('inf'), costevals=-1):
        reason = None
        if time.time() >= time0 + self._maxtime:
            reason = ("Terminated - max time reached after %d iterations."
                      % iter)
        elif iter >= self._maxiter:
            reason = ("Terminated - max iterations reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif gradnorm < self._mingradnorm:
            reason = ("Terminated - min grad norm reached after %d "
                      "iterations, %.2f seconds." % (
                          iter, (time.time() - time0)))
        elif stepsize < self._minstepsize:
            reason = ("Terminated - min stepsize reached after %d iterations, "
                      "%.2f seconds." % (iter, (time.time() - time0)))
        elif costevals >= self._maxcostevals:
            reason = ("Terminated - max cost evals reached after "
                      "%.2f seconds." % (time.time() - time0))
        return reason
