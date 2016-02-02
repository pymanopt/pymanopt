class Solver(object):
    def __init__(self, mingradnorm=1e-6, maxiter=1000, maxtime=1000,
                 minstepsize=1e-10, verbosity=2):
        """
        Generic solver base class.
        Variable attributes (defaults in brackets):
            - maxiter (1000)
                Max number of iterations to run.
            - maxtime (1000)
                Max time (in seconds) to run.
            - mingradnorm (1e-6)
                Terminate if the norm of the gradient is below this.
            - minstepsize (1e-10)
                Terminate if linesearch returns a vector whose norm is below
                this.
            - verbosity (2)
                Level of information printed by the solver while it operates, 0
                is silent, 2 is most information.
        """
        self._mingradnorm = mingradnorm
        self._maxiter = maxiter
        self._maxtime = maxtime
        self._minstepsize = minstepsize
        self._verbosity = verbosity
