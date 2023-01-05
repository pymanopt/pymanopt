from .conjugate_gradient import ConjugateGradient
from .nelder_mead import NelderMead
from .particle_swarm import ParticleSwarm
from .steepest_descent import SteepestDescent
from .trust_regions import TrustRegions


__all__ = [
    "ConjugateGradient",
    "NelderMead",
    "ParticleSwarm",
    "SteepestDescent",
    "TrustRegions",
]


OPTIMIZERS = __all__
