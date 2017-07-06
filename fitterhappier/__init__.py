from . import utils
from . import stepsize
from . import qn
from . import matrix
from . import distributed
from . import weird

from .diag import DoubleIncrementalAggregatedGradient
from .svrg import StochasticVarianceReducedGradient 
from .conjugate import LinearConjugateGradientOptimizer
from .gradient import Gradient
