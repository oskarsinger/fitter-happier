import numpy as np

from fitterhappier.qn import BFGSSolver as BFGSS
from data.loaders.shortcuts import get_LRGL
from data.servers.batch import BatchServer as BS
from models import LinearRegression as LR

class GaussianLinearRegressionBFGSTester:

    def __init__(self,
        n,
        p,
        max_rounds,
        epsilon=10**(-5),
        init_params=None,
        noisy=False):

        self.n = n
        self.p = p + 1 # Add 1 for bias term
        self.max_rounds = max_rounds
        self.noisy = noisy

        if init_params is None:
            init_params = np.random.randn(self.p, 1)

        self.init_params = init_params
        self.w = np.random.randn(self.p, 1)
        
        loader = get_LGRL(
            self.n, 
            [self.p],
            ws=[self.w],
            noisys=[self.noisy],
            bias=True)

        self.servers = [BS(l) for l in loaders]
        self.model = LR(self.p)
        self.w_hat = None

    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not yet been computed.')

        return self.w_hat

    def run(self):
        pass
