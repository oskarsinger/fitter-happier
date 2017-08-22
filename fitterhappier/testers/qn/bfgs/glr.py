import numpy as np

from fitterhappier.qn import BFGSSolver as BFGSS
from whitehorses.loaders.shortcuts import get_LRGL
from whitehorses.servers.batch import BatchServer as BS
from models import LinearRegression as LR

class GaussianLinearRegressionBFGSTester:

    def __init__(self,
        n,
        p,
        max_rounds=100,
        epsilon=10**(-5),
        init_params=None,
        noisy=False):

        self.n = n
        self.p = p + 1 # Add 1 for bias term
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.noisy = noisy

        if init_params is None:
            init_params = np.random.randn(self.p, 1)

        self.init_params = init_params
        self.w = np.random.randn(self.p, 1)
        
        loader = get_LRGL(
            self.n, 
            [self.p - 1],
            ws=[self.w],
            noisys=[self.noisy],
            bias=True)[0]

        self.server = BS(loader)
        self.model = LR(self.p, 0)
        self.w_hat = None

    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not yet been computed.')

        return self.w_hat

    def run(self):

        bfgs = BFGSS(
            self.model,
            self.server,
            initial_estimate=self.init_params,
            max_rounds=self.max_rounds,
            epsilon=self.epsilon)

        bfgs.compute_parameters()

        self.w_hat = bfgs.get_parameters()
        self.objectives = bfgs.objectives
