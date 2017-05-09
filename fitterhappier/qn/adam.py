import numpy as np

from .. import utils as ou
from ..utils.proximal import get_mirror_update as get_mu
from fitterhappier.utils import get_shrunk_and_thresholded as get_st
from linal.svd_funcs import get_multiplied_svd, get_svd_power
from linal.utils import get_sherman_morrison as get_sm
from drrobert.arithmetic import get_moving_avg as get_ma

class DiagonalAdamServer:

    def __init__(self, 
        delta=10**(-8),
        beta1=0.9,
        beta2=0.999,
        lower=None, 
        verbose=False):

        # TODO: try to enforce correct step-size sequence for RDA
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.lower = lower
        self.verbose = verbose

        self.d = None
        self.first_moment = None
        self.second_moment = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        if self.d is None:
            self.d = gradient.shape[0]
            self.first_moment = np.zeros((self.d, 1))
            self.second_moment = np.zeros((self.d, 1))

        self.second_moment = get_ma(
            self.second_moment,
            np.power(gradient, 2), 
            1 - self.beta2,
            self.beta2)
        self.first_moment = get_ma(
            self.first_moment, 
            gradient, 
            1 - self.beta1,
            self.beta1)

        denom = 1 - self.beta1**(self.num_rounds)
        fm_hat = self.first_moment / denom
        mirror_update = get_mu(
            parameters, 
            eta, 
            fm_hat,
            self._get_dual, 
            self._get_primal)

        return mirror_update

    def _get_dual(self, parameters):

        denom = 1 - self.beta2**(self.num_rounds)
        sm_hat = self.second_moment / denom

        # Get the dual transformation
        H = np.power(sm_hat, 0.5) + self.delta

        return H * parameters

    def _get_primal(self, dual_update):

        if self.lower is not None:
            dus = dual_update.shape

            if len(dus) == 2 and not 1 in set(dus):
                (U, s, V) = np.linalg.svd(dual_update)
                sparse_s = get_st(s, lower=self.lower)
                dual_update = get_multiplied_svd(U, s, V)
            else:
                dual_update = get_st(
                    dual_update, lower=self.lower) 

        denom = 1 - self.beta2**(self.num_rounds)
        sm_hat = self.second_moment / denom
        
        # Get the primal transformation
        H = np.power(sm_hat, 0.5) + self.delta
            
        return dual_update / H

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'second_moment': self.second_moment,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'grad': self.first_moment,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
