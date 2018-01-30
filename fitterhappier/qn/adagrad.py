import numpy as np

from fitterhappier.utils.proximal import get_mirror_update as get_mu
from fitterhappier.utils import get_shrunk_and_thresholded as get_st
from theline.svd import get_svd_power

class FullAdaGradOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=100,
        delta=10**(-5),
        epsilon=10**(-5),
        eta0=0.1,
        lower=None):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.d = d
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.eta0 = eta0

        if theta_init is None:
            theta_init = self.get_projected(
                np.random.randn(self.d, 1) / self.d)

        self.theta_init = theta_init
        self.adagrad = FullAdaGradServer(
            delta=delta,
            lower=lower)
        self.theta_hat = None
        self.objectives = []

    def get_parameters(self):

        if self.theta_hat is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.theta_hat)

    def run(self):

        estimate = np.copy(self.theta_init)

        self.objectives.append(
            self.get_objective(estimate))
        
        search_dir_size = float('inf')
        t = 0

        while search_dir_size > self.epsilon and t < self.max_rounds:

            # Compute unprojected update
            grad = self.get_gradient(estimate)
            update = self.adagrad.get_update(
                estimate,
                grad,
                self.eta0)

            # Compute convergence criterion
            search_dir = - (update - estimate) / self.eta0
            search_dir_size = np.linalg.norm(search_dir)**2

            # Project onto feasible region for new estimate
            estimate = self.get_projected(update)

            self.objectives.append(
                self.get_objective(estimate))

            if t % 100 == 0:
                print('Round:', t)
                print('Objective:', self.objectives[-1])
                print('Search direction size:', search_dir_size)

            t += 1

        self.theta_hat = estimate

class FullAdaGradServer:

    def __init__(self,
        delta=10**(-5),
        lower=None):

        self.delta = delta
        self.lower= lower

        self.num_rounds = 0
        self.G = None

    def get_update(self, parameters, gradient, eta):
        
        self.num_rounds += 1

        if self.G is None:
            self.G = np.dot(gradient, gradient.T)
            self.d = gradient.shape[0]
        else:
            self.G += np.dot(gradient, gradient.T)
            self.S = get_svd_power(self.G, 0.5)


        mirror_update = get_mu(
            parameters,
            eta,
            gradient, 
            self._get_dual,
            self._get_primal)

        return mirror_update

    def _get_dual(self, parameters):

        H = self.S + np.eye(self.d) * self.delta

        return np.dot(H, parameters)

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

        return np.linalg.solve(
            self.S + self.delta * np.eye(self.d),
            dual_update)

class DiagonalAdaGradServer:

    def __init__(self,
        delta=10**(-6),
        lower=None):

        self.delta = delta
        self.lower= lower

        self.num_rounds = 0
        self.H = None

    def get_update(self, parameters, gradient, eta):
        
        self.num_rounds += 1

        if self.H is None:
            self.H = np.abs(gradient)
        else:
            old = np.power(self.H, 2)
            new = np.power(gradient, 2)

            self.H = np.power(old + new, 0.5)

        mirror_update = get_mu(
            parameters,
            eta,
            gradient, 
            self._get_dual,
            self._get_primal)

        return mirror_update

    def _get_dual(self, parameters):

        return (self.H + self.delta) * parameters

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

        return dual_update / (self.H + self.delta)
