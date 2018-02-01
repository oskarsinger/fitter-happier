import numpy as np

from fitterhappier.utils.proximal import get_mirror_update as get_mu
from fitterhappier.utils import get_shrunk_and_thresholded as get_st
from theline.svd import get_svd_power

class FullAdaGradBlockCoordinateOptimizer:

    def __init__(self,
        ds,
        get_objective,
        get_gradients,
        get_projected,
        theta_inits=None,
        max_rounds=10000,
        deltas=None,
        eta0s=None,
        verbose=False):

        self.ds = ds
        self.get_objective = get_objective
        self.get_gradients = get_gradients
        self.get_projected = get_projected
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.verbose = verbose

        if eta0s is None:
            eta0s = [0.1] * len(self.ds)

        self.eta0s = eta0s

        if deltas is None:
            deltas = [10**(-5)] * len(self.ds)

        self.deltas = deltas

        if theta_inits is None:
            theta_inits = [np.random.randn(*d) / d
                           for d in self.ds]

        self.theta_inits = self.get_projected(theta_inits)
        FAS = FullAdaGradServer
        self.adagrads = [FAS(delta=delta)
                         for delta in deltas]
        self.theta_hats = None
        self.objectives = []

    def get_parameters(self):

        if self.theta_hats is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return [np.copy(th) for th in self.theta_hats]

    def run(self):

        estimates = [np.copy(ti) for ti in self.theta_inits]
        updatest1 = estimate
        updatest = None

        self.objectives.append(
            self.get_objective(estimates))
        
        search_dir_norm = float('inf')
        t = 0

        while mean_search_dir_norm > self.epsilon and t < self.max_rounds:

            # Compute unprojected update
            grads = self.get_gradient(estimates)
            updatest = [np.copy(ut1) for ut1 in updatest1]
            updates_info = zip(estimates, grads, self.eta0s)
            updatest1 = [ag.get_update(e, g, eta0)
                         for (e, g, eta0) in updates_info]

            # Compute convergence criterion
            search_dirs_info = zip(updatest1, updatest, self.eta0s)
            search_dirs = [- (ut1 - ut) / eta0
                           for (ut1, ut, eta0) in search_dirs_info]
            mean_search_dir_norm = np.mean(
                [np.linalg.norm(sd) for sd in search_dirs])


            # Project onto feasible region for new estimate
            estimates = self.get_projected(updatest1)

            if t % 1000 == 0:

                self.objectives.append(
                    self.get_objective(estimates))

                if self.verbose:
                    print('Round:', t)
                    print('Objective:', self.objectives[-1])
                    grads_norms = [np.linalg.norm(grad) 
                                   for grad in grads]
                    print('Gradient norms:', grad_norms)
                    print('Search direction norm:', mean_search_dir_norm)

            t += 1

        self.theta_hats = estimates

class FullAdaGradOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=10000,
        delta=10**(-5),
        epsilon=10**(-5),
        eta0=0.1,
        lower=None,
        verbose=False):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.d = d
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.eta0 = eta0
        self.verbose = verbose

        if theta_init is None:
            theta_init = np.random.randn(self.d, 1) / self.d

        self.theta_init = self.get_projected(theta_init)
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
        updatet1 = estimate
        updatet = None

        self.objectives.append(
            self.get_objective(estimate))
        
        search_dir_norm = float('inf')
        t = 0

        while search_dir_norm > self.epsilon and t < self.max_rounds:

            # Compute unprojected update
            grad = self.get_gradient(estimate)
            updatet = np.copy(updatet1)
            updatet1 = self.adagrad.get_update(
                estimate,
                grad,
                self.eta0)

            # Compute convergence criterion
            search_dir = - (updatet1 - updatet) / self.eta0
            search_dir_norm = np.linalg.norm(search_dir)

            # Project onto feasible region for new estimate
            estimate = self.get_projected(updatet1)

            if t % 1000 == 0:

                self.objectives.append(
                    self.get_objective(estimate))

                if self.verbose:
                    print('Round:', t)
                    print('Objective:', self.objectives[-1])
                    print('Gradient norm:', np.linalg.norm(grad))
                    print('Search direction norm:', search_dir_norm)

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
        self.H = self.S + np.eye(self.d) * self.delta

        return get_mu(
            parameters,
            eta,
            gradient, 
            self._get_dual,
            self._get_primal)

    def _get_dual(self, parameters):

        return np.dot(self.H, parameters)

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

        H_inv = get_svd_power(self.H, -1)

        return np.dot(H_inv, dual_update)

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
