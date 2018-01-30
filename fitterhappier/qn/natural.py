import numpy as np

from theline.utils import get_sherman_morrison as get_sm

class EmpiricalNaturalGradientOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=100,
        epsilon=10**(-5),
        eta0=0.1,
        verbose=False):

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
        self.eng = EmpiricalNaturalGradientServer()
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
            update = self.eng.get_update(
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

class EmpiricalNaturalGradientServer:

    def __init__(self):

        self.G_inv = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        if self.G_inv is None:
            self.G_inv = np.eye(gradient.shape[0])
            
        self.G_inv = get_sm(
            self.G_inv,
            gradient,
            gradient)
        search_direction = np.dot(self.G_inv, gradient)
        scaling = self.num_rounds * eta
            
        return parameters - scaling * search_direction
