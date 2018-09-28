import numpy as np

from fitterhappier.stepsize import InversePowerScheduler as IPS
from theline.svd import get_svd_power

class TwoPointBanditOptimizer:

    def __init__(self, 
        get_objective, 
        d,
        eta=10**(-1), 
        delta_scheduler=None,
        r_and_drdw_inv=(lambda w: 0.5 * np.linalg.norm(w)**2, lambda theta: theta),
        max_rounds=500):

        self.get_objective = get_objective
        self.d = d
        self.eta = eta
        (self.r, self.drdw_inv) = r_and_drdw_inv
        self.max_rounds = max_rounds

        if delta_scheduler is None:
            delta_scheduler = IPS()

        self.delta_scheduler = delta_scheduler

        self.w = None

    def get_parameters(self):

        return self.w

    def run(self):

        theta = None

        if len(self.d) == 1 or self.d[1] == 1:
            theta = np.zeros((self.d, 1))
        else:
            theta = np.zeros(self.d)

        w_t = None

        for t in range(self.max_rounds):
            
            # Update parameter estimate
            w_t = self.drdw_inv(theta)
            
            # Sample search direction
            sphere_sample = None
            
            if len(self.d) == 1 or self.d[1] == 1:
                normal_sample = np.random.randn(d, 1)
                sphere_sample = normal_sample / np.linalg.norm(normal_sample)
            else:
                normal_sample = np.random.randn(*d)
                quad = np.dot(normal_sample.T, normal_sample)
                normalizer = get_svd_power(quad, -0.5)
                shere_sample = np.dot(normal_sample, normalizer)

            # Compute one-dimensional finite difference approximation
            delta_t = self.delta_scheduler.get_stepsize()
            delta_plus = self.get_objective(w_t + delta_t * sphere_sample)
            delta_minus = self.get_objective(w_t - delta_t * sphere_sample)

            # Compute gradient scale and gradient
            grad_scale = 0.5 * delta_t**(-1) * self.d * (delta_plus - delta_minus)
            grad_approx = grad_scale * sphere_sample

            # Update theta
            theta -= self.eta * grad_approx

        self.w = w_t
