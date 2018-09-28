import numpy as np

class TwoPointBanditOptimizer:

    def __init__(self, 
        get_objective, 
        d,
        eta, 
        delta_scheduler, 
        r_and_drdw_inv=(lambda w: 0.5 * np.linalg.norm(w)**2, lambda theta: theta),
        max_rounds):

        self.get_objective = get_objective
        self.d = d
        self.eta = eta
        self.delta_scheduler = delta_scheduler
        (self.r, self.drdw_inv) = r_and_drdw_inv
        self.max_rounds = max_rounds

        self.w = None

    def get_parameters(self):

        return self.w

    def run(self):

        theta = np.zeros((self.d, 1))
        w_t = None

        for t in range(self.max_rounds):
            
            # Update parameter estimate
            w_t = self.drdw_inv(theta)
            
            # Sample search direction
            normal_sample = np.random.randn(d, 1)
            shere_sample = normal_sample / np.linalg.norm(normal_sample)

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
