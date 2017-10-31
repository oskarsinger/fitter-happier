import numpy as np

# TODO: quasi-Newton stuff gets weird here; figure it out
class StochasticCoordinateDescentOptimizer:

    def __init__(self, 
        p, 
        get_gradient,
        epsilon=10**(-5),
        batch_size=1, 
        max_rounds=10,
        theta_init=None):
        
        self.p = p
        self.get_gradient = get_gradient
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_rounds = max_rounds

        if theta_init is None:
            theta_init = np.zeros((self.p, 1))

        self.theta_init = theta_init
        self.theta = None
        self.cushion = 0 if self.batch_size > 1 \
            else self.p % self.batch_size
        self.num_batches = self.p / self.batch_size

        if self.cushion > 0:
            self.num_batches += 1

    def get_parameters(self):

        return self.theta
        
    def run(self):

        i = 0
        converged = False
        order = np.arange(self.p)
        theta_t = np.copy(self.theta_init)
        theta_t1 = np.zeros_like(theta_t)

        while i < self.max_rounds and not converged:

            np.random.shuffle(order)
            batches = None

            if self.batch_size > 1:
                if self.cushion > 0:
                    batches = np.hstack(
                        order,
                        order[:self.cushion])
                
                batches = np.sort(
                    batches.reshape((
                        self.num_batches,
                        self.batch_size)))
            else:
                batches = order
            
            for batch in batches:
                grad = self.get_gradient(
                    theta_t,
                    batch)

                theta_t1[batch,:] -= grad
            
            diff = theta_t - theta_t1
            converged = np.thelineg.norm(diff) < self.epsilon
            theta_t = np.copy(theta_t1) 

            i += 1

        self.theta = theta_t1
