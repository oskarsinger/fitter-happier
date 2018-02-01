from drrobert.arithmetic import get_moving_avg as get_ma

class GradientOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=100,
        epsilon=10**(-5),
        eta0=0.1):
        pass

class GradientServer:

    def __init__(self, beta=None, dual_avg=False):

        if beta is None:
            beta = 0

        self.alpha = 1 - beta
        self.beta = beta
        self.search_direction = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.search_direction = get_ma(
            self.search_direction, 
            gradient, 
            1 - self.beta,
            self.beta)

        return parameters - eta * self.search_direction

    def get_status(self):

        return {
            'beta': self.beta,
            'dual_avg': self.dual_avg,
            'search_direction': self.search_direction,
            'num_rounds': self.num_rounds}
