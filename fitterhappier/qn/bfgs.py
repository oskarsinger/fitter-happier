import numpy as np

from scipy.optimize import line_search
from linal.utils import get_multidot

# TODO: cite Mokhtari paper
class RegularizedStochasticBFGSServer:

    def __init__(self):
        pass

    def get_update(self):
        pass

# TODO: cite Schraudolph paper
class OnlineLBFGSServer:

    def __init__(self):
        pass

    def get_update(self):
        pass

class OnlineBFGSServer:

    def __init__(self):
        pass

    def get_update(self):
        pass

# TODO: cite Nocedal and Wright
class LBFGSServer:

    def __init__(self):
        pass

    def get_update(self):
        pass

# TODO: cite Nocedal and Wright
class BFGSSolver:

    def __init__(self, 
        model,
        server,
        initial_estimate=None,
        max_rounds=100,
        epsilon=10**(-5)):

        self.model = model
        self.server = server
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        self.d = self.model.get_parameter_shape()[0]

        if initial_estimate is None:
            initial_estimate = np.random.randn(self.d, 1)

        self.initial_estimate = initial_estimate

        self.data = self.server.get_data()
        self.objectives = []

    def _get_objective(self, estimate):

        return self.model.get_objective(
            self.data,
            self.estimate)

    def _get_gradient(self, estimate):

        return self.model.get_gradient(
            self.data,
            self.estimate)

    def compute_solution(self):

        estimatet = np.copy(self.initial_estimate)
        estimatet1 = None

        self.objectives.append(
            self._get_objective(estimatet))
        
        gradt = self._get_gradient(estimatet)
        gradt1 = None
        grad_norm = np.linalg.norm(grad)
        H = np.eye(self.d)
        t = 0

        while grad_norm > epsilon and t < self.max_rounds:

            # Compute new estimate
            s = self._get_s(H, gradt, estimatet)
            estimatet1 = estimatet + s

            self.objectives.append(
                self._get_objective(estimatet1))

            # Compute new gradient and y
            gradt1 = self.get_gradient(estimatet1)
            grad_norm = np.linalg.norm(gradt1)
            y = gradt1 - gradt

            # Update gradient and esimtate state
            estimatet = np.copy(estimatet1)
            gradt = np.copy(gradt1)

            # Update H (B's inverse)
            if t == 0:
                H *= np.dot(s.T, y) / np.dot(y.T, y)

            H = self._get_H(H, s, y)

            t += 1

    def _get_s(self, H, grad, estimate):

        p = np.dot(-H, grad)
        oofv = None if t == 0 else self.objectives[-2]
        eta = line_search(
            self._get_objective,
            self._get_gradient,
            estimate,
            p,
            gfk=grad,
            old_fval=self.objectives[-1],
            old_old_fval=oofv)

        return eta * p

    def _get_H(self, H, s, y):

        rho = 1 / np.dot(s.T, y)
        quad_terms = rho * np.dot(s, y.T)
        left = np.eye(self.d) - quad_terms
        right = np.eye(self.d) - quad_terms.T
        H_quad = get_multidot([left, H, right])
        s_cov = rho * np.dot(s, s.T)
        H = H_quad + s_cov

        return H
