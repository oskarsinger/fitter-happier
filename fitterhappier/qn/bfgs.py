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
class BFGSServer:

    def __init__(self, 
        get_objective, 
        get_gradient,
        initial_estimate,
        epsilon):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.initial_estimate = initial_estimate
        self.epsilon = epsilon

        self.d = self.initial_estimate.shape[0]
        self.obj_history = []

    def compute_solution(self):

        estimatet = np.copy(self.initial_estimate)
        estimatet1 = None

        self.obj_history.append(
            self.get_objective(estimatet))
        
        gradt = self.get_gradient(estimatet)
        gradt1 = None
        grad_norm = np.linalg.norm(grad)
        H = np.eye(self.d)
        t = 0

        while grad_norm > epsilon:

            # Compute new estimate
            p = np.dot(-H, gradt)
            eta = line_search(
                self.get_objective,
                self.get_gradient,
                estimatet,
                p,
                gfk=gradt,
                old_fval=self.obj_history[-1],
                old_old_fval=None if t == 0 else self.obj_history[-2])
            s = eta * p
            estimatet1 = estimatet + s

            self.obj_history.append(
                self.get_objective(estimatet1))

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

            H = self._get_updated_H(H, s, y)

            t += 1

    def _get_updated_H(self, H, s, y):

        rho = 1 / np.dot(s.T, y)
        quad_terms = rho * np.dot(s, y.T)
        left = np.eye(self.d) - quad_terms
        right = np.eye(self.d) - quad_terms.T
        H_quad = get_multidot([left, H, right])
        s_cov = rho * np.dot(s, s.T)
        H = H_quad + s_cov

        return H
