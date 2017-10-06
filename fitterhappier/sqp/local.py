import numpy as np

from .. import LinearConjugateGradientOptimizer

class LocalSQPOptimizer:

    def __init__(self, 
        model,
        data_server,
        max_iters=100,
        epsilon=0.001,
        theta_init=None, 
        lam_init=None):

        self.model = model
        self.ds = data_server
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.d = self.model.get_parameter_shape()
        self.c = self.model.get_constraint_shape()

        if theta_init is None:
            theta_init = np.random.randn(self.d, 1)

        self.theta_init = theta_init

        if lam_init is None:
            lam_init = np.random.randn(self.c, 1)

        self.lam_init = lam_init

        self.theta = None
        self.data = self.ds.get_data()
        self.objectives = []

        self._get_obj = self._get_no_data_func(
            self.model.get_objective)
        self._get_grad = self._get_no_data_func(
            self.model.get_gradient)
        self._get_cons = self._get_no_data_func(
            self.model.get_constraints)
        self._get_cons_grad = self._get_no_data_func(
            self.model.get_constraints_gradient)
        self._get_Lang_H = self._get_no_data_func(
            self.model.get_Lagrangian_Hessian)

    def get_parameters(self):

        return self.theta

    def run(self):

        t = 0
        converged = False
        theta = np.copy(self.theta_init)
        lam = np.copy(self.lam_init)
        params = (theta, lam)

        while t < self.max_iters and not converged:

            obj = self._get_obj(params)

            self.objectives.append(obj)

            grad = self._get_grad(params)
            cons = self._get_cons(params) 
            cons_grad = self._get_cons_grad(params)
            Lang_H = self._get_Lang_H(params)
            b = np.vstack([grad, cons])
            A = np.zeros((self.d+self.c, self.d+self.c))

            A[:self.d,:self.d] += Lang_H
            A[self.d:,:self.d] += cons_grad
            A[:self.d,self.d:] += -cons_grad.T

            lcgo = LinearConjugateGradientOptimizer(A, b)

            lcgo.run()

            result = lcgo.get_parameters()
            p = np.copy(result[:self.d])
            lam = np.copy(result[self.d:])

            theta += p

            params = (theta, lam)

            if t > 0:
                obj_prev = self.objectives[-1]
                converged = np.abs(obj_prev - obj) < self.epsilon

            t += 1

    def _get_no_data_func(self, func):

        return lambda params: func(self.data, params)
