import numpy as np

class LinearConjugateGradientOptimizer:

    def __init__(self, A, b, max_iters=None, epsilon=0.01):

        self.A = A
        self.b = b
        self.epsilon = epsilon

        self.cycle = np.sqrt(self.A.shape[0])

        if max_iters is None:
            max_iters = 5 * self.cycle

        self.max_iters = max_iters
        self.x = None

    def get_parameters(self):

        return self.x

    def run(self):

        x = np.random.randn(
            self.A.shape[1],
            self.b.shape[1])
        r = self.b - np.dot(self.A, x)
        d = np.copy(r)
        delta = np.linalg.norm(r)**2
        threshold = self.epsilon**2 * delta
        i = 0

        while i < self.max_iters and not converged:
            
            q = np.dot(A, d)
            rq_ip = np.trace(np.dot(d.T, q))
            alpha = delta / rq_ip
            x = x + alpha * d

            if i % self.cycle == 0:
                r = b - np.dot(A, x)
            else:
                r = r - alpha * q

            new_delta = np.linalg.norm(r)**2
            beta = new_delta / delta
            delta = new_delta
            d = r + beta * d
            converged = delta < threshold
            i += 1

        self.x = x
