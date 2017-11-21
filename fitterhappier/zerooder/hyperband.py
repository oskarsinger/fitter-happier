import numpy as np

# TODO: cite HyperBand paper
class HyperBandOptimizer:

    def __init__(self,
        get_sample,
        get_validation_loss,
        max_iter=81,
        eta=3):

        self.get_sample = get_sample
        self.get_evaluation = get_evaluation
        self.max_iter = max_iter
        self.eta = eta

        self.s_max = int(
            np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter
        self.theta = None

    def get_parameters(self):

        return self.theta

    def run(self):

        for s in reversed(range(s_max+1)):
            n = int(np.ceil(self.eta**s * self.B / self.max_iter / (s + 1)))
            r = self.max_iter * self.eta**(-s)
            samples = [self.get_sample() for _ in range(n)] 

            for i in range(s + 1):
                n_i = n * self.eta**(-i) 
                r_i = r * self.eta**i
                evals = [self.get_evaluation(num_iters=r_i, sample=sample)
                         for sample in samples]
                to_keep = np.argsort(evals)[:int(n_i / self.eta)]
                samples = [sample[j] for j in to_keep]

        self.theta = samples[0]
