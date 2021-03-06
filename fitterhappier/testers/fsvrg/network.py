import numpy as np

from fitterhappier.distributed import BanditFSVRG
from data.loaders.shortcuts import get_er_ESGWBEL
from data.servers.rl import BanditServer as BS
from models import BanditNetworkRademacherGaussianMixtureModel as BNRGMM

class BNRGMMBanditFSVRGTester:

    def __init__(self,
        num_nodes,
        budget,
        max_rounds=10,
        h=0.01,
        graph_p=0.6):

        self.num_nodes = num_nodes
        self.budget = budget
        self.max_rounds = max_rounds
        self.h = h

        self.loaders = get_er_ESGWBEL(
            num_nodes, graph_p=graph_p)

        self.init_params = np.random.randn(
            6 * self.num_nodes, 1)
        self.init_params[::6] = 0.5
        self.init_params[1::6] = 0.5

        self.servers = [BS(l) for l in self.loaders]
        # TODO: eventually involve unknown baseline
        self.get_model = lambda i: BNRGMM(budget, i)
        self.w_hat = None
        
    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not yet been computed.')

        return self.w_hat

    def run(self):

        self.bfsvrg = BanditFSVRG(
            self.get_model,
            self.servers,
            max_rounds=self.max_rounds,
            h=self.h,
            init_params=self.init_params)

        self.bfsvrg.compute_parameters()

        self.w_hat = self.bfsvrg.get_parameters()
        self.objectives = self.bfsvrg.objectives
