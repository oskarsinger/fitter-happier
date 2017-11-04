import numpy as np

from models.regression import LinearRegressionModel as LR
from fitterhappier.coordinate import StochasticCoordinateDescentOptimizer as SCDO

class LRSCDTester:

    def __init__(self,
        data_server,
        batch_size=1,
        max_rounds=10,
        epsilon=10**(-5),
        theta_init=None):

        self.ds = data_server
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if theta_init is None:
            theta_init = np.random.randn(self.ds.rows(), 1)
        else:
            theta_init = np.copy(theta_init)

        self.theta_init = theta_init
        self.model = LR(self.ds.cols())
        self.theta = None
        self.objectives = None

    def get_parameters(self):

        return self.theta

    def run(self):

        theta_t = np.copy(self.theta_init)

        self.data = self.ds.get_data()
        get_obj = lambda theta: self.model.get_objective(
            self.data, theta)
        get_grad = lambda theta, batch: self.model.get_gradient(
            self.data, theta, batch=batch)
        get_projected = lambda theta: self.model.get_projected(
            self.data, theta)
        scdo = SCDO(
            self.ds.rows(),
            get_obj,
            get_grad,
            get_projected,
            epsilon=self.epsilon,
            batch_size=self.batch_size,
            max_rounds=self.max_rounds,
            theta_init=self.theta_init)

        scdo.run()

        self.theta = scdo.get_parameters()
        self.objectives = scdo.objectives
