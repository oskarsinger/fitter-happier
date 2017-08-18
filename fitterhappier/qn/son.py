import numpy as np

class SketchedOnlineNewtonOptimizer:

    def __init__(self,
        C, alpha, m,
        verbose=False):

        self.C = C
        self.alpha = alpha
        self.m = m
        self.verbose = verbose

        self.u = None
        (self.S, self.H) = [None] * 2


