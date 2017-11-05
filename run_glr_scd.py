import click

import numpy as np

from fitterhappier.testers.coordinate import LRSCDTester as LRSCDT
from whitehorses.loaders.simple import GaussianLoader as GL
from whitehorses.loaders.supervised import LinearRegressionGaussianLoader as LRGL
from whitehorses.servers.batch import BatchServer as BS

@click.command()
@click.option('--n', default=1000)
@click.option('--p', default=500)
@click.option('--batch-size', default=1)
@click.option('--max-rounds', default=100)
@click.option('--epsilon', default=10**(-5))
@click.option('--bias', default=False)
@click.option('--noisy', default=False)
def run_it_all_day_bb(
    n,
    p,
    batch_size,
    max_rounds,
    epsilon,
    bias,
    noisy):

    base_loader = GL(n, p)
    lrgl = LRGL(base_loader, bias=bias, noisy=noisy)
    data_server = BS(lrgl)
    tester = LRSCDT(
        data_server,
        max_rounds=max_rounds,
        batch_size=batch_size,
        epsilon=epsilon)

    tester.run()

    w_hat = tester.get_parameters()
    print(np.linalg.norm(w_hat - lrgl.w))

    print(tester.objectives)

if __name__=='__main__':
    run_it_all_day_bb()
