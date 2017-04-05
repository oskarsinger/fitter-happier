import click

import numpy as np

from testers.fsvrg import GaussianLinearRegressionFSVRGTester as GLRFSVRGT
from lazyprojector import plot_lines

@click.command()
@click.option('--num-nodes', default=4)
@click.option('--n', default=3000)
@click.option('--p', default=500)
@click.option('--max-rounds', default=5)
@click.option('--h', default=0.01)
@click.option('--noisy', default=False)
def run_it_all_day_bb(
    num_nodes,
    n,
    p,
    max_rounds,
    h,
    noisy):

    tester = GLRFSVRGT(
        num_nodes,
        n,
        p,
        max_rounds=max_rounds,
        h=h,
        noisy=noisy)

    tester.run()

    objs = np.array(
        [sum(os) for os in tester.objectives])
    objs = objs[:,np.newaxis]
    y = np.arange(max_rounds)[:,np.newaxis]
    data_map = {
        'FSVRG': (y,objs,None)}
    title = 'objective value vs communication round'
    path = '_'.join(title.split()) + '.pdf'
    ax = plot_lines(
        data_map,
        'communication round',
        'objective value',
        title).get_figure().savefig(
        path, format='pdf')


if __name__=='__main__':
    run_it_all_day_bb()
