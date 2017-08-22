import click

from fitterhappier.testers.qn.bfgs import GaussianLinearRegressionBFGSTester as GLRBFGST

@click.command()
@click.option('--n', default=1000)
@click.option('--p', default=500)
@click.option('--max-rounds', default=100)
@click.option('--epsilon', default=10**(-5))
@click.option('--noisy', default=False)
def run_it_all_day_bb(
    n,
    p,
    max_rounds,
    epsilon,
    noisy):

    tester = GLRBFGST(
        n,
        p,
        max_rounds=max_rounds,
        epsilon=epsilon,
        noisy=noisy)

    tester.run()

    print(tester.objectives)

if __name__=='__main__':
    run_it_all_day_bb()
