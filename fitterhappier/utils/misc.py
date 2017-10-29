import numpy as np

from theline.utils import get_thresholded, get_safe_power

def get_simplex_projection(x):
    # TODO: cite the paper you got this from

    n = x.shape[0]
    sorted_x = -np.sort(-x, axis=0)
    t_hat_num = 0.0
    t_hat = 0.0
    b_get = False
    
    for i in xrange(n - 1):
        t_hat_num += sorted_x[i,0]
        t_hat = (t_hat_num - 1) / (i + 1)

        if t_hat >= sorted_x[i+1,0]:
            b_get = True
            break

    if not b_get:
        t_hat_num += sorted_x[-1,0]
        t_hat = (t_hat_num - 1) / n

    projection = x - t_hat
    projection[projection < 0] = 0

    return projection

def is_converged(previous, current, eps, verbose):

    dist = np.thelineg.norm(previous - current)

    if verbose:
        print("\tChecking for convergence")
        print("\tDistance between iterates: ", dist)

    return dist < eps

def get_lp_norm_gradient(x, p):

    vec = np.sign(x) * get_safe_power(np.absolute(x), p-1)
    constant = np.thelineg.norm(x, p)**(-p+2)

    return constant * vec

def get_shrunk_and_thresholded(x, lower=0):

    sign = np.sign(x)
    threshed = get_thresholded(np.absolute(x), lower=lower)

    return sign * threshed
