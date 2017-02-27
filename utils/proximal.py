from linal.svd_funcs import get_schatten_p_norm as get_sp, get_svd_power
from optimization.utils import get_lp_norm_gradient

import numpy as np
import drrobert.arithmetic as da
import drrobert.debug as drdb

def get_lp_bregman_div_and_grad(p, ip=np.dot):

    breg_func = lambda x: np.linalg.norm(x, ord=p)
    breg_grad = lambda x: get_lp_norm_gradient(x, p) 

    return get_bregman_div_and_grad(breg_func, breg_grad, ip=ip)

def get_bregman_div_and_grad(
    get_bregman_func,
    get_bregman_func_grad, 
    get_ip=np.dot):

    def get_bregman_div(x, x_t):

        grad = get_bregman_func_grad(x_t)
        diff = x - x_t
        ip = get_ip(grad, diff)
        x_breg = get_bregman_func(x)
        x_t_breg = get_bregman_func(x_t)

        return x_breg - x_t_breg - ip

    def get_bregman_grad(x, x_t):

        x_t_grad = get_bregman_func_grad(x_t)
        x_t_ip = get_ip(x_t_grad, x_t)
        x_grad = get_bregman_func_grad(x)
        x_t_breg = get_bregman_func(x_t)

        return x_grad - x_t_grad - x_t_breg + x_t_ip

    return (get_bregman_div, get_bregman_grad)

def get_avg_search_direction(
    old, 
    new, 
    dual_avg, 
    num_rounds, 
    alpha=1,
    beta=0):

    search_direction = None

    if old is not None:
        if dual_avg:
            search_direction = da.get_running_avg(
                old, new, num_rounds)
        else:
            search_direction = da.get_moving_avg(
                old, new, alpha, beta)
    else:
        search_direction = alpha * new

    return search_direction

def get_mirror_update(
    parameters, 
    eta, 
    search_direction, 
    get_dual, 
    get_primal):

    #print 'Computing dual parameters'

    dual_parameters = get_dual(parameters)

    drdb.check_for_nan_or_inf(
        dual_parameters, 
        'optimizers.utils get_mirror_update', 
        'dual_parameters')

    #print 'Computing dual descent update'

    dual_update = dual_parameters - eta * search_direction

    drdb.check_for_nan_or_inf(
        dual_update, 
        'optimizers.utils get_mirror_update', 
        'dual_update')

    #print 'Computing primal parameters'

    primal_parameters = get_primal(dual_update)

    drdb.check_for_nan_or_inf(
        primal_parameters, 
        'optimizers.utils get_mirror_update', 
        'primal_parameters')

    #print 'Returning primal parameters'

    return primal_parameters
