from theline.svd import get_schatten_p_norm as get_sp, get_svd_power
from fitterhappier.utils import get_lp_norm_gradient

import numpy as np
import drrobert.arithmetic as da
import drrobert.debug as drdb

def get_lp_bregman_div_and_grad(p, ip=np.dot):

    breg_func = lambda x: np.thelineg.norm(x, ord=p)
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

def get_mirror_update(
    parameters, 
    eta, 
    search_direction, 
    get_dual, 
    get_primal):

    dual_parameters = get_dual(parameters)
    dual_update = dual_parameters - eta * search_direction
    primal_parameters = get_primal(dual_update)

    return primal_parameters
