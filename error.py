import numpy                as np
import scipy.special        as sp
import matplotlib.pyplot    as plt 

def stabilize_boundary(start_boundary, boundary, Le, t, t_star):
    """ This function smooths out the boundary condition so that it doesn't cause instability.
    
    Args:
    
    Returns:
    
    """

    return (1 + sp.erf( (- 2 * Le * t / t_star)   +     Le)) * 0.5 * boundary
    # transition_factor = 0.5 * (1 + sp.erf((t - t_star / 2) * Le / t_star))
    # return transition_factor * boundary

    # transition_factor = 0.5 * (1 + sp.erf((t - t_star / 2) * Le / t_star))
    # return start_boundary + (boundary - start_boundary) * transition_factor

    # return 0.5 * boundary * (1 + sp.erf((t-t_star)/Le))


