import numpy as np # Numpy for numpy

def trapezoidal_rule(fx, dx):
    """Performs trapezoidal rule
    
    Args:
        fx  (array):    A list of the population at different steps.
        dx  (int):      The partition of steps.
        
    Returns:
        result  (float): Represents the time
    """
    
    fx_sum = float(np.sum(fx[1:-1]))    # sum of the middle of the array

    result = ( fx[0] + 2.0 * fx_sum + fx[-1] ) * dx / 2.0 

    return result