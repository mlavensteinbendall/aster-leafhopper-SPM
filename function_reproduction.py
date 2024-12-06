import numpy as np
import matplotlib.pyplot as plt
from function_trapezoidal_rule import trapezoidal_rule

def reproduction(N, age, da, k, type_k):
    """Calculate the reproduction 
    
    Args:
        N     (array): the number of individuals of each age
    
    Returns:
        births (float): number of births
    """

    reproduction_rate = np.full(len(age), 0.0)

    reproduction_rate = k_ind(age, k, type_k)

    # for i in range(0,len(age)):
    #     if age[i] > 10:
    #         p = 2
    #         q = 400
    #         b = 8.0/3.0

    #         total_pop = trapezoidal_rule(N, da)

    #         reproduction_rate[i] = (b * np.exp(-age[i] / 10.0) * q**p) / (q**p + total_pop**p)

    return trapezoidal_rule(reproduction_rate * N, da)


def k_ind(age, k, type_k):
    
    reproduction_rate = np.full(len(age), 0.0)

    if type_k == "constant_reproduction":

        # print("constant reproduction")

        for i in range(0, len(age)):
            if age[i] > 15:
                reproduction_rate[i] = k
    
    if type_k == "linear_reproduction":

        for i in range(0, len(age)):
            if age[i] > 15:
                reproduction_rate[i] = k * age[i]
                # print(reproduction_rate[i])

    if type_k == "gaussian_reproduction":

        for i in range(0, len(age)):
            # if age[i] > 10:
            # reproduction_rate[i] =  np.exp(-(age[i] - 13)**6) 
            reproduction_rate[i] =  1 * np.exp(-(1/5000) * (age[i] - 18)**6) 


    return reproduction_rate


# def k_ind(a, u, age_max):
#     """
#     Function to calculate k, the reproduction rate.
#     Daphnia Manga can have clutches up to 100 eggs every 3-4 days until death.
#     """
#     # Parameters
#     p = 2.0
#     q = 400
#     b = 8.0 / 3.0
    
#     # Calculate total population using the helper function
#     total_pop = total_population(u, age_max)
    
#     # Return k value
#     k = (b * np.exp(-a / 10.0) * q**p) / (q**p + total_pop**p)
#     return k
