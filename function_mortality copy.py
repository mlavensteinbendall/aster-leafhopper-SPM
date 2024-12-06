import numpy as np
import matplotlib.pyplot as plt


def mortality(age, par):
    return np.full(len(age), par)





# def mortality(age_max, age, m, b, constant, linear_function, hill_function):
#     """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
#     Args:
#         age     (array): A list of all the ages
#         m       (int):   A constant for the slope
#         b       (int):   y-intercept
#         constant(bool):  Check whether a constant mortality is wanted or not
        
#     Returns:
#         mu       (array): Represents mortality rate at each age
#     """
#     if constant == True:
#         # Apply constant mortality rate
#         mu = np.full(len(age), m)  # Fill array with constant value of m (assuming m in [0,1])

#     else:
#         if linear_function == True:
#             mu = m * age + b                                # linear function

#         elif hill_function == True:
#             mu = (age / 15) * (30**2 / (30**2 + age**2))    # hill function

#         else:
#             mu = np.exp(-6 * np.exp(-0.15 * age ))
 
#         # Apply mortality based on the linear equation: y = m * (age / age_max) + b
#         # mu = 0.5 *(1 - np.cos(age/age_max * np.pi))

#         # c1 = -1
#         # c2 = 10

#         # mu = 1 + c1 * (c2**2/(c2**2 + age**2))

#     return mu


