import numpy as np # Numpy for numpy
from function_reproduction import reproduction


def solveSPM(age, time, da, dt, mu, k, type_k):
    """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
    Args:
        age    (array): 
        time    (array):
        da      (int):
        dt      (int):
        mu      (array):
        
    Returns:
        N       (array of arrays): Represents the time
    """

    # inital condition -- population at t=0
    N = np.zeros([len(time),len(age)])
    N[0,:] = np.exp(-(age - 5)**2) 

    # Time Splitting
    Ntemp  = np.zeros([len(age)])
    Ntemp2 = np.zeros([len(age)])

    print('indirect')
    print(age)
    print(time)

    ## NUMERICAL SOLUTION 
    for t in range(0, len(time)-1):

        # Time Splitting
        Ntemp  = N[t,:]
        Ntemp2 = N[t,:]


        # Step 1 -- half time-step to Age Population (Advection)
        for a in range(1,len(age)-1): 

            first_centeral_diff = 0
            second_centeral_diff = 0

            # Centeral Finite Difference 
            # --- We are using the current time-step to make a half-time step for each age a
            first_centeral_diff  = (N[t,a+1]              - N[t,a-1]) / (2*da)
            second_centeral_diff = (N[t,a+1] - 2 * N[t,a] + N[t,a-1]) / da**2

            Ntemp[a] = N[t,a] - (dt/2) * first_centeral_diff + (dt**2/8) * second_centeral_diff


        # Step 2 -- full time-step to decrease populations based on age (Death)
        for a in range(0,len(age)): 
        # for a in range(1,len(age)-1): 

            # RK2 for reaction term with mu(a) as a function of age
            # --- We are using the half-time step to move fully an age 
            k1        = - Ntemp[a] *  mu[a]                       # slope at the beginning of the time step (age = s)
            N_star    =   Ntemp[a] + (dt / 2) * k1                # estimate N at the midpoint of the time step
            k2        = - N_star   *  mu[a]                       # slope at the midpoint (age still = s)
            Ntemp2[a] =   Ntemp[a] +  dt  * k2                    # update N for the full time step


        # age 3 -- half time-step to Age Population (Advection)
        for a in range(1,len(age)-1):

            first_centeral_diff = 0
            second_centeral_diff = 0

            # Centeral Finite Difference 
            # --- We are using the full time-step for death to make a half-time step for each age a
            first_centeral_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2*da) # first order finite difference
            second_centeral_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2  # second order finite difference

            N[t+1, a] = Ntemp2[a] - (dt/2) * first_centeral_diff + (dt**2/8) * second_centeral_diff


        # Boundary Condition
        N[t+1, 0] = reproduction(N[t, :], age, da, k, type_k)
                
    return N
