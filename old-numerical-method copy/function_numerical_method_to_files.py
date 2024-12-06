import numpy as np # Numpy for numpy
from function_reproduction import reproduction

def solveSPM_file(age, time, da, dt, Tmax, Amax, mu, k, type_k, filename, ntag):
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
    N = np.zeros([len(age)])
    N[:] = np.exp(-(age - 5)**2) 

    # Time Splitting
    Ntemp  = np.zeros([len(age)])
    Ntemp2 = np.zeros([len(age)])

    print('file')
    print(age[-1])
    print(time[-1])

    count = 0

    ## NUMERICAL SOLUTION 
    with open(filename + 'test_' + str(ntag) + '.txt', 'w') as file: # Initialise an outputter file (safe)
        for t,T in enumerate(time): # Loop on times

            for n in N: # Output the current time solution
                file.write(str(n))
                file.write(" ")
            file.write("\n")


            # Step 1 -- half time step, age Population (Advection)
            for a in range(1,len(age)-1): 

                first_centeral_diff = 0
                second_centeral_diff = 0

                # Centeral Finite Difference 
                # --- We are using the current time-step to make a half-time step for each age a
                first_centeral_diff  = (N[a+1]            - N[a-1]) / (2*da) # first order finite difference
                second_centeral_diff = (N[a+1] - 2 * N[a] + N[a-1]) / da**2  # second order finite difference

                Ntemp[a] = N[a] - (dt/2) * first_centeral_diff + (dt**2/8) * second_centeral_diff


            # Step 2 -- full time-step to decrease populations based on age (Death)
            for a in range(0,len(age)): 
            # for a in range(1,len(age)-1): 

                # RK2 for reaction term with mu(a) as a function of age
                # --- We are using the half-time step to move fully an age 
                k1        = - Ntemp[a] *  mu[a]                       # slope at the beginning of the time step (age = s)
                N_star    =   Ntemp[a] + (dt / 2) * k1                # estimate N at the midpoint of the time step
                k2        = - N_star   *  mu[a]                       # slope at the midpoint (age still = s)
                Ntemp2[a] =   Ntemp[a] +  dt  * k2                    # update N for the full time step


            # Step 3 -- half time-step to age Population (Advection)
            for a in range(1,len(age)-1):

                first_centeral_diff = 0
                second_centeral_diff = 0

                # Centeral Finite Difference 
                # --- We are using the full time-step for death to make a half-time step for each age a
                first_centeral_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2*da) # first order finite difference
                second_centeral_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2  # second order finite difference

                N[a] = Ntemp2[a] - (dt/2) * first_centeral_diff + (dt**2/8) * second_centeral_diff


            # Boundary Condition
            N[0]  = reproduction(N, age, da, k, type_k)
            Ntemp[0] = N[0]

            count += 1

        for n in N: # Output the final time solution
            file.write(str(n))
            file.write(" ")
        file.write("\n")

    print(count)


                
