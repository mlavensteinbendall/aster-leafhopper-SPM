import numpy as np
import os
import zipfile
from function_trapezoidal_rule import trapezoidal_rule


def mortality(age, par):
    return np.full(len(age), par)                 # constant
    # return par * age                              # linear
    # return 1 / (1 + np.exp(-par* (age - 20)) )    # logistic
    # return (age / 15) * (30**2 / (30**2 + age**2)) # hill

def reproduction(N, age, da, par):
    """Calculate the reproduction 
    
    Args:
        N     (array): the number of individuals of each age
    
    Returns:
        births (float): number of births
    """

    # reproduction_rate = k_dep(age, par)

    return trapezoidal_rule(k_dep(age, par) * N, da)

def k_dep(age, par):

    reproduction_rate = np.full(len(age), 0.0)

    for i in range(0, len(age)):
        # if age[i] > 15:                           #step function                           
        #     reproduction_rate[i] = par            # constant 
            # reproduction_rate[i] = par * age[i]   # linear 

        reproduction_rate[i] =  par * np.exp(-(1/100000) * (age[i] - 20)**6)      # Gaussian

        # reproduction_rate[i] = par / (1 + np.exp(-15 * (age[i] - 10.5)))       # Logistic

        # reproduction_rate[i] = par            # constant

    return reproduction_rate


def solveSPM(par, age, time, da, dt, k, filename, ntag, save_rate):
    """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
    Args:
        age    (array): Array of age values.
        time   (array): Array of time values.
        da      (float): Age step size.
        dt      (float): Time step size.
        Tmax    (float): Maximum time.
        Amax    (float): Maximum age.
        mu      (array): Mortality rates as a function of age.
        k       (float): Reproduction scaling factor.
        type_k  (str): Type of reproduction kernel.
        filename (str): Output filename prefix.
        ntag    (int): Tag for file identification.
    
    Returns:
        None
    """

    print('Running simulation')

    # Initial condition -- population at t=0
    N = np.zeros([len(age)])
    N = np.exp(-(age - 5)**2)            # Gaussian centered at 5

    # Time Splitting -- temporary N's
    Ntemp  = np.zeros([len(age)])      
    Ntemp2 = np.zeros([len(age)])

    # calculate what the mortality term will be
    mu = mortality(age, par)

    count = 0

    # Temporary directory to store individual .npy files
    temp_dir = f"{filename}_temp_{ntag}"
    os.makedirs(temp_dir, exist_ok=True)

    ## NUMERICAL SOLUTION 
    for t in range(0, len(time)-1):  # Loop on times

        # Save the current solution to a temporary .npy file
        if t == 0 or t == len(time) - 1 or t % save_rate == 0:
            np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)

        # Step 1 -- half time step, age Population (Advection)
        for a in range(1, len(age) - 1): 

            first_central_diff = 0
            second_central_diff = 0

            # Central Finite Difference 
            first_central_diff  = (N[a+1]            - N[a-1]) / (2 * da)  # first order finite difference
            second_central_diff = (N[a+1] - 2 * N[a] + N[a-1]) / da**2     # second order finite difference

            Ntemp[a] = N[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

        Ntemp[-1] = N[-2]
        Ntemp[0] = reproduction(Ntemp, age, da, k)


        # Step 2 -- full time-step to decrease populations based on age (Death)
        for a in range(0, len(age)): 
            # RK2 for reaction term with mu(a) as a function of age
            k1        = - Ntemp[a] * mu[a]                       # slope at the beginning of the time step (age = s)
            N_star    =   Ntemp[a] + (dt / 2) * k1               # estimate N at the midpoint of the time step
            k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
            Ntemp2[a] =   Ntemp[a] + dt * k2                     # update N for the full time step


        # Step 3 -- half time-step to age Population (Advection)
        for a in range(1, len(age) - 1):

            first_central_diff = 0
            second_central_diff = 0

            # Central Finite Difference 
            first_central_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2 * da)  # first order finite difference
            second_central_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2     # second order finite difference

            N[a] = Ntemp2[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

        # N[0] = Ntemp2[1]
        N[-1] = Ntemp2[-2]

        # Boundary Condition
        N[0] = reproduction(N, age, da, k)

        count +=1

    # Save the final time step
    np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)

    # Combine all .npy files into a compressed .zip archive
    zip_filename = f"{filename}_results_{ntag}_da_{da}_dt_{dt}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in os.listdir(temp_dir):
            zf.write(os.path.join(temp_dir, file), arcname=file)

    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"Number of iterations: {count}")
