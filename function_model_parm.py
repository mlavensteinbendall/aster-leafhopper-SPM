import numpy as np
import os
import zipfile
from function_trapezoidal_rule import trapezoidal_rule


def mortality(par, age):
    # return np.full(len(age), par)                 # constant
    # return  par * age /30                             # linear
    return 1 / (1 + np.exp(-par* (age - 18)) )    # logistic
    # return (age / 15) * (par**2 / (par**2 + age**2)) # hill

def reproduction(par, N, age, da):
    """Calculate the reproduction 
    
    Args:
        N     (array): the number of individuals of each age
    
    Returns:
        births (float): number of births
    """

    # reproduction_rate = k_dep(age, par)

    # print(k_dep(age, par) * N)

    return trapezoidal_rule(k_dep(age, par) * N, da)

def k_dep(age, par):

    reproduction_rate = np.full(len(age), 0.0)

    for i in range(0, len(age)):

        reproduction_rate[i] =  par * np.exp(-(1/100000) * (age[i] - 20)**6)      # Gaussian
        # reproduction_rate[i] = par / (1 + np.exp(-15 * (age[i] - 10)))            # Logistic but basiclly a constant
        # reproduction_rate[i] = 5 / (1 + np.exp(-1 * (age[i] - 20)))               # Logistic
        # reproduction_rate[i] = par                                                # no reproduction

    return reproduction_rate


def solveSPM(params, age, time, da, dt, filename, ntag, save_rate):
# def solveSPM(par, age, time, da, dt, k, filename, ntag, save_rate):
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
    mu = mortality(params[0], age)

    count = 0

    # Temporary directory to store individual .npy files
    temp_dir = f"{filename}_temp_{ntag}"
    os.makedirs(temp_dir, exist_ok=True)

    ## NUMERICAL SOLUTION 
    for t in range(0, len(time)-1):  # Loop on times

        # Save the current solution to a temporary .npy file
        if t == 0 or t == len(time) - 1 or t % save_rate == 0:
            np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)

        # Step 1 -- half time step, diffuse population (mortality)
        for a in range(0, len(age)-1): 
            # RK2 for reaction term with mu(a) as a function of age
            k1        = - N[a] * mu[a]                       # slope at the beginning of the time step (age = s)
            N_star    =   N[a] + (dt / 4) * k1               # estimate N at the midpoint of the time step
            k2        = - N_star * mu[a]                     # slope at the midpoint (age still = s)
            Ntemp[a]  =   N[a] + (dt/2) * k2                     # update N for the full time step

        # Step 2 -- full time-step, advect population (age)
        for a in range(1, len(age) - 1):
            # Central Finite Difference 
            first_central_diff  = (Ntemp[a+1]                 - Ntemp[a-1]) / (2 * da)  # first order finite difference
            second_central_diff = (Ntemp[a+1] - 2 * Ntemp[a] + Ntemp[a-1]) / da**2     # second order finite difference

            Ntemp2[a] = Ntemp[a] - dt * first_central_diff + (dt**2 / 2) * second_central_diff

        # Step 2 -- half time step, diffuse population (mortality)
        for a in range(0, len(age)-1): 
            # RK2 for reaction term with mu(a) as a function of age
            k1        = - Ntemp2[a] * mu[a]                       # slope at the beginning of the time step (age = s)
            N_star    =   Ntemp2[a] + (dt / 4) * k1               # estimate N at the midpoint of the time step
            k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
            N[a]      =   Ntemp2[a] + (dt/2) * k2                     # update N for the full time step

        # Boundary Condition
        N[0] = reproduction(params[1], N, age, da)
        N[-1] = 0

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
