import numpy as np
import os
import zipfile
from function_trapezoidal_rule import trapezoidal_rule


def mortality(age, par):
    # return np.full(len(age), par)                 # constant
    # return  par * age /30                             # linear
    return 1 / (1 + np.exp(-par* (age - 18)) )    # logistic
    # return (age / 15) * (par**2 / (par**2 + age**2)) # hill

def reproduction(N, age, da, par):
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
        
        # if age[i] >= 15:
        # reproduction_rate[i] = par                                                  # no reproduction
        # else:
        #     reproduction_rate[0] = 0.

    return reproduction_rate

def compute_correction(N, age, da, par):
    """Compute the correction term q based on the boundary condition."""
    # z = np.zeros_like(N)  # Initialize z as zero everywhere
    z = reproduction(N, age, da, par)  # Enforce boundary condition
    return z

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
            N[a] =   Ntemp2[a] + (dt/2) * k2                     # update N for the full time step

        # Boundary Condition
        N[0] = reproduction(N, age, da, k)
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


# def solveSPM(par, age, time, da, dt, k, filename, ntag, save_rate):
#     """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
#     Args:
#         age    (array): Array of age values.
#         time   (array): Array of time values.
#         da      (float): Age step size.
#         dt      (float): Time step size.
#         Tmax    (float): Maximum time.
#         Amax    (float): Maximum age.
#         mu      (array): Mortality rates as a function of age.
#         k       (float): Reproduction scaling factor.
#         type_k  (str): Type of reproduction kernel.
#         filename (str): Output filename prefix.
#         ntag    (int): Tag for file identification.
    
#     Returns:
#         None
#     """

#     print('Running simulation')

#     # Initial condition -- population at t=0
#     N = np.zeros([len(age)])
#     N = np.exp(-(age - 5)**2)            # Gaussian centered at 5

#     # Time Splitting -- temporary N's
#     Ntemp  = np.zeros([len(age)])      
#     Ntemp2 = np.zeros([len(age)])

#     first_central_diff = 0.0
#     second_central_diff = 0.0

#     # calculate what the mortality term will be
#     mu = mortality(age, par)
#     # print(mu)

#     count = 0

#     # Temporary directory to store individual .npy files
#     temp_dir = f"{filename}_temp_{ntag}"
#     os.makedirs(temp_dir, exist_ok=True)

#     ## NUMERICAL SOLUTION 
#     for t in range(0, len(time)-1):  # Loop on times

#         # Save the current solution to a temporary .npy file
#         if t == 0 or t == len(time) - 1 or t % save_rate == 0:
#             np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)


#         # Step 1 -- Solve Dz_0 = 0 using the boundary condition
#         z = reproduction(N, age, da, k)
#         f_z = - mu * z 

#         # Step 2 -- Compute the intial value
#         Ntest = N.copy() 
#         Ntest[0] = z

#         # Step 3 -- Compute the solution of the advection with Nstar
#         for a in range(1, len(age) - 1): 
#             # Central Finite Difference 
#             first_central_diff  = (Ntest[a+1]            - Ntest[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (Ntest[a+1] - 2 * Ntest[a] + Ntest[a-1]) / da**2     # second order finite difference

#             # Ntemp[a] = Ntest[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff
#             Ntemp[a] = Ntest[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff + f_z[a] - (z - compute_correction(Ntemp, age, da, k)) / (dt/2)

#         Ntemp[-1] = 0
#         Ntemp[0] = 0

#         # Step 4 -- full time-step to decrease populations based on age (Death)
#         for a in range(0, len(age)): 
#             # RK2 for reaction term with mu(a) as a function of age
#             k1        = - (Ntemp[a] + z) * mu[a] - f_z[a]                      # slope at the beginning of the time step (age = s)
#             N_star    =   Ntemp[a] + (dt / 2) * k1               # estimate N at the midpoint of the time step
#             k2        = - (N_star + z)   * mu[a] - f_z[a]                       # slope at the midpoint (age still = s)
#             Ntemp2[a] =   Ntemp[a] + dt * k2                     # update N for the full time step


#         # Step 5 -- Compute the solution of advection 
#         for a in range(1, len(age) - 1):
#             # Central Finite Difference 
#             first_central_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2     # second order finite difference

#             z0 = compute_correction(Ntemp2, age, da, k)
#             # N[a] = Ntemp2[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff
#             N[a] = Ntemp2[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff + f_z[a] - (z - z0) / (dt/2)

#         N[-1] = 0
#         N[0] = reproduction(N, age, da, k)  # Apply final correction

#         count +=1

#     # Save the final time step
#     np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)
#     print(f"last time is {time[t+1]} and in step_{t}.npy")

#     # Combine all .npy files into a compressed .zip archive
#     zip_filename = f"{filename}_results_{ntag}_da_{da}_dt_{dt}.zip"
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
#         for file in os.listdir(temp_dir):
#             zf.write(os.path.join(temp_dir, file), arcname=file)

#     # Clean up temporary files
#     for file in os.listdir(temp_dir):
#         os.remove(os.path.join(temp_dir, file))
#     os.rmdir(temp_dir)

#     print(f"Number of iterations: {count}")

# import numpy as np
# import os
# import zipfile
# from function_trapezoidal_rule import trapezoidal_rule


# def mortality(age, par):
#     return np.full(len(age), par)                 # constant
#     # return  .5 * age /30                             # linear
#     # return 1 / (1 + np.exp(-1* (age - 18)) )    # logistic
#     # return (age / 15) * (30**2 / (30**2 + age**2)) # hill

# def reproduction(N, age, da, par):
#     """Calculate the reproduction 
    
#     Args:
#         N     (array): the number of individuals of each age
    
#     Returns:
#         births (float): number of births
#     """

#     # reproduction_rate = k_dep(age, par)

#     # print(k_dep(age, par) * N)

#     return trapezoidal_rule(k_dep(age, par) * N, da)

# def k_dep(age, par):

#     reproduction_rate = np.full(len(age), 0.0)

#     for i in range(0, len(age)):

#         # reproduction_rate[i] =  par * np.exp(-(1/100000) * (age[i] - 20)**6)      # Gaussian
#         # reproduction_rate[i] = par / (1 + np.exp(-15 * (age[i] - 10)))            # Logistic but basiclly a constant
#         reproduction_rate[i] = 5 / (1 + np.exp(-1 * (age[i] - 20)))               # Logistic
#         # reproduction_rate[i] = 1 / (1 + np.exp(-1 * (age[i] - 13)))               # Logistic
#         # reproduction_rate[i] = 0                                                  # no reproduction

#         # reproduction_rate[i] = par            # constant

#     return reproduction_rate

# def compute_correction(N, age, da, mu, par):
#     """Compute the correction term q based on the boundary condition."""
#     q = np.zeros_like(N)  # Initialize q as zero everywhere
#     q[0] = -mu[0] * N[0] + mu[0] * reproduction(N, age, da, par)  # Enforce boundary condition
#     return q


# def solveSPM(par, age, time, da, dt, k, filename, ntag, save_rate):
#     """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
#     Args:
#         age    (array): Array of age values.
#         time   (array): Array of time values.
#         da      (float): Age step size.
#         dt      (float): Time step size.
#         Tmax    (float): Maximum time.
#         Amax    (float): Maximum age.
#         mu      (array): Mortality rates as a function of age.
#         k       (float): Reproduction scaling factor.
#         type_k  (str): Type of reproduction kernel.
#         filename (str): Output filename prefix.
#         ntag    (int): Tag for file identification.
    
#     Returns:
#         None
#     """

#     print('Running simulation')

#     # Initial condition -- population at t=0
#     N = np.zeros([len(age)])
#     N = np.exp(-(age - 5)**2)            # Gaussian centered at 5

#     # Time Splitting -- temporary N's
#     Ntemp  = np.zeros([len(age)])      
#     Ntemp2 = np.zeros([len(age)])

#     first_central_diff = 0.0
#     second_central_diff = 0.0

#     # calculate what the mortality term will be
#     mu = mortality(age, par)
#     # print(mu)

#     count = 0

#     # Temporary directory to store individual .npy files
#     temp_dir = f"{filename}_temp_{ntag}"
#     os.makedirs(temp_dir, exist_ok=True)

#     ## NUMERICAL SOLUTION 
#     for t in range(0, len(time)-1):  # Loop on times

#         # print(N[0])

#         # Save the current solution to a temporary .npy file
#         if t == 0 or t == len(time) - 1 or t % save_rate == 0:
#             np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)

#         # Step 1 -- half time step, age Population (Advection)
#         # rep = reproduction(N, age, da, k) 
#         # bc = rep/ (1- (time[t+1] - time[t])/2 * rep)

#         Ntemp[0] = reproduction(N, age, da, k)   # boundary condition works for 

#         # N[0] = reproduction(N, age, da, k)          # boundary condition
#         # Ntemp[0] = N[0]    

#         for a in range(1, len(age) - 1): 
#             # Central Finite Difference 
#             first_central_diff  = (N[a+1]            - N[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (N[a+1] - 2 * N[a] + N[a-1]) / da**2     # second order finite difference

#             Ntemp[a] = N[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

#         Ntemp[-1] = 0
#         # N[0] = reproduction(N, age, da, k)          # boundary condition
#         # Ntemp[-1] = N[-2]
#         # Ntemp[0] = reproduction(Ntemp, age, da, k)


#         # Step 2 -- full time-step to decrease populations based on age (Death)
#         for a in range(0, len(age)): 
#             # RK2 for reaction term with mu(a) as a function of age
#             k1        = - Ntemp[a] * mu[a]                       # slope at the beginning of the time step (age = s)
#             N_star    =   Ntemp[a] + (dt / 2) * k1               # estimate N at the midpoint of the time step
#             k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
#             Ntemp2[a] =   Ntemp[a] + dt * k2                     # update N for the full time step


#         # Step 3 -- half time-step to age Population (Advection)

#         # rep = reproduction(Ntemp2, age, da, k) 
#         # bc = rep/ (1- (time[t+1] - time[t])/2 * rep)
#         # N[0] = bc

#         # Ntemp2[0] = reproduction(Ntemp2, age, da, k)         # boundary condition
#         # N[0] = Ntemp2[0] #reproduction(Ntemp2, age, da, k)  

#         for a in range(1, len(age) - 1):
#             # Central Finite Difference 
#             first_central_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2     # second order finite difference

#             N[a] = Ntemp2[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

#         # N[-1] = Ntemp2[-2]          # boundary condition
#         N[-1] = 0
#         N[0] = reproduction(N, age, da, k)

#         count +=1

#     # Save the final time step
#     np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)
#     print(f"last time is {time[t+1]} and in step_{t}.npy")

#     # Combine all .npy files into a compressed .zip archive
#     zip_filename = f"{filename}_results_{ntag}_da_{da}_dt_{dt}.zip"
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
#         for file in os.listdir(temp_dir):
#             zf.write(os.path.join(temp_dir, file), arcname=file)

#     # Clean up temporary files
#     for file in os.listdir(temp_dir):
#         os.remove(os.path.join(temp_dir, file))
#     os.rmdir(temp_dir)

#     print(f"Number of iterations: {count}")

# import numpy as np
# import os
# import zipfile
# from function_trapezoidal_rule import trapezoidal_rule


# def mortality(age, par):
#     return np.full(len(age), par)                 # constant
#     # return  age / 30                             # linear
#     # return 1 / (1 + np.exp(-1* (age - 15)) )    # logistic
#     # return (age / 15) * (30**2 / (30**2 + age**2)) # hill

# def reproduction(N, age, da, par):
#     """Calculate the reproduction 
    
#     Args:
#         N     (array): the number of individuals of each age
    
#     Returns:
#         births (float): number of births
#     """

#     return trapezoidal_rule( k_dep(age, par) * N, da)

# def k_dep(age, par):

#     reproduction_rate = np.full(len(age), 0.0)

#     for i in range(0, len(age)):
#         # if age[i] > 15:                           #step function                           
#         #     reproduction_rate[i] = 0.5  * age[i]          # constant 
#         # else:
#         #     age[i] = 0
#             # reproduction_rate[i] = par * age[i]   # linear 

#         # reproduction_rate[i] =  par * np.exp(-(1/100000) * (age[i] - 20)**6)      # Gaussian

#         # reproduction_rate[i] = 1 / (1 + np.exp(-15 * (age[i] - 10.5)))       # Logistic but basiclly a constant
#         reproduction_rate[i] = 5 / (1 + np.exp(-1 * (age[i] - 20)))       # Logistic
#         # reproduction_rate[i] = 1 / (1 + np.exp(-1 * (age[i] - 13)))       # Logistic
#         # reproduction_rate[i] = 0                                          # no reproduction

#         # reproduction_rate[i] = par            # constant

#     return reproduction_rate

# # def compute_correction(N, age, da, mu, par):
# #     """Compute the correction term q based on the boundary condition."""
# #     q = np.zeros_like(N)  # Initialize q as zero everywhere
# #     q[0] = -mu[0] * N[0] + mu[0] * reproduction(N, age, da, par)  # Enforce boundary condition
# #     return q

# def compute_correction(N, age, da, par):
#     """Compute the correction term z based on the boundary condition."""
#     z = np.zeros_like(N)
#     z[0] = reproduction(N, age, da, par)  # Ensures boundary condition is respected
#     return z


# def solveSPM(par, age, time, da, dt, k, filename, ntag, save_rate):
#     """Calculates the numerical solution using strang splitting, lax-wendroff, and runge-kutta method. 
    
#     Args:
#         age    (array): Array of age values.
#         time   (array): Array of time values.
#         da      (float): Age step size.
#         dt      (float): Time step size.
#         Tmax    (float): Maximum time.
#         Amax    (float): Maximum age.
#         mu      (array): Mortality rates as a function of age.
#         k       (float): Reproduction scaling factor.
#         type_k  (str): Type of reproduction kernel.
#         filename (str): Output filename prefix.
#         ntag    (int): Tag for file identification.
    
#     Returns:
#         None
#     """

#     print('Running simulation')

#     # Initial condition -- population at t=0
#     N = np.zeros([len(age)])
#     N = np.exp(-(age - 5)**2)            # Gaussian centered at 5

#     # Time Splitting -- temporary N's
#     Ntemp  = np.zeros([len(age)])      
#     Ntemp2 = np.zeros([len(age)])

#     first_central_diff = 0.0
#     second_central_diff = 0.0

#     # calculate what the mortality term will be
#     mu = mortality(age, par)
#     # print(mu)

#     count = 0

#     # Temporary directory to store individual .npy files
#     temp_dir = f"{filename}_temp_{ntag}"
#     os.makedirs(temp_dir, exist_ok=True)

#     ## NUMERICAL SOLUTION 
#     for t in range(0, len(time)-1):  # Loop on times

#         # print(N[0])

#         # Save the current solution to a temporary .npy file
#         if t == 0 or t == len(time) - 1 or t % save_rate == 0:
#             np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)


#         # # Step 1: Compute correction z_0
#         # z0 = compute_correction(N, age, da, par)
#         # print(z0)

#         # # Step 2: Compute v(0)
#         # v = N - z0  # Homogeneous boundary condition adjustment
#         # # print(v)

#         # # Step 3: Half-step Advection (homogeneous BC)
#         # v_temp = np.zeros_like(N)
#         # for a in range(1, len(age) - 1):
#         #     first_central_diff  = (v[a+1] - v[a-1]) / (2 * da)
#         #     second_central_diff = (v[a+1] - 2 * v[a] + v[a-1]) / da**2
#         #     v_temp[a] = v[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff
        
#         # v_temp[0] = 0  # Homogeneous boundary condition
#         # v_temp[-1] = 0  # Aging-out condition

#         # # Step 4: Full-step Mortality
#         # w = np.zeros_like(N)
#         # for a in range(len(age)):
#         #     k1      = -v_temp[a] * mu[a]
#         #     v_star  = v_temp[a] + (dt / 2) * k1
#         #     k2      = -v_star * mu[a]
#         #     w[a]    = v_temp[a] + dt * k2  # Full mortality step

#         # # Step 5: Second Half-step Advection
#         # v_new = np.zeros_like(N)
#         # for a in range(1, len(age) - 1):
#         #     first_central_diff  = (w[a+1] - w[a-1]) / (2 * da)
#         #     second_central_diff = (w[a+1] - 2 * w[a] + w[a-1]) / da**2
#         #     v_new[a] = w[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

#         # v_new[0] = 0  # Homogeneous boundary condition
#         # v_new[-1] = 0  # Aging-out condition

#         # # Step 6: Compute correction z_1
#         # # z1 = compute_correction(v_new, age, da, par)

#         # # Step 7: Compute final solution u1
#         # N = v_new + z0



#         # lbc = reproduction(N, age, da, k)          # using the current population calculate the boundary condition 

#         # q = compute_correction(N, age, da, mu, par)

#         # Step 1 -- half time-step to decrease populations based on age (Death)
#         # for a in range(0, len(age)): 
#         #     # RK2 for reaction term with mu(a) as a function of age
#         #     k1        = - N[a] * mu[a]                      # slope at the beginning of the time step (age = s)
#         #     N_star    =   N[a] + (dt / 4) * k1               # estimate N at the midpoint of the time step
#         #     k2        = - N_star   * mu[a]                 # slope at the midpoint (age still = s)
#         #     Ntemp[a]  =   N[a] + (dt/2) * k2                  # update N for the full time step


#         # # Step 2 -- full time step, age Population (Advection)
#         # rep = reproduction(Ntemp, age, da, k) 
#         # bc = rep / (1 - (time[t+1] - time[t]) * rep)
#         # Ntemp2[0] = bc  

#         # # Ntemp[0] = reproduction(Ntemp, age, da, k)          # using the current population calculate the boundary condition 
#         # # Ntemp2[0] = Ntemp[0]
#         # for a in range(1,len(age) -1):
#         #     # Central Finite Difference 
#         #     first_central_diff  = (Ntemp[a+1]            - Ntemp[a-1]) / (2 * da)      # first order finite difference
#         #     second_central_diff = (Ntemp[a+1] - 2 * Ntemp[a] + Ntemp[a-1]) / da**2     # second order finite difference

#         #     Ntemp2[a] = Ntemp[a] - (dt) * first_central_diff + (dt**2 / 2) * second_central_diff

#         # Ntemp2[0]  = Ntemp[0] #lbc_advect       # set left boundary condition
#         # Ntemp2[-1] = 0 # Ntemp[-2]      # set right boundary condition

#         # # Step 3 -- half time-step to decrease populations based on age (Death)
#         # for a in range(0, len(age)): 
#         #     # RK2 for reaction term with mu(a) as a function of age
#         #     k1        = - Ntemp2[a] * mu[a]                      # slope at the beginning of the time step (age = s)
#         #     N_star    =   Ntemp2[a] + (dt / 4) * k1              # estimate N at the midpoint of the time step
#         #     k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
#         #     N[a]      =   Ntemp2[a] + (dt/2) * k2                     # update N for the full time step


#         # Step 1 -- half time step, age Population (Advection)
#         # rep = reproduction(N, age, da, k) 
#         # bc = rep/ (1- (time[t+1] - time[t])/2 * rep)

#         N[0] = reproduction(N, age, da, k)          # boundary condition
#         Ntemp[0] = N[0]   

#         for a in range(1, len(age) - 1): 
#             # Central Finite Difference 
#             first_central_diff  = (N[a+1]            - N[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (N[a+1] - 2 * N[a] + N[a-1]) / da**2     # second order finite difference

#             Ntemp[a] = N[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

#         # Ntemp[-1] = N[-2]               # boundary condition
#         Ntemp[-1] = 0

#         # Ntemp[0] = reproduction(Ntemp, age, da, k)    # works for constant mortality, not limited by time max



#         # Step 2 -- full time-step to decrease populations based on age (Death)
#         for a in range(0, len(age)): 
#             # RK2 for reaction term with mu(a) as a function of age
#             k1        = - Ntemp[a] * mu[a]                       # slope at the beginning of the time step (age = s)
#             N_star    =   Ntemp[a] + (dt / 2) * k1               # estimate N at the midpoint of the time step
#             k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
#             Ntemp2[a] =   Ntemp[a] + dt * k2                     # update N for the full time step


#         # Step 3 -- half time-step to age Population (Advection)

#         # rep = reproduction(Ntemp2, age, da, k) 
#         # bc = rep/ (1- (time[t+1] - time[t])/2 * rep)
#         # N[0] = bc

#         Ntemp2[0] = reproduction(Ntemp2, age, da, k)         # boundary condition
#         N[0] = Ntemp2[0] #reproduction(Ntemp2, age, da, k)  

#         for a in range(0, len(age) - 1):
#             # Central Finite Difference 
#             first_central_diff  = (Ntemp2[a+1]                 - Ntemp2[a-1]) / (2 * da)  # first order finite difference
#             second_central_diff = (Ntemp2[a+1] - 2 * Ntemp2[a] + Ntemp2[a-1]) / da**2     # second order finite difference

#             N[a] = Ntemp2[a] - (dt / 2) * first_central_diff + (dt**2 / 8) * second_central_diff

#         # N[-1] = Ntemp2[-2]          # boundary condition
#         N[-1] = 0
#         # N[-1] = Ntemp2[-2]
#         # N[0] = reproduction(N, age, da, k) # works for constant mortality, not limited by time max

#         count +=1

#     # Save the final time step
#     np.save(os.path.join(temp_dir, f"step_{t}.npy"), N)
#     print(f"last time is {time[t+1]} and in step_{t}.npy")

#     # Combine all .npy files into a compressed .zip archive
#     zip_filename = f"{filename}_results_{ntag}_da_{da}_dt_{dt}.zip"
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
#         for file in os.listdir(temp_dir):
#             zf.write(os.path.join(temp_dir, file), arcname=file)

#     # Clean up temporary files
#     for file in os.listdir(temp_dir):
#         os.remove(os.path.join(temp_dir, file))
#     os.rmdir(temp_dir)

#     print(f"Number of iterations: {count}")










        # # Step 1 -- half time-step to decrease populations based on age (Death)
        # for a in range(0, len(age)): 
        #     # RK2 for reaction term with mu(a) as a function of age
        #     k1        = - N[a] * mu[a]                       # slope at the beginning of the time step (age = s)
        #     N_star    =   N[a] + (dt / 4) * k1               # estimate N at the midpoint of the time step
        #     k2        = - N_star   * mu[a]                   # slope at the midpoint (age still = s)
        #     Ntemp[a] =   N[a] + (dt/2) * k2                  # update N for the full time step


        # # Step 2 -- full time step, age Population (Advection)
        # Ntemp[0] = reproduction(Ntemp, age, da, k)          # boundary condition

        # for a in range(1,len(age) -1):
        #     # Central Finite Difference 
        #     first_central_diff  = (Ntemp[a+1]            - Ntemp[a-1]) / (2 * da)      # first order finite difference
        #     second_central_diff = (Ntemp[a+1] - 2 * Ntemp[a] + Ntemp[a-1]) / da**2     # second order finite difference

        #     Ntemp2[a] = Ntemp[a] - (dt) * first_central_diff + (dt**2 / 2) * second_central_diff

        # Ntemp2[-1] = Ntemp[-2]

        # # Step 3 -- half time-step to decrease populations based on age (Death)
        # for a in range(0, len(age)): 
        #     # RK2 for reaction term with mu(a) as a function of age
        #     k1        = - Ntemp2[a] * mu[a]                      # slope at the beginning of the time step (age = s)
        #     N_star    =   Ntemp2[a] + (dt / 4) * k1              # estimate N at the midpoint of the time step
        #     k2        = - N_star   * mu[a]                       # slope at the midpoint (age still = s)
        #     N[a] =   Ntemp2[a] + (dt/2) * k2                     # update N for the full time step