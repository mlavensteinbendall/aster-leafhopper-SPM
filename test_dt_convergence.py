import numpy                        as np
import matplotlib.pyplot            as plt 
from function_model import solveSPM
from function_figures                   import plt_mortality_func, plt_reproduction_rate_func, plt_total_pop, plt_boundary_condition, plt_numerical_solution
import zipfile
import re

import timeit

# from function_convergence_w_exact_sol import convergence_dt_plt, conservation_plt
from function_convergence             import convergence_dt_plt, conservation_plt
from function_convergence_table                 import tabulate_conv

# Start timer
start = timeit.default_timer()

## INITIAL CONDITIONS
Amax = 30      # max age
Tmax = 20      # max time
order = 2       # order of method
Ntest = 5       # number of cases

save_rate = 15  # save the first, last, and mod this number

k = 0           # reproduction parameter
par = 0         # mortality parameter

# need to chose da and dt so that the last value in the array are Amax and Tmax
da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
da[0] = 0.1
da[1] = 0.05
da[2] = 0.025
da[3] = 0.0125
da[4] = 0.00625

dt = 0.5 * da

## TEST MORTALITY ---------------------------------------------------
# have to go into function_model.py to change which mortality to test

# no mortality
# folder = 'convergence/mortality/no_mortality/varied_dt/no-mortality'
# k = 0
# par = 0      # mortality parameter

# constant mortality
# folder = 'convergence/mortality/half_constant/varied_dt/half-constant'
# k = 0
# par = 0.5      # mortality parameter
## folder = 'convergence/mortality/constant/varied_dt/tenth-constant'
## par = 0.1      # mortality parameter

# linear mortality 
# folder = 'convergence/mortality/linear/varied_dt/linear'
# par = 1. 

# logistic mortality
# folder = 'convergence/mortality/logistic/varied_dt/logistic'
# k = 0
# par = 0.4 

# hill mortality
# folder = 'convergence/mortality/hill/varied_dt/hill'
# k = 0
# par = 30. 


## TEST Reproduction ---------------------------------------------------
# have to go into function_model.py to change which mortality to test

# constant reproduction
# folder = 'convergence/reproduction/constant/varied_dt/tenth-constant'
# k = 0.1     # reproduction parameter
# folder = 'convergence/reproduction/half_constant/varied_dt/half-constant'
# k = 0.5     # reproduction parameter
# par = 0.

# logistic reproduction
# folder = 'convergence/reproduction/logistic/varied_dt/logistic'
# k = 1 

# gaussian reproduction
# folder = 'convergence/reproduction/gaussian/varied_dt/gaussian'
# k = 1

## TEST Both ------------------------------------------------------------

# # constant mortality and logistic reproduction
# test = "constant_m_logistic_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = 0.05      # mortality parameter

# linear mortality and logistic reproduction
# test = "linear_m_logistic_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = .5      # mortality parameter

# logistic mortality and logistic reproduction
# test = "logistic_m_logistic_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = 0.4      # mortality parameter

# hill mortality and logistic reproduction
# test = "hill_m_logistic_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = 30.      # mortality parameter

# # constant mortality and gaussian reproduction
# test = "constant_m_gaussian_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = 0.05      # mortality parameter

# linear mortality and gaussian reproduction
# test = "linear_m_gaussian_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = .5      # mortality parameter

# logistic mortality and gaussian reproduction
test = "logistic_m_gaussian_r"
folder = 'convergence/both/' + test +'/varied_dt/' + test
k = 1        # reproduction parameter
par = 0.4      # mortality parameter

# hill mortality and gaussian reproduction
# test = "hill_m_gaussian_r"
# folder = 'convergence/both/' + test +'/varied_dt/' + test
# k = 1        # reproduction parameter
# par = 30.      # mortality parameter



# Using the given da and dt values, this loop calculates the numerical solution, solve the analytical 
# solution, and plots the numerical vs. analytical solution. 
# BEGIN LOOP ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(Ntest):

    print('Entering loop ' + str(i))                # progress update, loop began

    # initalize age array
    age_num_points = int(Amax / da[i]) + 1
    age = np.linspace(0, Amax, age_num_points)
    Nage = len(age)                               # number of elements in age
    print("Age:", age[-1])                        # check that last element is Amax


    # initalize time array
    time_num_points = int(Tmax / dt[i]) + 1
    time = np.linspace(0, Tmax, time_num_points)
    print("Time:", time[-1])                        # check that last element is Tmax

    # check CFL condition
    lambda_cfl = dt[i] / da[i]
    if lambda_cfl > 1:
        raise ValueError("CFL condition violated! Reduce dt or increase da.")
    else: print('CFL: ' + str(round(lambda_cfl, 5)))

    # calculate solution
    solveSPM(par, age, time, da[i], dt[i], k, folder, i, save_rate)


    # Specify the ZIP file and extraction directory
    zip_filename = folder + f"_results_{i}_da_{da[i]}_dt_{dt[i]}.zip"
    output_dir = "unzipped_output"  # Directory to extract files

    # Open the ZIP file
    with zipfile.ZipFile(zip_filename, 'r') as zf:

        # List all files in the ZIP
        file_list = zf.namelist()
        
        # Sort the files (important if time steps should be in order)
        file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

        # print(file_list)
        
        # Read each file into memory
        solutions = []
        for file in file_list:
            with zf.open(file) as f:
                solutions.append(np.load(f))  # Load the .npy file into a numpy array

    data = np.stack(solutions, axis = 0)


    # Now `solutions` is a list of numpy arrays, one for each time step
    print(f"Loaded {len(solutions)} time steps.")

    plt_mortality_func(age, par, folder)
    plt_reproduction_rate_func(age, k, folder)
    plt_numerical_solution(age, time, da[i], dt[i], par, i, folder, save_rate)

    plt_total_pop(data, time, da[i], dt[i], folder, save_rate)
    plt_boundary_condition(data, time, da[i], dt[i], folder, save_rate)


    print('Loop ' + str(i) + ' Complete.')                      # progress update, loop end


# END LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



## CONVERGENCE 
# Calculate and plot the convergence, returns an matrix with Norm2, L2norm, NormMax, and LMaxnorm
# Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Amax, Tmax, da, dt, par, folder)
Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(da, dt, folder)

## TOTAL POPULATION ERROR 
# Checks conservation, returns norm and order of conservation
# Norm1, L1norm = conservation_plt(da, dt, par, Amax, Tmax, folder)
Norm1, L1norm = conservation_plt(da, dt, folder)

## PRINT NORMS 
# print table that can be added to latex
tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)


## RUNTIME 
# print the time it took to simulation
stop = timeit.default_timer()

print(f"Simulation took {round((stop - start)/ 60, 2)} minutes to run.")
