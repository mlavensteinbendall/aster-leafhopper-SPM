import numpy                        as np
import matplotlib.pyplot            as plt 
from function_model import solveSPM
from function_figures                   import plt_mortality_func, plt_reproduction_rate_func, plt_total_pop, plt_boundary_condition, plt_numerical_solution
import zipfile
import re

import timeit

# from function_convergence_w_exact_sol import convergence_da_plt, conservation_plt
from function_convergence             import convergence_da_plt, conservation_plt
from function_convergence_table       import tabulate_conv

# Start timer
start = timeit.default_timer()

## INITIAL CONDITIONS
Amax = 30      # max age
Tmax = 20      # max time
order = 2       # order of method
Ntest = 5       # number of cases

save_rate = 15  # save the first, last, and mod this number

k = 0           # reproduction parameter
par = 0      # mortality parameter

# need to chose da and dt so that the last value in the array are Amax and Tmax
da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
da[0] = 0.1
da[1] = 0.05
da[2] = 0.025
da[3] = 0.0125
da[4] = 0.00625

dt = 0.001

## TEST MORTALITY ---------------------------------------------------
# have to go into function_model.py to change which mortality to test

# no mortality
# folder = 'convergence/mortality/no_mortality/fixed_dt/no-mortality'
# par = 0      # mortality parameter

# constant mortality
# folder = 'convergence/mortality/constant/fixed_dt/half-constant'
# par = 0.5      # mortality parameter
# folder = 'convergence/mortality/constant/fixed_dt/tenth-constant'
# par = 0.1      # mortality parameter

# linear mortality 
# folder = 'convergence/mortality/linear/fixed_dt/linear'
# par = 1. / Amax

# logistic mortality
# folder = 'convergence/mortality/logistic/fixed_dt/logistic'
# par = 0.4 

# hill mortality
# folder = 'convergence/mortality/hill/fixed_dt/hill'
# par = 0 # doesn't matter

## TEST Reproduction ---------------------------------------------------
# have to go into function_model.py to change which mortality to test

# constant reproduction
# folder = 'convergence/reproduction/constant/fixed_dt/tenth-constant'
# k = 0.1     # reproduction parameter
# folder = 'convergence/reproduction/constant/fixed_dt/half-constant'
# k = 0.5     # reproduction parameter

# logistic reproduction
# folder = 'convergence/reproduction/logistic/fixed_dt/logistic'
# k = 1 

# gaussian reproduction
folder = 'convergence/reproduction/gaussian/fixed_dt/gaussian'
k = 1

# initalize time array
time_num_points = int(Tmax / dt) + 1
time = np.linspace(0, Tmax, time_num_points)
print("Time:", time[-1])                        # check that last element is Tmax

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

    # print CFL
    print('CFL: ' + str(round(dt/da[i], 5)))

    # calculate solution
    solveSPM(par, age, time, da[i], dt, k, folder, i, save_rate)


#     # Specify the ZIP file and extraction directory
    zip_filename = folder + f"_results_{i}_da_{da[i]}_dt_{dt}.zip"
#     output_dir = "unzipped_output"  # Directory to extract files

#     # Open the ZIP file
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

    # save plots
    plt_mortality_func(age, par, folder)
    plt_reproduction_rate_func(age, k, folder)
    plt_numerical_solution(age, time, da[i], dt, par, i, folder, save_rate)

    plt_total_pop(data, time, da[i], dt, folder, save_rate)
    plt_boundary_condition(data, time, da[i], dt, folder, save_rate)

    print('Loop ' + str(i) + ' Complete.')                      # progress update, loop end

    

# END LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## CONVERGENCE 
# Calculate and plot the convergence, returns an matrix with Norm2, L2norm, NormMax, and LMaxnorm
# Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(Amax, Tmax, da, dt, par, folder)  # exact solution
Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(da, dt, folder)

## TOTAL POPULATION ERROR 
# Checks conservation, returns norm and order of conservation
# Norm1, L1norm = conservation_plt(da, dt, par, Amax, Tmax, folder) # exact solution 
Norm1, L1norm = conservation_plt(da, dt, folder)

## PRINT NORMS 
# print table that can be added to latex
tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)


## RUNTIME 
# print the time it took to simulation
stop = timeit.default_timer()

print(f"Simulation took {round((stop - start)/ 60, 2)} minutes to run.")
