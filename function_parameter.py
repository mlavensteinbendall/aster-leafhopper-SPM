import numpy as np
from scipy.optimize import minimize     # Allows min (or max) objective functions
import matplotlib.pyplot as plt

from function_model_parm import solveSPM
from function_trapezoidal_rule import trapezoidal_rule
from function_figures                   import plt_mortality_func, plt_reproduction_rate_func, plt_total_pop, plt_boundary_condition, plt_numerical_solution

import zipfile
import re
import os
# import time

# time.sleep(1)  # Wait for file to be written

# Define OLS error estimation function
def ols_error_estimate(par, age, time, da, dt, filename, ntag, save_rate, data):
    # Solve the SPM with current parameters
    solveSPM(par, age, time, da, dt, filename, ntag, save_rate)

    # Specify the ZIP file and extraction directory
    zip_filename = filename + f"_results_{ntag}_da_{da}_dt_{dt}.zip"
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

    sol = np.stack(solutions, axis = 0)
    
    # Calculate residuals with proportional scaling
    resids = data - sol
    print(resids)
    
    # Return the sum of squared residuals (OLS cost)
    return np.sum(resids**2)

# Define GLS error estimation function
def gls_error_estimate(par, age, time, da, dt, filename, ntag, save_rate, prop_data, weights):
    
    # Solve the SPM with current parameters
    solveSPM(par, age, time, da, dt, filename, ntag, save_rate)

    # Specify the ZIP file and extraction directory
    zip_filename = filename + f"_results_{ntag}_da_{da}_dt_{dt}.zip"
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

    sol = np.stack(solutions, axis = 0)

    # Calculate residuals with proportional scaling
    resids = (prop_data - sol) / (sol**gammas)
    
    weighted_resids = resids * weights
    
    return np.sum(weighted_resids**2)

# Create synthetic data to do parameter estimation
Amax = 30      # max age
Tmax = 20      # max time

da = 0.025
dt = 0.5 * da

# initalize age array
age_num_points = int(Amax / da) + 1
age = np.linspace(0, Amax, age_num_points)
Nage = len(age)                               # number of elements in age
print("Age:", age[-1])                        # check that last element is Amax


# initalize time array
time_num_points = int(Tmax / dt) + 1
time = np.linspace(0, Tmax, time_num_points)
print("Time:", time[-1])                        # check that last element is Tmax

inital_parm = [1, 0.4]
save_rate = 15

folder = "parameter/test"
ntag = 0

solveSPM(inital_parm, age, time, da, dt, folder, ntag, save_rate)

# Specify the ZIP file and extraction directory
zip_filename = folder + f"_results_{ntag}_da_{da}_dt_{dt}.zip"
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


plt_numerical_solution(age, time, da, dt, inital_parm[0], ntag, folder, save_rate)

# print(f"Shape of data: {data.shape}")


# total_pop = trapezoidal_rule(data[:,], da)
# print(f"Shape of total_pop: {total_pop.shape}")



# Define the proportional error for synthetic data generation
sigma = 0.1  # Set the proportional error to 10% of the true value
# Generate synthetic data with proportional error
# Each data point in 'y' is scaled by (1 + a random error term from a normal distribution scaled by sigma)
prop_data = data * (1 + sigma * np.random.randn(*data.shape))
# prop_data = (total_pop * np.array([(1 + sigma * np.random.randn(total_pop.shape[0]))])).T
# print(f"Shape of prop_data: {prop_data.shape}")
# print(prop_data)


# set initial parameter guess
init_guess = np.array([.6, .1])


# GLS Formulation with proportional error data
gammas = 1
weights = np.ones(prop_data.shape)
gls_optpar = init_guess
old_gls_optpar = init_guess
tol = 1e-4
maxits = 5000
minits = 10
partol = 0.1
parchange = 100
oldparchange = 100
ii = 1

folder = "parameter/test_min"
ntag = 1

while (ii < maxits and parchange > partol and oldparchange > partol) or (ii < minits):

    ops = {'maxiter':5000,'disp':False}
    gls_optpar = minimize(gls_error_estimate, gls_optpar, args=(age, time, da, dt, folder, ntag, save_rate, prop_data, weights), options=ops).x
    # w = solve_ivp(logistic, [t0,tf], [y0], t_eval=ts,args=(gls_optpar,))
    solveSPM(gls_optpar, age, time, da, dt, folder, ntag, save_rate)

    # Specify the ZIP file and extraction directory
    zip_filename = folder + f"_results_{ntag}_da_{da}_dt_{dt}.zip"
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

    weights = np.stack(solutions, axis = 0)

    # weights = w.y.T
    weights[weights < tol] = 0
    weights[weights > tol] = weights[weights > tol]**(-2 * gammas)
    inds = old_gls_optpar > 1e-10
    parchange = 1 / 2 * np.sum(np.abs(gls_optpar[inds] - old_gls_optpar[inds]) / old_gls_optpar[inds])
    ii = ii + 1
    old_gls_optpar = gls_optpar
    print(f"test param: {gls_optpar}")

    # Print the parameters
print('GLS Estimation')
print(gls_optpar)

# Residual Examination
solveSPM(gls_optpar, age, time, da, dt, folder, ntag, save_rate)

# Specify the ZIP file and extraction directory
zip_filename = folder + f"_results_{ntag}_da_{da}_dt_{dt}.zip"
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

gls_sol = np.stack(solutions, axis = 0)


gls_resids = (prop_data - gls_sol) / (gls_sol**gammas)

# Plot the Residuals versus model value and time
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(gls_resids, 'r*')
plt.title('GLS')
plt.xlabel('Time (Days)')


plt.subplot(1, 2, 2)
plt.plot(gls_sol, gls_resids, 'r*')
plt.xlabel('Model Value')
plt.show()