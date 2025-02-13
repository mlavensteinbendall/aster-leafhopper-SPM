import numpy as np
from scipy.optimize import minimize     # Allows min (or max) objective functions
import matplotlib.pyplot as plt

from function_model_parm import solveSPM
from function_trapezoidal_rule import trapezoidal_rule
from function_figures                   import plt_mortality_func, plt_reproduction_rate_func, plt_total_pop, plt_boundary_condition, plt_numerical_solution

import zipfile
import re
import os
import timeit

start = timeit.default_timer()

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
def gls_error_estimate(par, age, time, da, dt, filename, ntag, save_rate, prop_total, weights):
    
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

    sol_total = trapezoidal_rule(sol[:,], da)
    print(f"shape of sol_total: {sol_total.shape}")

    # Calculate residuals with proportional scaling
    resids = (prop_total - sol_total) / (sol_total**gammas)
    
    weighted_resids = resids * weights
    
    return np.sum(weighted_resids**2)

# Create synthetic data to do parameter estimation
Amax = 30      # max age
Tmax = 15      # max time

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

saved_times = []  # List to store time values for saved steps

for t in range(0, len(time)-1):  # Loop on times
    if t == 0 or t == len(time)-1 or t % save_rate == 0:
        saved_times.append(time[t])  # Store the corresponding time

print(saved_times)
saved_time = np.array(saved_times)
print(f"shape of saved_time: {saved_time.shape}")
print(f"shape of time: {time.shape}")
print(f"length of time: {len(time)}")
times = np.linspace(0,Tmax, 1201)

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

total_pop = np.array([trapezoidal_rule(data[t, :], da) for t in range(data.shape[0])])
print(f"Shape of total_pop: {total_pop.shape}")

# Define the proportional error for synthetic data generation
sigma = 0.1  # Set the proportional error to 10% of the true value
# Generate synthetic data with proportional error
# Each data point in 'y' is scaled by (1 + a random error term from a normal distribution scaled by sigma)
# prop_data = data * (1 + sigma * np.random.randn(*data.shape))
prop_data = (total_pop * np.array([(1 + sigma * np.random.randn(total_pop.shape[0]))])).T
print(f"Shape of prop_data: {prop_data.shape}")
print(prop_data)

# Temporary directory to store individual .npy files
temp_dir = f"prop_data"
os.makedirs(temp_dir, exist_ok=True)
np.save(os.path.join(temp_dir, f".npy"), prop_data)


# set initial parameter guess
init_guess = np.array([.7, .2])


# GLS Formulation with proportional error data
gammas = 1
weights = np.ones(prop_data.shape)
gls_optpar = init_guess
old_gls_optpar = init_guess
tol = 1e-4
maxits = 2500
minits = 10
partol = 0.1
parchange = 100
oldparchange = 100
ii = 1

folder = "parameter/test_min"
ntag = 1

while (ii < maxits and parchange > partol and oldparchange > partol) or (ii < minits):

    ops = {'maxiter':2500,'disp':False}
    gls_optpar = minimize(gls_error_estimate, gls_optpar, args=(age, time, da, dt, folder, ntag, save_rate, prop_data, weights), options=ops).x
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

    weights_y = np.stack(solutions, axis = 0)
    weights = trapezoidal_rule(weights_y[:], da)
    

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
# gls_optpar = [1.16953601, 0.382891]
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

gls_sol_y = np.stack(solutions, axis = 0)
gls_sol =  trapezoidal_rule(gls_sol_y[:], da)
print(f"shape of gls_sol = {gls_sol.shape}")


gls_resids = (prop_data - gls_sol) / (gls_sol**gammas)

stop = timeit.default_timer()

print(f"Simulation took {round((stop - start)/ 60, 2)} minutes to run.")


# Plot the Residuals versus model value and time
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(age, gls_resids, 'r*')
plt.title('GLS')
plt.xlabel('Time (Days)')


plt.subplot(1, 2, 2)
plt.plot(gls_sol, gls_resids, 'r*')
plt.xlabel('Model Value')
plt.show()
plt.savefig(folder + f'_residual_gls_plot_da_{da}_dt_{dt}.png', dpi=300)
plt.close()

plt.figure(figsize=[10,8])
plt.scatter(age, prop_data, label="data", )
plt.plot(age, gls_sol, label="gls")
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Data vs. Statistical Model')
plt.legend()
plt.savefig(folder + f'_data_v_gls_plot_da_{da}_dt_{dt}.png', dpi=300)
plt.show()
plt.close()