""" This main is testing writing the numerical files to txt and npy. I found that saving and reading through zip was a lot faster."""


import numpy                        as np
import matplotlib.pyplot            as plt 
from function_numerical_method_to_files_npy import solveSPM_file_npy
# from function_numerical_method_to_files import solveSPM_file
# from function_numerical_method          import solveSPM
from function_mortality                 import mortality
from function_figures                   import plt_mortality_func, plt_reproduction_func, plt_total_pop, plt_boundary_condition, plt_numerical_sol
import zipfile
import re


## INITIAL CONDITIONS
Amax = 30       # max age
Tmax = 20
# order = 2       # order of method
# Ntest = 5       # number of cases


# Mortality set up
m = 0 # 1/30            # constant for mux
b = 0              # y-intercept
constant_mortality = True   # True for constant mu, False for function mu
analytical_sol = True
hill_func_mortality = False
linear_slope_func_mortality = False

# Reproduction set up
k = 1
constant_reproduction = False
linear_reproduction = False
gaussian_reproduction = False
test_reproduction = False


# need to chose da and dt so that the last value in the array are Amax and Tmax
# da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
# da[0] = 0.1
# da[1] = 0.05
# da[2] = 0.025
# da[3] = 0.0125
# da[4] = 0.00625


# da = np.zeros([1])
da = 0.00625
dt = 0.5 * 0.00625

age_num_points = int(Amax / da) + 1
age = np.linspace(0, Amax, age_num_points)


time_num_points = int(Tmax / dt) + 1
time = np.linspace(0, Tmax, time_num_points)

Ntime = len(time)

# print("Amax:", Amax, "Type:", type(Amax))
# print("da:", da, "Type:", type(da))

print('age: ' + str(age[-1]))
print('time: ' + str(time[-1]))


mu = np.zeros(len(age))
# mu = mortality(Amax, age, m, b, constant_mortality, linear_slope_func_mortality, hill_func_mortality)
mu = mortality(Amax, age, m, b, True, False, False)

# print(mu)
# 
# solveSPM_file(age, time, da, dt, Tmax, Amax, mu, k, 'constant_reproduction', 'test/', 0)
# solveSPM_file_npy(age, time, da, dt, Tmax, Amax, mu, k, 'constant_reproduction', 'test/', 0, 20)

# data2 = solveSPM(age, time, da, dt, mu, k, 'constant_reproduction')


data = np.loadtxt('test/test_0.txt')
# data = np.load('_results_0.zip')

# Specify the ZIP file and extraction directory
zip_filename = "test/_results_0.zip"
output_dir = "unzipped_output"  # Directory to extract files

# Open the ZIP file
with zipfile.ZipFile(zip_filename, 'r') as zf:
    # List all files in the ZIP
    file_list = zf.namelist()
    
    # Sort the files (important if time steps should be in order)
    file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Read each file into memory
    solutions = []
    for file in file_list:
        with zf.open(file) as f:
            solutions.append(np.load(f))  # Load the .npy file into a numpy array

# Now `solutions` is a list of numpy arrays, one for each time step
print(f"Loaded {len(solutions)} time steps.")

# print(solutions[0])

# COMPARTISION PLOT BTWN NUMERICAL AND ANALYTICAL
# get inidices of initial, middle, and last time step
plot_indices = [0, Ntime // 2, Ntime - 1]

print(len(data))
# plot numerical and analytical solution
for t_index in plot_indices:

    if solutions[int(t_index/20)].all() == data[t_index, :].all():
        print("same")
    else:
        print("nope")
    plt.plot(age, solutions[int(t_index/20)], label=f'npy Numerical at time  {round(time[t_index], 1)  }', linestyle='-')    # numerical 
    plt.plot(age, data[t_index, :], label=f'txt Numerical at time  {round(time[t_index], 1)  }', linestyle='-')    # numerical 
    # plt.plot(age, data2[t_index, :], label=f'Numerical at time  {round(time[t_index], 1)  }', linestyle='-')    # numerical 

plt.axhline(y=1, color='r', linestyle='--', label='y=1')
plt.xlabel('Age')
plt.ylabel('Population')
plt.title('Age Distribution of Population (' + r'$\Delta a$' + ' = ' + str(da) + ', ' + r'$\Delta t$' + ' = ' + str(dt) + ')')
plt.legend()
plt.show()        # show plot
plt.close()
