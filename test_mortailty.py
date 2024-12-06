
import numpy                        as np
import matplotlib.pyplot            as plt 
from function_model import solveSPM
from function_figures                   import plt_mortality_func, plt_reproduction_func, plt_total_pop, plt_boundary_condition, plt_numerical_sol
import zipfile
import re

import timeit
# from convergence_da         import convergence_da_plt
# from function_conservation  import conservation_plt
# from function_conservation_no_exact_sol import conservation_plt
# from convergence_dt         import convergence_dt_plt
# from convergence_da         import convergence_da_plt
# from function_dt_convergence_compare    import convergence_dt_plt
# from function_da_convergence_compare    import convergence_da_plt
from function_convergence_table                 import tabulate_conv

from function_convergence_compare import convergence_da_plt, convergence_dt_plt, conservation_plt


start = timeit.default_timer()

## INITIAL CONDITIONS
Amax = 30       # max age
# Tmax = 5     # max time
Tmax = 20
order = 2       # order of method
Ntest = 5       # number of cases

k = 0

par = 0.01

# need to chose da and dt so that the last value in the array are Amax and Tmax
da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
da[0] = 0.1
da[1] = 0.05
da[2] = 0.025
da[3] = 0.0125
da[4] = 0.00625

# da = 0.1 * da

dt = 0.5 * da
# dt = 0.01


testing_folder = 'mortality'
# testing_folder = 'reproduction'

function_folder = 'no_mortality'

if isinstance(dt, np.ndarray):
    convergence_folder = 'varied_dt'

else:
    convergence_folder = 'fixed_dt'


folder = 'convergence/' #+ testing_folder + '/' + function_folder + '/' + convergence_folder + '/'


# Using the given da and dt values, this loop calculates the numerical solution, solve the analytical 
# solution, and plots the numerical vs. analytical solution. 
# BEGIN LOOP ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(Ntest):

    print('Entering loop ' + str(i))                # progress update, loop began

    # initalize arrays
    age_num_points = int(Amax / da[i]) + 1
    age = np.linspace(0, Amax, age_num_points)
    Nage = len(age)                               # number of elements in age
    print("Age:", age[-1])                        # check that last element is Amax


    ## NUMERICAL SOLUTION 
    # IF da and dt are varied, do this -----------------------------------------------------------
    if isinstance(dt, np.ndarray):

        # initalize arrays
        time_num_points = int(Tmax / dt[i]) + 1
        time = np.linspace(0, Tmax, time_num_points)
        print("Time:", time[-1])                    # check that last element is Tmax

        # print CFL 
        print('CFL: ' + str(round(dt[i]/da[i], 5)))   

        # calculate solution
        solveSPM(par, age, time, da[i], dt[i], k, 'gaussian_reproduction', folder, i, 10)

            
    # ELSE da is varied and dt is constant, do this ------------------------------------------------
    else:
        # initalize arrays
        time_num_points = int(Tmax / dt) + 1
        time = np.linspace(0, Tmax, time_num_points)
        print("Time:", time[-1])                    # check that last element is Tmax

        # print CFL
        print('CFL: ' + str(round(dt/da[i], 5)))

        # calculate solution
        solveSPM(par, age, time, da[i], dt, k, 'gaussian_reproduction', folder, i, 20)


    # Specify the ZIP file and extraction directory
    zip_filename = folder + f"_results_{i}.zip"
    output_dir = "unzipped_output"  # Directory to extract files

    # Open the ZIP file
    with zipfile.ZipFile(zip_filename, 'r') as zf:
        # List all files in the ZIP
        file_list = zf.namelist()
        
        # Sort the files (important if time steps should be in order)
        file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

        print(file_list)
        
        # Read each file into memory
        solutions = []
        for file in file_list:
            with zf.open(file) as f:
                solutions.append(np.load(f))  # Load the .npy file into a numpy array

    data = np.stack(solutions, axis = 0)
    # Now `solutions` is a list of numpy arrays, one for each time step
    print(f"Loaded {len(solutions)} time steps.")

    print(data.shape)

    plt.plot(age, data[0, :],  label=f'txt Numerical at time  {round(time[0] , 1)  }', linestyle='-') 
    plt.plot(age, data[-1, :], label=f'txt Numerical at time  {round(time[-1], 1)  }', linestyle='-')    # numerical 

    plt.axhline(y=1, color='r', linestyle='--', label='y=1')
    plt.xlabel('Age')
    plt.ylabel('Population')
    plt.title('Age Distribution of Population (' + r'$\Delta a$' + ' = ' + str(da[i]) + ', ' + r'$\Delta t$' + ' = ' + str(dt) + ')')
    plt.legend()
    plt.show()        # show plot
    plt.close()

    print('Loop ' + str(i) + ' Complete.')                      # progress update, loop end


# END LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



## CONVERGENCE ------------------------------------------------------------------------------------------
# Calculate and plot the convergence, returns an matrix with Norm2, L2norm, NormMax, and LMaxnorm
if isinstance(dt, np.ndarray):
#     # Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Amax, Tmax, da, dt, order, m, b, constant_mortality, folder) 
    Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Tmax, da, dt, order, folder)
else:
#     # Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(Amax, Tmax, da, dt, order, m, b, constant_mortality, False, folder)
    Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(Tmax, da, dt, order, folder)


# ## TOTAL POPULATION ERROR --------------------------------------------------------------------------------
# # Checks conservation, returns norm and order of conservation
# # Norm1, L1norm = conservation_plt(Ntest, da, m, Amax, Tmax, dt, order, folder, constant, hill_func)   # only works for constant 
Norm1, L1norm = conservation_plt(da, dt, order, folder)


# ## PRINT NORMS --------------------------------------------------------------------------------------------
# # print latex table
tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)

# # print excel compatible table
# if isinstance(dt, np.ndarray):
#     for i in range(len(da)):
#         print(f"{dt[i]}, {da[i]}, {Norm2[i]}, {L2norm[i]}, {NormMax[i]}, {LMaxnorm[i]}, {Norm1[i]}, {L1norm[i]}")
# else:
#     for i in range(len(da)):
#         print(f"{dt}, {da[i]}, {Norm2[i]}, {L2norm[i]}, {NormMax[i]}, {LMaxnorm[i]}, {Norm1[i]}, {L1norm[i]}")


stop = timeit.default_timer()

print('Time: ', stop - start)