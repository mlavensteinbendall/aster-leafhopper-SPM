# Author: Morgan Lavenstein Bendall
# Objective: This calls the function of our model and runs at different da and dt.

import numpy                        as np
import matplotlib.pyplot            as plt 
import timeit
from old.function_upwind_age        import UPW_SPM
# from convergence_da         import convergence_da_plt
# from function_conservation  import conservation_plt
from function_conservation_no_exact_sol import conservation_plt
# from convergence_dt         import convergence_dt_plt
from function_dt_convergence_compare    import convergence_dt_plt
from function_da_convergence_compare    import convergence_da_plt
# from convergence_da         import convergence_da_plt
from print_tab_conv                 import tabulate_conv
# from function_numerical_method                    import solveSPM
from function_figures                           import plt_mortality_func, plt_reproduction_func, plt_total_pop, plt_boundary_condition, plt_numerical_sol

from function_mortality import mortality


start = timeit.default_timer()

## INITIAL CONDITIONS
Amax = 30       # max age
# Tmax = 5     # max time
Tmax = 2
order = 2       # order of method
Ntest = 5       # number of cases


# Mortality set up
m = 0.5 # 1/30            # constant for mux
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

# da[0] = 0.01
# da[1] = 0.005
# da[2] = 0.0025
# da[3] = 0.00125
# da[4] = 0.000625

da = np.zeros([1])
da[0] = 2
dt = 1


dt = np.zeros([Ntest]) # order smallest to largest

# dt[0] = 0.5 * 0.1
# dt[1] = 0.5 * 0.05
# dt[2] = 0.5 * 0.025
# dt[3] = 0.5 * 0.0125
# dt[4] = 0.5 * 0.00625

# dt = da
# dt = 0.5 * da
# dt = 0.01
# dt = 0.001
# dt = 0.001


testing_folder = 'mortality'
# testing_folder = 'reproduction'

if constant_mortality == True:
    if m == 0 :
        function_folder = "no_mortality"

    else:
        function_folder = "constant_mortality"

elif hill_func_mortality == True:
    function_folder = "hill_mortality"

elif linear_slope_func_mortality == True:
    function_folder = "linear_mortality"

else:
    function_folder = "gompertz_mortality"

type_k = "none"

if testing_folder == 'reproduction':

    if constant_reproduction == True:
        if k == 0:
            function_folder = function_folder + '/' + 'no_reproduction'
            type_k = 'no_reproduction'

        else:
            function_folder = function_folder + '/' + 'constant_reproduction/rep_' + str(k)   # using a step function
            type_k = 'constant_reproduction'

    elif linear_reproduction == True:
        function_folder = function_folder + '/' + 'linear_reproduction'                         # using a step function
        type_k = 'linear_reproduction'

    elif gaussian_reproduction == True:
        function_folder = function_folder + '/' + 'gaussian_reproduction'
        type_k = 'gaussian_reproduction' 

    elif test_reproduction == True:
        function_folder = function_folder + '/' + 'test'

print(function_folder)


if isinstance(dt, np.ndarray):
    convergence_folder = 'varied_dt'

else:
    convergence_folder = 'fixed_dt'


folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder


# Using the given da and dt values, this loop calculates the numerical solution, solve the analytical 
# solution, and plots the numerical vs. analytical solution. 
# BEGIN LOOP ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(0):

    print('Entering loop ' + str(i))                # progress update, loop began

    # initalize arrays
    age = np.arange(0, Amax + da[i], da[i])       # array from 0 to Amax
    # print(age)
    Nage = len(age)                               # number of elements in age
    print("age:", age[-1])                        # check that last element is Amax

    mu = np.zeros(Nage)
    mu = mortality(Amax, age, m, b, constant_mortality, linear_slope_func_mortality, hill_func_mortality)


    ## NUMERICAL SOLUTION 
    # IF da and dt are varied, do this -----------------------------------------------------------
    if isinstance(dt, np.ndarray):

        # initalize arrays
        time = np.arange(0,Tmax + dt[i], dt[i])     # array from 0 to Tmax
        Ntime = len(time)                           # number of elements in time
        print("Time:", time[-1])                    # check that last element is Tmax

        # initalize data matrix
        data = np.zeros([Ntime,Nage])

        # print CFL 
        print('CFL: ' + str(round(dt[i]/da[i], 5)))   

        # calculate solution
        data = solveSPM(age, time, da[i], dt[i], mu, k, type_k)              # lax-wendroff method

            
    # ELSE da is varied and dt is constant, do this ------------------------------------------------
    else:
        # initalize arrays
        time = np.arange(0, Tmax + dt, dt)           # array from 0 to Tmax
        Ntime = len(time)                           # number of elements in time
        print("Time:", time[-1])                    # check that the last element is Tmax

        # initialize matrix
        data = np.zeros([Ntime, Nage])

        # print CFL
        print('CFL: ' + str(round(dt/da[i], 5)))

        # calculate solution
        data = solveSPM(age, time, da[i], dt, mu, k, type_k)                 # lax-wendroff method


    # Save data to a file --------------------------------------------------------------------------
    if isinstance(dt, np.ndarray):   
        np.savetxt(folder + '/solutions/num_'+ str(i) +'.txt', data)     # save data to file 

    else:
        np.savetxt(folder + '/solutions/dt_' + str(dt) + '/num_'+ str(i) +'.txt', data)     # save data to file 

    if i == 0:

        # Plot the mortality function
        plt_mortality_func(age, mu, dt, folder)

        # Plot the reproduction function
        plt_reproduction_func(age, k, type_k, dt, folder)

        # Plot the total population
        plt_total_pop(data, time, da[i], dt, i, folder)

        # Plot the boundary condition
        plt_boundary_condition(data, time, da[i], dt, i, folder)



    ## ANALYTICAL SOLUTION     
    # calculate the analytical solution for every age at time t
    for i_t in range(0, len(time)):

        # initialize analytical solution matrix
        sol = np.zeros([len(time),len(age)])

        if analytical_sol == True:

            if constant_mortality == True:
                sol[i_t,:] = np.exp(-(age - ( time[i_t] + 5))**2) * np.exp( - mu * time[i_t])     # with advection -- CONSTANT

            elif hill_func_mortality == True:
                sol[i_t,:] = np.exp(-(age - ( time[i_t] + 5))**2) * np.exp(- (30 * np.log(age**2 + 30**2) - 30 * np.log((age - time[i_t])**2 +30**2))) # with advection -- hill function

            elif linear_slope_func_mortality == True:
                sol[i_t,:] = np.exp(-(age - ( time[i_t] + 5))**2) * np.exp(- m * (age )* time[i_t] + 0.5 * m * (time[i_t])**2)     # with advection -- NON CONSTANT linear slope

    plt_numerical_sol(analytical_sol, sol, data, age, time, da, dt, Ntime, i, folder)

    print('Loop ' + str(i) + ' Complete.')                      # progress update, loop end


# END LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



## CONVERGENCE ------------------------------------------------------------------------------------------
# Calculate and plot the convergence, returns an matrix with Norm2, L2norm, NormMax, and LMaxnorm
# if isinstance(dt, np.ndarray):
#     # Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Amax, Tmax, da, dt, order, m, b, constant_mortality, folder) 
#     Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Tmax, da, dt, order, folder)
# else:
#     # Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(Amax, Tmax, da, dt, order, m, b, constant_mortality, False, folder)
#     Norm2, L2norm, NormMax, LMaxnorm = convergence_da_plt(Tmax, da, dt, order, folder)


# ## TOTAL POPULATION ERROR --------------------------------------------------------------------------------
# # Checks conservation, returns norm and order of conservation
# # Norm1, L1norm = conservation_plt(Ntest, da, m, Amax, Tmax, dt, order, folder, constant, hill_func)   # only works for constant 
# Norm1, L1norm = conservation_plt(da, dt, order, folder)


# ## PRINT NORMS --------------------------------------------------------------------------------------------
# # print latex table
# tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)

# # print excel compatible table
# if isinstance(dt, np.ndarray):
#     for i in range(len(da)):
#         print(f"{dt[i]}, {da[i]}, {Norm2[i]}, {L2norm[i]}, {NormMax[i]}, {LMaxnorm[i]}, {Norm1[i]}, {L1norm[i]}")
# else:
#     for i in range(len(da)):
#         print(f"{dt}, {da[i]}, {Norm2[i]}, {L2norm[i]}, {NormMax[i]}, {LMaxnorm[i]}, {Norm1[i]}, {L1norm[i]}")


stop = timeit.default_timer()

print('Time: ', stop - start)