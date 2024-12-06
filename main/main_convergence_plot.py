# import numpy as np
# # from convergence_da import convergence_da_plt
# from convergence_dt import convergence_dt_plt
# from function_conservation import conservation_plt
# from print_tab_conv import tabulate_conv
# import matplotlib.pyplot            as plt 

# ## INITIAL CONDITIONS
# Amax = 30       # max age
# # Tmax = 5     # max time
# Tmax = 20
# order = 2       # order of method
# Ntest = 5       # number of cases


# # Mortality set up
# m = 0 #1/30            # constant for mux
# b = 0              # y-intercept

# # Reproduction set up
# k = 1

# # need to chose da and dt so that the last value in the array are Amax and Tmax
# da = np.zeros([Ntest]) # order smallest to largest

# # # vary da and dt cases:
# da[0] = 0.1
# da[1] = 0.05
# da[2] = 0.025
# da[3] = 0.0125
# da[4] = 0.00625

# dt = np.zeros([Ntest]) # order smallest to largest

# # dt[0] = 0.5 * 0.1
# # dt[1] = 0.5 * 0.05
# # dt[2] = 0.5 * 0.025
# # dt[3] = 0.5 * 0.0125
# # dt[4] = 0.5 * 0.00625

# dt = 0.5 * da
# # dt = 0.01
# # dt = 0.001
# # dt = 0.0001

# time = np.arange(0,Tmax + dt[0], dt[0])
# age = np.arange(0, Amax + da[0], da[0])

# testing_folder = "mortality"
# function_folder = "constant_mortality"

# if isinstance(dt, np.ndarray):
#     convergence_folder = 'varied_dt'

# else:
#     convergence_folder = 'fixed_dt'


# # folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder

# # Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Amax, Tmax, da, dt, 2, m, b, True, folder)

# # Norm1, L1norm = conservation_plt(Ntest, da, m, Amax, Tmax, dt, 2, folder, True, False)

# # tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)

# print('Plot boundary condition')

# folder = '/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/reproduction/no_mortality/constant_reproduction/rep_1-original/varied_dt/solutions/'

# data = np.loadtxt(f'{folder}/num_0.txt') 

# print(data[55,0])
# print(data[55,1])
# print(data[55,2])

# print(data[100,0])
# print(data[100,1])
# print(data[100,2])
# print(data[101,0])
# print(data[101,1])
# print(data[101,2])

# # 55 

# # plt.plot(age, data[55,:])
# plt.plot(time, data[:,0], label = 'age = 0')
# plt.plot(time, data[:,1], label = 'age = da')
# plt.plot(time, data[:,2], label = 'age = 2 * da')
# plt.plot(time, data[:,3], label = 'age = 3 * da')
# plt.plot(time, data[:,4], label = 'age = 4 * da')
# plt.plot(time, data[:,5], label = 'age = 5 * da')
# plt.plot(time, data[:,40], label = 'age = 40 * da')
# plt.plot(time, data[:,60], label = 'age = 60 * da')
# plt.plot(time, data[:,100], label = 'age = 100 * da')
# plt.plot(time, data[:,250], label = 'age = 250 * da')
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.title('Population for different ages over time')
# plt.legend()
# plt.show()

    # if isinstance(dt, np.ndarray):
    #     plt.savefig(folder + '/plots/boundary_condition_for_da_' + str(da) + '_dt_' + str(dt[index]) + '.png', dpi=300)

    # else:
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/boundary_condition_for_da_' + str(da) + '_dt_' + str(dt) + '.png', dpi=300)

    # plt.close()


import numpy                        as np
import matplotlib.pyplot            as plt 
import timeit
from old.function_upwind_age        import UPW_SPM
# from convergence_da         import convergence_da_plt
# from function_conservation  import conservation_plt
from old.function_conservation_no_exact_sol import conservation_plt
# from convergence_dt         import convergence_dt_plt
from old.function_dt_convergence_compare    import convergence_dt_plt
from old.function_da_convergence_compare    import convergence_da_plt
# from convergence_da         import convergence_da_plt
from function_convergence_table                 import tabulate_conv
from function_numerical_method                    import LW_SPM
from function_figures                           import plt_mortality_func, plt_reproduction_func, plt_total_pop, plt_boundary_condition, plt_numerical_sol

from function_mortality import mortality


start = timeit.default_timer()

## INITIAL CONDITIONS
Amax = 30       # max age
# Tmax = 5     # max time
Tmax = 15
order = 2       # order of method
Ntest = 5       # number of cases


# Mortality set up
m = 0 #1/30            # constant for mux
b = 0              # y-intercept
constant_mortality = True   # True for constant mu, False for function mu
analytical_sol = False
hill_func_mortality = False
linear_slope_func_mortality = False

# Reproduction set up
k = 1
constant_reproduction = True
linear_reproduction = False
gaussian_reproduction = False
test_reproduction = False


# need to chose da and dt so that the last value in the array are Amax and Tmax
da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
# da[0] = 0.1
# da[1] = 0.05
# da[2] = 0.025
# da[3] = 0.0125
# da[4] = 0.00625

da[0] = 0.01
da[1] = 0.005
da[2] = 0.0025
da[3] = 0.00125
da[4] = 0.000625


dt = np.zeros([Ntest]) # order smallest to largest

# dt[0] = 0.5 * 0.1
# dt[1] = 0.5 * 0.05
# dt[2] = 0.5 * 0.025
# dt[3] = 0.5 * 0.0125
# dt[4] = 0.5 * 0.00625

# dt = da
dt = 0.5 * da
# dt = 0.01
# dt = 0.001
# dt = 0.001


# testing_folder = 'mortality'
testing_folder = 'reproduction'

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


Norm2, L2norm, NormMax, LMaxnorm = convergence_dt_plt(Tmax, da, dt, order, folder)

Norm1, L1norm = conservation_plt(da, dt, order, folder)

tabulate_conv(dt, da, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder)