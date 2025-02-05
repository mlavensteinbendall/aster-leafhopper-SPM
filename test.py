import numpy                        as np
import matplotlib.pyplot            as plt 
from function_model import solveSPM
from function_figures                   import plt_mortality_func, plt_reproduction_rate_func, plt_total_pop, plt_boundary_condition, plt_numerical_solution
import zipfile
import re

# from function_convergence_w_exact_sol import convergence_dt_plt, conservation_plt
from function_convergence             import convergence_dt_plt, conservation_plt, last_time_solution
from function_convergence_table                 import tabulate_conv

## INITIAL CONDITIONS
Amax = 30      # max age
Tmax = 2      # max time
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

folder_0 = 'convergence/both/logistic/varied_dt/constant/constant_zero_'
# folder_logistic = 'convergence/both/logistic/varied_dt/logistic_reproduction/logistic_reproduction_'
folder_logistic = 'convergence/both/logistic/varied_dt/logistic_repo_20_tmax2/logistic_reproduction_'

# folder_0 = 'convergence/both/logistic/varied_dt/constant_tmax20/constant_zero_'
# folder_logistic = 'convergence/both/logistic/varied_dt/logistic_repo_20_tmax20/constant_zero_'

for i in range(0, 5):
    data1 = last_time_solution(folder_0, i, da, dt)
    data2 = last_time_solution(folder_logistic, i, da, dt)

    age_num_points = int(30 / da[i]) + 1
    age = np.linspace(0, 30, age_num_points)

    plt.plot(age, data1, label="repo 0")
    plt.plot(age, data2, label="repo logistic")
    plt.xlabel('Age')
    plt.ylabel('Population')
    plt.title('Last Time Step Solution ' + r'$\Delta a$ = ' + str(da[i]) + ' and ' + r'$\Delta t$ = '+ str(dt[i]))
    plt.legend()
    plt.show()
    plt.close()


    plt.plot(age, data2 - data1)
    plt.xlabel('Age')
    plt.ylabel('Difference')
    plt.title('Difference between repro 0 & logistic ' + r'$\Delta a$ = ' + str(da[i]) + ' and ' + r'$\Delta t$ = '+ str(dt[i]))
    # plt.legend()
    plt.show()
    plt.close()