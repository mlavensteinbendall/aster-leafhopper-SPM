import numpy as np
import matplotlib.pyplot as plt
from function_trapezoidal_rule import trapezoidal_rule
from function_model import mortality, k_dep
import re
import zipfile

def plt_mortality_func(age, par, folder):

    print('Plot mortality function')

    mu = mortality(age, par)

    plt.plot(age, mu)
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    plt.title('Age-Specific Mortality Rate')

    # if isinstance(dt, np.ndarray): 
    #     plt.savefig(folder + '/plots/mortality_plot.png', dpi=300)
    # else:
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/mortality_plot.png', dpi=300)

    plt.show()
    plt.close()

def plt_reproduction_rate_func(age, par, folder):

    print('Plot mortality function')

    reproduction_rate = k_dep(age, par)

    plt.plot(age, reproduction_rate)
    plt.xlabel('Age')
    plt.ylabel('Reproduction Rate')
    plt.title('Age-Specific Reproduction Rate')

    # if isinstance(dt, np.ndarray): 
    #     plt.savefig(folder + '/plots/reproduction_plot.png', dpi=300)
    # else:
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/reproduction_plot.png', dpi=300)

    plt.show()
    plt.close()
    

def plt_total_pop(data, time, da, dt, index, folder, save_rate):

    print('Plot total population')

    times = []

    for t,T in enumerate(time):
        if t == 0 or t == len(time) - 1 or t % save_rate == 0:
            times.append(T)

    times = np.array(times)
    totalPop = np.zeros(len(times))

    for i in range(0,len(times)):
        print(i)
        totalPop[i] = trapezoidal_rule(data[i,:], da) 
        # totalPop[i] = trapezoidal_rule(data[i], da)

    # print(totalPop)
    # print(data[-1,:])

    plt.plot(times, totalPop)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Total Population over Time')

    # if isinstance(dt, np.ndarray):
    #     plt.savefig(folder + '/plots/tot_pop_over_time_for_da_' + str(da) + '_dt_' + str(dt[index]) + '.png', dpi=300)

    # else:
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/tot_pop_over_time_for_da_' + str(da) + '_dt_' + str(dt) + '.png', dpi=300)
    plt.show()
    plt.close()



def plt_boundary_condition(data, time, da, dt, index, folder, save_rate):

    print('Plot boundary condition')

    # data = np.array(data)
    # bc = data[:,0]

    times = []

    for t,T in enumerate(time):
        if t == 0 or t == len(time) - 1 or t % save_rate == 0:
            times.append(T)

    plt.plot(times, data[:,0])
    # plt.plot(bc)
    plt.xlabel('Time')
    plt.ylabel('Number of Newborns')
    plt.title('Boundary Condition')

    # if isinstance(dt, np.ndarray):
    #     plt.savefig(folder + '/plots/boundary_condition_for_da_' + str(da) + '_dt_' + str(dt[index]) + '.png', dpi=300)

    # else:
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/boundary_condition_for_da_' + str(da) + '_dt_' + str(dt) + '.png', dpi=300)
    plt.show()
    plt.close()

def first_time_solution(folder, i, da, dt):
    # Specify the ZIP file and extraction directory
    if isinstance(dt, np.ndarray):
        zip_filename = folder + f"_results_{i}_da_{da[i]}_dt_{dt[i]}.zip"
    else:
        zip_filename = folder + f"_results_{i}_da_{da[i]}_dt_{dt}.zip"
    # output_dir = "unzipped_output"  # Directory to extract files

    with zipfile.ZipFile(zip_filename, 'r') as zf:
        # List all files in the ZIP
        file_list = zf.namelist()
        
        # Sort the files (important if time steps should be in order)
        file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
        
        # Read the last file in the list
        first_file = file_list[0]
        with zf.open(first_file) as f:
            first_solution = np.load(f)  # Load the .npy file into a numpy array

    print(f"Loaded the first file: {first_solution}")
    return np.array(first_solution)

def plt_numerical_sol(analytical_sol, sol, data, age, time, da, dt, Ntime, index, folder):

    # COMPARTISION PLOT BTWN NUMERICAL AND ANALYTICAL
    # get inidices of initial, middle, and last time step
    plot_indices = [0, Ntime // 2, Ntime - 1]

    plt.close()
    # plot numerical and analytical solution
    for t_index in plot_indices:
        if analytical_sol == True: 
            plt.plot(age, sol [t_index, :], label=f'Analytical at time {round(time[t_index], 1)  }', linestyle=':')     # analytical 
        plt.plot(age, data[t_index, :], label=f'Numerical at time  {round(time[t_index], 1)  }', linestyle='-')    # numerical 

    plt.axhline(y=1, color='r', linestyle='--', label='y=1')
    plt.xlabel('Age')
    plt.ylabel('Population')
    if isinstance(dt, np.ndarray):
        # plt.title(f'Population by Step when $\Delta a$ = {da[i] } and $\Delta t$ = {dt[i] }')
        plt.title('Age Distribution of Population (' + r'$\Delta a$' + ' = ' + str(da[index]) + ', ' + r'$\Delta t$' + ' = ' + str(dt[index]) + ')')
    else:
        plt.title('Age Distribution of Population (' + r'$\Delta a$' + ' = ' + str(da[index]) + ', ' + r'$\Delta t$' + ' = ' + str(dt) + ')')
        # plt.title(f'Population by Step when' + r'$\Delta a$' + f' = {da[i] } and ' + r'$\Delta t' + f' = {dt }')
    plt.legend()

    # save plots to folder
    if isinstance(dt, np.ndarray):

        plt.savefig(folder + '/plots/num_' + str(index) + '_da_' + str(da[index]) + '_dt_' + str(round(dt[index],5)) + '.png', dpi=300)
    else:

        plt.savefig(folder + '/plots/dt_' + str(dt) + '/num_' + str(index) + '_da_' + str(da[index]) + '_dt_' + str(dt) + '.png', dpi=300)
    
    # plt.show()        # show plot

    plt.close()



