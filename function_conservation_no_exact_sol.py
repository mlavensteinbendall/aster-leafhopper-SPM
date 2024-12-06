import numpy as np
import matplotlib.pyplot as plt
from function_trapezoidal_rule import trapezoidal_rule
import re
import zipfile


def conservation_plt(da, dt, order, folder):

    print('Calculate conservation (without using an analytic solution)')

    Ntest    = len(da)

    Norm1    = np.zeros([Ntest-1])

    L1norm   = np.zeros([Ntest-2])

    totalPop_1 = np.zeros([5]) 
    totalPop_2 = np.zeros([5]) 

    for i in range(0, Ntest-1):

        # Load in relevant data for both mesh sizes
        # if isinstance(dt, np.ndarray):
        #     data1 = np.loadtxt(f'{folder}/solutions/num_{i}.txt') 
        #     data2 = np.loadtxt(f'{folder}/solutions/num_{i+1}.txt')
        # else:
        #     data1 = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i}.txt') 
        #     data2 = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i+1}.txt')

        data1 = last_time_solution(folder, i)
        data2 = last_time_solution(folder, i+1)
 

        # Time of interest to compare
        # totalPop_1[i] = trapezoidal_rule( data1[-1,:], da[i])           # using data at Tmax
        # totalPop_2[i] = trapezoidal_rule( data2[-1,:], da[i+1])         # using data at Tmax
        totalPop_1[i] = trapezoidal_rule( data1[:], da[i])           # using data at Tmax
        totalPop_2[i] = trapezoidal_rule( data2[:], da[i+1])         # using data at Tmax
        print('Numerical total pop using da = ' + str(da[i]) +'  = '      + str(totalPop_1[i]))
        print('Numerical total pop using da = '  + str(da[i+1]) +'  = '    + str(totalPop_2[i]))

        # Solve for L2 and L-max norms
        Norm1[i] = np.abs( totalPop_1[i] - totalPop_2[i])  # L2 norm


        # Calculate the order of convergence for norms
        if i > 0:
            L1norm[i-1]   = np.log(Norm1[i-1]   / Norm1[i])   / np.log(da[i-1] / da[i])

    # Display the norms and errors
    for i in range(0, Ntest-1):
        print(f'For da =   {round(da[i], 10)} vs da = {round(da[i+1], 10)}')
        print(f'Norm 1 :   {Norm1[i]}')


        if i > 0:
            print(f'L1 q error:   {L1norm[i-1]}')  # L2 order of convergence
        print('')

    # Plot the log-log for the errors
    plt.loglog(da[:-1], Norm1,          label='Norm1')
    plt.loglog(da[:-1], da[:-1]**order, label=f'Order-{order}')

    plt.xlabel(r'$\Delta a$')
    plt.ylabel('Norm')
    plt.title('Total Population Convergence')
    plt.legend()

    # Convert ds array values to a string
    ds_values_str = '_'.join(map(str, np.round(da, 3) ))

    # # if isinstance(dt, np.ndarray):
    # if isinstance(dt, np.ndarray):
    #     # plt.savefig('da_plot/'+ folder +'/varied_dt/lw-ex_plot_totPop_mu__ds_' + ds_values_str + '.png', dpi=300)
    #     dt_values_str = '_'.join(map(str, np.round(dt, 3)))
    #     plt.savefig(folder + '/plots/tot_pop_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt_values_str) + '.png', dpi=300)

    # else:
    #     # plt.savefig('da_plot/'+ folder +'/fixed_dt/lw-ex_plot_totPop_mu__ds_' + ds_values_str + '.png', dpi=300)  
    #     plt.savefig(folder + '/plots/dt_' + str(dt) + '/tot_pop_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt) + '.png', dpi=300)
 
    plt.show()
    plt.close()

    return Norm1, L1norm

def last_time_solution(folder, i):
    # Specify the ZIP file and extraction directory
    zip_filename = folder + f"_results_{i}.zip"
    output_dir = "unzipped_output"  # Directory to extract files

    with zipfile.ZipFile(zip_filename, 'r') as zf:
        # List all files in the ZIP
        file_list = zf.namelist()
        
        # Sort the files (important if time steps should be in order)
        file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
        
        # Read the last file in the list
        last_file = file_list[-1]
        with zf.open(last_file) as f:
            last_solution = np.load(f)  # Load the .npy file into a numpy array

    print(f"Loaded the last file: {last_file}")
    return np.array(last_solution)

# da = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
# dt = 0.0001

# # # Run the function with these parameters
# Smax = 30.0  # Example Smax
# order = 2   # Example order of accuracy
# m=0.5


# testing_folder = 'mortality'

# convergence_folder = 'fixed_dt'

# constant = True    # True for constant mu, False for function mu
# analytical_sol = True
# hill_func = False
# linear_slope_func = False

# if constant == True:
#     if m == 0 :
#         function_folder = "no_mortality"

#     else:
#         function_folder = "constant_mortality"

# elif hill_func == True:
#     function_folder = "hill_mortality"

# elif linear_slope_func == True:
#     function_folder = "linear_mortality"

# else:
#     function_folder = "gompertz_mortality"

# folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder


# conservation_plt(da, dt, order, folder)
