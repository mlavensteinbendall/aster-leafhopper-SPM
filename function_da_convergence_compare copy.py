import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile

def convergence_da_plt(Tmax, da, dt, order, folder):

    print('Calculate convergence varying da and fixing dt (without using analytic solution)')

    Ntest    = len(da)

    Norm2    = np.zeros([Ntest-1])
    NormMax  = np.zeros([Ntest-1])

    L2norm   = np.zeros([Ntest-2])
    LMaxnorm = np.zeros([Ntest-2])

    # Time of interest to compare
    # time = np.arange(0, Tmax + dt, dt)            # array from 0 to Tmax
    # Ntime = len(time)                          # number of elements in time
    # n = Ntime // 2                             # use the floor to find the center
    n = int(Tmax / dt) // 2
    # print(f"number of time-steps: {Tmax / dt}")   
    # print(f"time-steps we are testing: {n}")   

    for i in range(0, Ntest-1):

        # Load in relevant data for both mesh sizes
        # data1 = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i}.txt') 
        # data2 = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i+1}.txt')

        data1 = last_time_solution(folder, i)
        data2 = last_time_solution(folder, i+1)

        print(data1.shape)
        print(data2.shape)

        # Solve for L2 and L-max norms
        # Norm2[i]   = np.sqrt(np.mean((data1[n, :]  - data2[n,  ::2])**2))  # L2 norm
        # NormMax[i] = np.max (np.abs(  data1[n, :]  - data2[n,  ::2]))      # L∞ norm
        Norm2[i]   = np.sqrt(np.mean((data1[:]  - data2[::2])**2))  # L2 norm
        NormMax[i] = np.max (np.abs(  data1[:]  - data2[::2]))      # L∞ norm

        # Calculate the order of convergence for norms
        if i > 0:
            print(da[i])
            print(da[i-1])
            L2norm[i-1]   = np.log(Norm2[i-1]   / Norm2[i])   / np.log(da[i-1] / da[i])
            LMaxnorm[i-1] = np.log(NormMax[i-1] / NormMax[i]) / np.log(da[i-1] / da[i])

    # Display the norms and errors
    for i in range(0, Ntest-1):
        print(f'For da =   {round(da[i], 10)} vs da = {round(da[i+1], 10)}')
        print(f'Norm 2 :   {round(Norm2[i], 10)}')
        print(f'Norm inf : {round(NormMax[i], 10)}')

        if i > 0:
            print(f'L2 q error:   {round(L2norm[i-1], 10)}')  # L2 order of convergence
            print(f'LMax q error: {round(LMaxnorm[i-1], 10)}')  # L∞ order of convergence
        print('')

    # Plot the log-log for the errors
    plt.loglog(da[:-1], Norm2,          label='Norm2')
    plt.loglog(da[:-1], NormMax,        label='NormMax')
    plt.loglog(da[:-1], da[:-1]**order, label=f'Order-{order}')

    plt.xlabel(r'$\Delta a$')
    plt.ylabel('Norm')
    plt.title('Numerical Convergence with varied ' + r'$\Delta a$' + ' and fixed ' + r'$\Delta t$')
    plt.legend()

    # Convert ds array values to a string
    ds_values_str = '_'.join(map(str, np.round(da, 5) ))

    # Save the plot to a file -- labels with da values and dt  
    plt.savefig(folder + '/plots/dt_' + str(dt) + '/da_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt) + '.png', dpi=300)
    plt.show()
    plt.close()

    return Norm2, L2norm, NormMax, LMaxnorm


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
# Tmax = 5
# order = 2   # Example order of accuracy
# m = 0.5

# testing_folder = 'mortality'

# convergence_folder = 'fixed_dt'

# constant = False    # True for constant mu, False for function mu
# analytical_sol = True
# hill_func = True
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

# convergence_da_plt(Tmax, da, dt, order, folder)
