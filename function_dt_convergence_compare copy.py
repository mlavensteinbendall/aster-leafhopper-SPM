import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile

def convergence_dt_plt(Tmax, ds, dt, order, folder):

    print('Calculate convergence varying da and dt (without using analytic solution)')

    Ntest = len(dt)

    Norm2 = np.zeros([Ntest-1])
    NormMax = np.zeros([Ntest-1])

    L2norm = np.zeros([Ntest-2])
    LMaxnorm = np.zeros([Ntest-2])

    for i in range(0, Ntest-1):

        # # Load in relevant data for both mesh sizes
        # data1 = np.loadtxt(f'{folder}/solutions/num_{i}.txt') 
        # data2 = np.loadtxt(f'{folder}/solutions/num_{i+1}.txt')
        data1 = last_time_solution(folder, i)
        data2 = last_time_solution(folder, i+1)
 

        # Time of interest to compare
        n = int(Tmax / dt[i]) // 2   

        # # Solve for L2 and L-max norms
        # Norm2[i]   = np.sqrt(np.mean((data1[n, :]  - data2[n*2,  ::2])**2))  # L2 norm
        # NormMax[i] = np.max( np.abs(  data1[n, :]  - data2[n*2,  ::2]))      # L∞ norm
        Norm2[i]   = np.sqrt(np.mean((data1[:]  - data2[::2])**2))  # L2 norm
        NormMax[i] = np.max (np.abs(  data1[:]  - data2[::2]))      # L∞ norm


        # Calculate the order of convergence for norms
        if i > 0:
            print(ds[i])
            print(ds[i-1])
            L2norm[i-1] = np.log(Norm2[i-1] / Norm2[i]) / np.log(ds[i-1] / ds[i])
            LMaxnorm[i-1] = np.log(NormMax[i-1] / NormMax[i]) / np.log(ds[i-1] / ds[i])

    # Display the norms and errors
    for i in range(0, Ntest-1):
        print(f'For dt = {round(dt[i], 10)} vs dt = {round(dt[i+1], 10)}')
        print(f'Norm 2 : {round(Norm2[i], 10)}')
        print(f'Norm inf : {round(NormMax[i], 10)}')

        if i > 0:
            print(f'L2 q error: {round(L2norm[i-1], 10)}')  # L2 order of convergence
            print(f'LMax q error: {round(LMaxnorm[i-1], 10)}')  # L∞ order of convergence
        print('')

    # Plot the log-log for the errors
    plt.loglog(dt[:-1], Norm2, label='Norm2')
    plt.loglog(dt[:-1], NormMax, label='NormMax')
    plt.loglog(dt[:-1], dt[:-1]**order, label=f'Order-{order}')

    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Norm')
    plt.title('Numerical Convergence with varied ' + r'$\Delta a$' + ' and ' + r'$\Delta t$')
    plt.legend()

    # Convert ds array values to a string
    ds_values_str = '_'.join(map(str, np.round(ds, 3) ))
    dt_values_str = '_'.join(map(str, np.round(dt, 3)))


    # Save the plot to a file -- labels with da values and dt 
    plt.savefig(folder + '/plots/dt_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt_values_str) + '.png', dpi=300)
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
        print(last_file)
        with zf.open(last_file) as f:
            last_solution = np.load(f)  # Load the .npy file into a numpy array

    print(f"Loaded the last file: {last_file}")
    return np.array(last_solution)


# da = np.zeros([4]) # order smallest to largest

# # # vary da and dt cases:
# da[0] = 0.1
# da[1] = 0.05
# da[2] = 0.025
# da[3] = 0.0125

# da = 0.1 * da

# dt = 0.5 * da

# testing_folder = 'mortality'
# # testing_folder = 'reproduction'

# function_folder = 'no_mortality'

# if isinstance(dt, np.ndarray):
#     convergence_folder = 'varied_dt'

# else:
#     convergence_folder = 'fixed_dt'


# folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder + '/'

# convergence_dt_plt(10, da, dt, 2, folder)


# # # Test if convergence is working
# # # Mesh options and dt
# # da = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
# # dt = 0.5 * da  # Time steps based on mesh size

# # # Run the function with these parameters
# # Tmax = 5 # Example Smax
# # Smax = 30.0  # Example Smax
# # order = 2   # Example order of accuracy
# # m = 0

# # testing_folder = 'mortality'

# # if isinstance(dt, np.ndarray):
# #     convergence_folder = 'varied_dt'

# # else:
# #     convergence_folder = 'fixed_dt'

# # constant = True    # True for constant mu, False for function mu
# # analytical_sol = True
# # hill_func = False
# # linear_slope_func = False

# # if constant == True:
# #     if m == 0 :
# #         function_folder = "no_mortality"

# #     else:
# #         function_folder = "constant_mortality"

# # elif hill_func == True:
# #     function_folder = "hill_mortality"

# # elif linear_slope_func == True:
# #     function_folder = "linear_mortality"

# # else:
# #     function_folder = "gompertz_mortality"

# # folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder

# # convergence_dt_plt(Tmax, da, dt, order, folder)
