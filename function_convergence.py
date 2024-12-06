import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile
from function_trapezoidal_rule import trapezoidal_rule

def convergence_da_plt(Tmax, da, dt, order, folder):

    print('Calculate convergence varying da and fixing dt (without using analytic solution)')

    Ntest    = len(da)

    Norm2    = np.zeros([Ntest-1])
    NormMax  = np.zeros([Ntest-1])

    L2norm   = np.zeros([Ntest-2])
    LMaxnorm = np.zeros([Ntest-2])

    for i in range(0, Ntest-1):

        # Load in last time step from data with differnt mesh sizes
        data1 = last_time_solution(folder, i)       # delta t
        data2 = last_time_solution(folder, i+1)     # delta t / 2

        print(data1.shape)
        print(data2.shape)

        # Solve for L2 and L-max norms
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
    # ds_values_str = '_'.join(map(str, np.round(da, 5) ))

    # Save the plot to a file -- labels with da values and dt  
    # plt.savefig(folder + '/plots/dt_' + str(dt) + '/da_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt) + '.png', dpi=300)
    plt.show()
    plt.close()

    return Norm2, L2norm, NormMax, LMaxnorm


def convergence_dt_plt(Tmax, ds, dt, order, folder):

    print('Calculate convergence varying da and dt (without using analytic solution)')

    Ntest = len(dt)

    Norm2 = np.zeros([Ntest-1])
    NormMax = np.zeros([Ntest-1])

    L2norm = np.zeros([Ntest-2])
    LMaxnorm = np.zeros([Ntest-2])

    for i in range(0, Ntest-1):

        # # Load in relevant data for both mesh sizes
        data1 = last_time_solution(folder, i)
        data2 = last_time_solution(folder, i+1)
 
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
    # ds_values_str = '_'.join(map(str, np.round(ds, 3) ))
    # dt_values_str = '_'.join(map(str, np.round(dt, 3)))


    # Save the plot to a file -- labels with da values and dt 
    # plt.savefig(folder + '/plots/dt_convergence_for_da_' + str(ds_values_str) + '_dt_' + str(dt_values_str) + '.png', dpi=300)
    plt.show()
    plt.close()

    return Norm2, L2norm, NormMax, LMaxnorm

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