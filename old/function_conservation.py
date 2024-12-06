import numpy as np # Numpy for numpy
import matplotlib.pyplot as plt
from function_trapezoidal_rule import trapezoidal_rule


def conservation_plt(ds, c, Smax, Tmax, dt, order, folder, constant, hill_func):
    """Performs trapezoidal rule
    
    Args:
        Ntest  (array):    A list of the population at different steps.
        time    ():
        ds      (int):
        c       (int):
        Smax    ():
        Tmax    ():
        dt      ():
        order   ():
        
    Returns:
        Norm1  (array):     A list of 1-norm errors
        L1norm  (array):    A list of 1-norm orders.
    """

    Norm1 = np.zeros([5])
    L1norm = np.zeros([5])


    # Calculate the total population using trapezoidal rule
    for i in range(5):
        totalPop_num = 0
        totalPop_sol = 0

        age = np.arange(0, Smax + ds[i], ds[i])      # array from 0 to age_max


        if constant == True:
            sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp( - c * Tmax)     # with advection -- CONSTANT
        else:
            if hill_func == True:
                sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp(- (30 * np.log(age**2 + 30**2) - 30 * np.log((age - Tmax)**2 +30**2))) # with advection -- hill function
                # sol = np.exp(-(age - (Tend + 5))**2) / ((age**2 + 30**2) / ((age - Tend)**2 + 30**2))**30

            # else:
            #     sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp(- m * (age )* Tmax + 0.5 * m * (Tmax)**2)     # with advection -- NON CONSTANT


        # data = np.loadtxt('da_convergence/num_' + str(i) + '.txt')      # Load in relevant data.
                # Load in relevant data for both mesh sizes
        if isinstance(dt, np.ndarray):
            data = np.loadtxt(f'{folder}/solutions/num_{i}.txt') 
        else:
            data = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i}.txt') 

        # Calculate the total population 
        totalPop_sol = trapezoidal_rule( sol,            ds[i])      # solution for different ds
        print('Exact total pop      = ' + str(totalPop_sol))

        totalPop_num = trapezoidal_rule( data[-1,:],     ds[i])
        print('Numerical total pop  = ' + str(totalPop_num))


        Norm1[i]    = np.abs( totalPop_num - totalPop_sol )
        print('Norm of total pop    = ' + str(Norm1[i]))


        # Calculate the order of convergence for norms
        if i > 0:

            # Check if Norm1 values are zero or NaN
            if Norm1[i] == 0 or Norm1[i-1] == 0:
                print(f"Warning: Zero error value at index {i} or {i-1}.")
            if np.isnan(Norm1[i]) or np.isnan(Norm1[i-1]):
                print(f"Warning: NaN error value at index {i} or {i-1}.")

            # Calculate the order of convergence for norm
            L1norm[i]   = np.log(Norm1[i-1]   / Norm1[i])   / np.log(ds[i-1] / ds[i])


    for i in range(0, 5):

        print('For ds ='                + str( ds[i] ) )
        print('Norm1 (abs error): '     + str( Norm1[i] ))
        if i > 0:
            print('L1 q order: '        + str( L1norm[i] ) )


    # Plot absolute error oftotal population over time
    # plt.figure(figsize=(8, 6))  # Adjust the width and height as needed
    plt.loglog(ds, Norm1, label='Norm 1')
    plt.loglog(ds, ds**(order), label=f'order-{(order) }')

    plt.xlabel(r'$\Delta a$')
    plt.ylabel('Norm')
    plt.title('Total Population Convergence')
    plt.legend()

    # Convert ds array values to a string
    ds_values_str = '_'.join(map(str, np.round(ds, 3) ))

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


    # combine = [Norm1, L1norm]

    return Norm1, L1norm
