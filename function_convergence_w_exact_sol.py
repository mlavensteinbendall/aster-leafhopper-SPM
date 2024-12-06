import numpy as np # Numpy for numpy
import matplotlib.pyplot as plt
from function_model import mortality
import re
import zipfile
from function_trapezoidal_rule import trapezoidal_rule


def convergence_da_plt(age_max, time_max, da, dt, order, par, folder):
    """Calculates the convergence for varying ds and constant dt.
    
    Args:
        age_max (int):      The max number of age-steps.
        time_max(float):    The max number of time-steps.
        da      (float):    The partitions for age-steps.
        dt      (float):    The partitions for time-steps.
        order   (int):      The order of the method.
        c       (int):      The constant of death rate.
        
    Returns:
        Norm2   (array):    A list of the 2-norms.
        L2norm  (array):    A list of the order of the 2-norms.
        NormMax (array):    A list of the infinity-norms.
        LMaxnorm(array):    A list of the order of the infinity-norms.
    """
    print('Calculate convergence using the analytic solution (fixed dt).')

    n = int(time_max/dt) + 1 # Time-step of comparison.
    Tend = n*dt # Get the associated timepoint value.

    Ntest = len(da)

    Norm2 = np.zeros([Ntest])
    NormMax = np.zeros([Ntest])

    L2norm = np.zeros([Ntest])
    LMaxnorm = np.zeros([Ntest])


    for i in range(0, Ntest):

        # initialize values
        age_num_points = int(age_max / da[i]) + 1
        age = np.linspace(0, age_max, age_num_points)
        Nage = len(age)                                 # number of elements in age

        mu = np.zeros(Nage)
        mu = mortality(age, par)

        data = np.zeros([Nage])    # initialize matrix for numerical solution
        sol = np.zeros([Nage])                          # initialize array for analytical solution

        # Numerical solution -- download relevent data
        data = last_time_solution(folder, i)

        # Analyticial solution -- changes for what ds is
        sol = np.exp(-(age - ( Tend + 5))**2) * np.exp( - mu * Tend)    # with advection 
        # sol = np.exp(-(age - 5)**2) * np.exp( - mu * Tend)            # without advection        
        # sol = np.exp(-(age - ( Tend + 5))**2) * np.exp(- (30 * np.log(age**2 + 30**2) - 30 * np.log((age - Tend)**2 +30**2))) # with advection -- hill function

        # plt data vs sol
        plt.plot(age,data[:]) 
        plt.plot(age,sol) # looks right
        plt.show()

        # Calculate the norm 
        Norm2[i]    = ( ( 1 / Nage ) * np.sum( np.abs( data[:] - sol[:] ) **2 ) ) **0.5  # L2 error.
        NormMax[i]  = np.max( np.abs( data[:] - sol[:] ) )                               # L-Max error.


    # Iterate to calculates the L norms -- comparing with the last (Note: ds are decressing)
    for ii in range(0, Ntest - 1):
        L2norm[ii+1]    = np.log( Norm2[ii+1]   / Norm2[ii] )   / np.log( da[ii+1] / da[ii] )
        LMaxnorm[ii+1]  = np.log( NormMax[ii+1] / NormMax[ii] ) / np.log( da[ii+1] / da[ii] )


    # Print error and order for each combination of ds and dt
    for i in range(0, Ntest):

        print('For ds ='            + str( round( da[i],        10  ) ) )
        print('Norm 2 error: '      + str( round( Norm2[i],     10  ) ) )
        print('Norm inf error: '    + str( round( NormMax[i],   10  ) ) )

        if i > 0:
            print('L2 q order: '    + str( round( L2norm[i-1]   , 10    ) ) ) # L2 q estimate.
            print('LMax q order: '  + str( round( LMaxnorm[i-1] , 10    ) ) ) # L-Max q estimate.
            print(' ')


    # Plot the log-log for the errors.
    plt.loglog(da, Norm2, label='Norm2')
    plt.loglog(da, NormMax, label='NormMax')
    plt.loglog(da, da**(order), label=f'order-{order }')

    plt.xlabel(r'$\Delta a$')
    plt.ylabel('Norm')
    plt.title('Convergence based on ' + r'$\Delta a$')
    plt.legend()

    # Convert ds array values to a string
    # ds_values_str = '_'.join(map(str, da))

    # Save the plot to a file -- labels with da values and dt 
    # plt.savefig(folder + '/plots/dt_' + str(dt) + '/da_convergence_exact_for_da_' + str(ds_values_str) + '_dt_' + str(dt) + '.png', dpi=300)
    plt.show()
    plt.close()

    return Norm2, L2norm, NormMax, LMaxnorm


def convergence_dt_plt(age_max, Tmax, da, dt, order, par, folder):
    """Calculates the convergence for varying ds and dt.
    
    Args:
        Smax    (int):      The max number of steps.
        Tmax    (float):    The max number of time-steps.
        ds      (float):    The partitions for steps.
        dt      (float):    The partitions for time-steps.
        order   (int):      The order of the method.
        c       (int):      The constant of death rate.
        
    Returns:
        Norm2   (array):    A list of the 2-norms.
        L2norm  (array):    A list of the order of the 2-norms.
        NormMax (array):    A list of the infinity-norms.
        LMaxnorm(array):    A list of the order of the infinity-norms.
    """
    print('Calculate convergence using the analytic solution (varied dt).')
    Ntest = len(da)                 # number of cases

    # initialize the arrays for the norms and norm orders to zero
    Norm2 = np.zeros([Ntest])       # 2 norm
    NormMax = np.zeros([Ntest])     # infinity norm
    L2norm = np.zeros([Ntest])      # order for 2 norm
    LMaxnorm = np.zeros([Ntest])    # order for infinity norm


    # Iterate over the number of tests (0 to Ntest)
    for i in range(0, Ntest):

        # initialize values
        age_num_points = int(age_max / da[i]) + 1
        age = np.linspace(0, age_max, age_num_points)
        Nage = len(age)                                 # number of elements in age

        mu = np.zeros(Nage)
        mu = mortality(age, par)

        Ntime = int(Tmax/dt[i])     # Time-step of comparison.
        Tend = Ntime * dt[i]        # Get the associated timepoint value.

        # data = np.zeros([int(Tmax/ds[i]), Nstep])       # initialize matrix for numerical solution
        data = np.zeros([Nage])                   # initialize matrix for numerical solution
        sol = np.zeros([Nage])                           # initialize array for analytical solution


        # Numerical solution -- download relevent data
        # data = np.loadtxt(f'{folder}/solutions/num_{i}.txt') 
        data = last_time_solution(folder, i)

        # Analyticial solution -- changes for what ds is
        sol = np.exp(-(age - ( Tend + 5))**2) * np.exp(-mu * Tend)     # with advection -- constant
        # sol = np.exp(-(step - ( Tend + 5))**2) * np.exp( - m *  step  * Tend + m * Tend**2 * 0.5)     # with advection -- non-constant
        # sol = np.exp(-(step - 5)**2) * np.exp( - mu * Tend)             # without advection

        # sol = np.exp(-(step - ( Tend + 5))**2) * np.exp(- m * step * Tend + 0.5 * m * (Tend)**2)     # with advection -- NON CONSTANT - slope

        # sol = np.exp(-(step - ( Tend + 5))**2) * np.exp(- (Tend + 200 * np.log(step**2 + 400) - 200 * np.log((step - Tend)**2 +400))) # hill function
        # sol = np.exp(-(step - ( Tend + 5))**2) * np.exp(- (20 * np.log(step**2 + 400) - 20 * np.log((step - Tend)**2 +400)))  # hill function

        # Calculate the norm 
        Norm2[i]    = np.sqrt(np.mean((data[:] - sol[:])**2))       # L2 norm
        NormMax[i]  = np.max( np.abs( data[:] - sol[:] ) )          # Lmax norm


    # Iterate to calculates the L norms -- comparing with the last (Note: ds and dt are decressing with same CFL value)
    for ii in range(0, Ntest - 1):
        L2norm[ii+1]    = np.log( Norm2[ii+1]   / Norm2[ii] )   / np.log( dt[ii+1] / dt[ii] )   # order from L2
        LMaxnorm[ii+1]  = np.log( NormMax[ii+1] / NormMax[ii] ) / np.log( dt[ii+1] / dt[ii] )   # order from Lmax


    # Print error and order for each combination of ds and dt
    for i in range(0, Ntest):

        print('For ds ='    + str( round( da[i],10      ) ) + ' and dt ='    + str( round( dt[i],10      ) ) )
        print('Norm 2 error: '   + str( round( Norm2[i], 10  ) ) )
        print('Norm inf error: ' + str( round( NormMax[i], 10) ) )

        if i > 0:
            print('L2 q order: '     + str( round( L2norm[i-1]   , 10    ))) # L2 q estimate.
            print('LMax q order: '   + str( round( LMaxnorm[i-1] , 10    ))) # L-Max q estimate.
            print(' ')


    plt.loglog(dt, Norm2, label='Norm2')
    plt.loglog(dt, NormMax, label='NormMax')
    plt.loglog(dt, dt**(order), label=f'order-{order }')

    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Norm')
    plt.title('Numerical Convergence with varied ' + r'$\Delta a$' + ' and ' + r'$\Delta t$')
    plt.legend()

    # Convert ds array values to a string
    # ds_values_str = '_'.join(map(str, np.round(ds, 3) ))
    # dt_values_str = '_'.join(map(str, np.round(dt, 3)))


    # Save the plot to a file -- labels with da values and dt 
    # plt.savefig(folder + '/plots/dt_convergence_exact_for_da_' + str(ds_values_str) + '_dt_' + str(dt_values_str) + '.png', dpi=300)
    plt.show()
    plt.close()

    return Norm2, L2norm, NormMax, LMaxnorm




def conservation_plt(ds, c, Smax, Tmax, order, folder):
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

    print('Calculate convergence using the analytic solution.')

    Norm1 = np.zeros([5])
    L1norm = np.zeros([5])


    # Calculate the total population using trapezoidal rule
    for i in range(5):
        totalPop_num = 0
        totalPop_sol = 0

        # age = np.arange(0, Smax + ds[i], ds[i])      # array from 0 to age_max
        # initialize values
        age_num_points = int(Smax / ds[i]) + 1
        age = np.linspace(0, Smax, age_num_points)
        Nage = len(age)                                 # number of elements in age

        mu = np.zeros(Nage)
        mu = mortality(age, c)

        # Analyticial solution -- changes for what ds is
        sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp(-mu * Tmax)     # with advection -- constant

        # if constant == True:
        #     sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp( - c * Tmax)     # with advection -- CONSTANT
        # else:
        #     if hill_func == True:
        #         sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp(- (30 * np.log(age**2 + 30**2) - 30 * np.log((age - Tmax)**2 +30**2))) # with advection -- hill function
                # sol = np.exp(-(age - (Tend + 5))**2) / ((age**2 + 30**2) / ((age - Tend)**2 + 30**2))**30

            # else:
            #     sol = np.exp(-(age - ( Tmax + 5))**2) * np.exp(- m * (age )* Tmax + 0.5 * m * (Tmax)**2)     # with advection -- NON CONSTANT


        # data = np.loadtxt('da_convergence/num_' + str(i) + '.txt')      # Load in relevant data.
                # Load in relevant data for both mesh sizes
        # if isinstance(dt, np.ndarray):
        #     data = np.loadtxt(f'{folder}/solutions/num_{i}.txt') 
        # else:
        #     data = np.loadtxt(f'{folder}/solutions/dt_{dt}/num_{i}.txt') 
        data = last_time_solution(folder, i)

        # Calculate the total population 
        totalPop_sol = trapezoidal_rule( sol,            ds[i])      # solution for different ds
        print('Exact total pop      = ' + str(totalPop_sol))

        totalPop_num = trapezoidal_rule( data[:],     ds[i])
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
    # ds_values_str = '_'.join(map(str, np.round(ds, 3) ))

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