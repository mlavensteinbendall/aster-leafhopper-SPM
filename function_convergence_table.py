from tabulate import tabulate
import numpy as np


def ensure_size(test, arr):
    """Adds zeros to the begginning of arrays to make sure they all have the number of tests that where performed
    
    Args:
        test    (int):      Number of tests performed
        arr     (array):    Array 
    
    Returns:
        arr     (array):    Array of length of test"""
    
    if len(arr) < test:
        # Calculate how many zeros to prepend
        num_zeros = test - len(arr)
        # Prepend the zeros
        arr = np.pad(arr, (num_zeros, 0), 'constant')
    return arr



def tabulate_conv(dt, ds, Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm, folder):
    """Prints table for latex document.
    
    Args:
        dt      (float):    The partitions for time-steps.
        ds      (int):      The partitions for steps.
        Norm2   (array):    A list of 2-norm errors.
        L2norm  (array):    A list of 2-norm orders.
        NormMax (array):    A list of infinity-norm errors.
        LMaxnorm(array):    A list of infinity-norm orders.
        Norm1   (array):    A list of 1-norm errors.
        L1norm  (array):    A list of 1-norm orders.
        
    Returns:
        Print table
    """

    # Define the headers
    # headers = ["$\Delta{t}$","$\Delta{s}$", "$||N-N_{ex}||_2$", "$q_2$", "$||N-N_{ex}||_\infty$", "$q_\infty$", "$\int$ Error$", "$\int q$"]
    headers = [r"$\Delta{t}$", r"$\Delta{s}$", r"$||N-N_{ex}||_2$", r"$q_2$", r"$||N-N_{ex}||_\infty$", r"$q_\infty$", r"$\int$ Error", r"$\int q$"]


    time = np.zeros([len(ds)])
    time[:] = dt

    test = len(ds)

    Norm2    = ensure_size(test, Norm2)
    L2norm   = ensure_size(test, L2norm)
    NormMax  = ensure_size(test, NormMax)
    LMaxnorm = ensure_size(test, LMaxnorm)
    Norm1    = ensure_size(test, Norm1)
    L1norm   = ensure_size(test, L1norm)


    # Combine the data into a list of tuples
    latex = list(zip(time, ds,  Norm2, L2norm, NormMax, LMaxnorm, Norm1, L1norm))

    # Generate the LaTeX table
    latex_table = tabulate(latex, headers=headers, tablefmt="latex_raw")

    # Print or save the LaTeX table
    print(latex_table)

    # Convert ds array values to a string
    ds_values_str = '_'.join(map(str, np.round(ds, 6) ))

    # save plots to folder
    # if isinstance(dt, np.ndarray):
    #     dt_values_str = '_'.join(map(str, np.round(dt, 6) ))
    #     # file2write=open('da_plot/' + folder + '/varied_dt/lw-ex_plot_mu_' + str(c) + '_da_' + ds_values_str + '_dt_' + dt_values_str + '.txt' ,'w')
    #     file2write=open(folder + '/solutions/order_table_' + '_da_' + ds_values_str + '_dt_' + dt_values_str + '.txt' ,'w')
    #     # file2write.write(latex_table)
    #     # file2write.close()
    # else:
    #     # Save the plot to a file -- labels with da values and dt 
    #     # file2write=open('da_plot/' + folder + '/fixed_dt/lw-ex_plot_mu_' + str(c) + '_da_' + ds_values_str + '_dt_' + str(dt) + '.txt'  , 'w')
    #     file2write=open(folder + '/solutions/dt_' + str(dt) + '/order_table_' + '_da_' + ds_values_str + '_dt_' + str(dt) + '.txt' ,'w')
    # file2write.write(latex_table)
    # file2write.close()
