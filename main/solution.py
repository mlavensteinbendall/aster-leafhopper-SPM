
import numpy                        as np
import matplotlib.pyplot            as plt 
from function_numerical_method                    import LW_SPM
from function_reproduction          import reproduction

from function_mortality import mortality

## INITIAL CONDITIONS
Amax = 30       # max age
Tmax = 5     # max time
order = 2       # order of method
Ntest = 5       # number of cases


# Mortality set up
m = 0       #1/30            # constant for mux
c = round(m)
b = 0              # y-intercept

da = np.zeros([Ntest]) # order smallest to largest
da[0] = 0.1
da[1] = 0.05
da[2] = 0.025
da[3] = 0.0125
da[4] = 0.00625

dt = np.zeros([Ntest]) # order smallest to largest
dt = 0.5 * da

# 
for i in range(len(da)):

    print('Entering loop ' + str(i))                # progress update, loop began

    # initalize arrays
    age = np.arange(0, Amax + da[i], da[i])       # array from 0 to Amax
    Nage = len(age)                               # number of elements in age
    print("age:", age[-1])                        # check that last element is Amax

    mu = np.zeros(Nage)
    mu = mortality(Amax, age, m, b, True, False, False)

    # initalize arrays
    time = np.arange(0,Tmax + dt[i], dt[i])     # array from 0 to Tmax
    Ntime = len(time)                           # number of elements in time
    print("Time:", time[-1])                    # check that last element is Tmax
    
    ## ANALYTICAL SOLUTION 
    # initialize analytical solution matrix
    sol = np.zeros([len(time),len(age)])

    # calculate the analytical solution for every age at time t
    for i_t in range(0, len(time)):
        # Calculate the analytical solution

        for i_a in range(0, len(age)):

            sol[i_t, i_a] = np.exp(-(age[i_a] - ( time[i_t] + 5))**2) * np.exp( - mu[i_a] * time[i_t])     # with advection -- CONSTANT

        sol[i_t, 0] = reproduction(np.exp(-(age - ( time[i_t] + 5))**2) * np.exp( - mu * time[i_t]), 0, da[i])
        # if i_t > 0:
            # print('type = ' + str(type(sol[i_t, :])))
            # print('shape = ' + str((sol[i_t, :]).shape))
            # print(type(sol[i_t, 0]))
            # print('beepbeep' + str(type(reproduction(sol[i_t, :], 0, da) )))
            # temp = sol[i_t, :]
            # print(temp.shape)
            # sol[, 0] = reproduction(temp, 0, da) 
            # print(type(sol[i_t,0]))
            # print("type of boundry:" + str(type(reproduction(temp, 0, da) )))

    plot_indices = [0, Ntime // 2, Ntime - 1]

    plt.close()
    # plot numerical and analytical solution
    for t_index in plot_indices:
        plt.plot(age, sol [t_index, :], label=f'Analytical at time {round(time[t_index], 1)  }', linestyle='-')     # analytical 
    plt.axhline(y=1, color='r', linestyle='--', label='y=1')
    plt.xlabel('Age')
    plt.ylabel('Population')
    plt.title('Age Distribution of Population (' + r'$\Delta a$' + ' = ' + str(da[i]) + ', ' + r'$\Delta t$' + ' = ' + str(dt[i]) + ')')
    plt.legend()
    plt.show()