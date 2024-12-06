import numpy as np
import matplotlib.pyplot as plt
from function_trapezoidal_rule import trapezoidal_rule

## INITIAL CONDITIONS
Amax = 30       # max age
Tmax = 5     # max time
order = 2       # order of method
Ntest = 5       # number of cases


# Mortality set up
m = 0       #1/30            # constant for mux
b = 0              # y-intercept
constant_mortality = True   # True for constant mu, False for function mu
analytical_sol = False
hill_func_mortality = False
linear_slope_func_mortality = False

# Reproduction set up
rep = 1
constant_reproduction = True
linear_reproduction = False

# testing_folder = 'mortality'
testing_folder = 'reproduction'

if constant_mortality == True:
    if m == 0 :
        function_folder = "no_mortality"

    else:
        function_folder = "constant_mortality"

elif hill_func_mortality == True:
    function_folder = "hill_mortality"

elif linear_slope_func_mortality == True:
    function_folder = "linear_mortality"

else:
    function_folder = "gompertz_mortality"

if testing_folder == 'reproduction':

    if constant_reproduction == True:
        if rep == 0:
            function_folder = function_folder + '/' + 'no_reproduction'
            
        else:
            function_folder = function_folder + '/' + 'constant_reproduction/rep_true' + str(rep)

    elif linear_reproduction == True:
        function_folder = function_folder + '/' + 'linear_reproduction'




# need to chose da and dt so that the last value in the array are Amax and Tmax
da = np.zeros([Ntest]) # order smallest to largest

# # vary da and dt cases:
da[0] = 0.1
da[1] = 0.05
da[2] = 0.025
da[3] = 0.0125
da[4] = 0.00625


dt = np.zeros([Ntest]) # order smallest to largest

dt = 0.5 * da
# dt = 0.01
# dt = 0.001
# dt = 0.0001
# dt = 0.00001

if isinstance(dt, np.ndarray):
    convergence_folder = 'varied_dt'

else:
    convergence_folder = 'fixed_dt'


folder = 'convergence/' + testing_folder + '/' + function_folder + '/' + convergence_folder
print(folder)


data = np.loadtxt(f'{folder}/solutions/num_0.txt') 
time = np.arange(0,Tmax + dt[0], dt[0])     # array from 0 to Tmax
age = np.arange(0,Amax + da[0], da[0])     # array from 0 to Tmax
# print(time[8])
# plt.plot(age, data[8,:])
# plt.xlabel('age')
# plt.ylabel('population')
# plt.show()


plt.plot(age, data[len(time)-1,:])
plt.xlabel('age')
plt.ylabel('population')
plt.show()


# print('Population       at (0,0) :   ' + str(data[0,0]))
# print('Total Population at (0,a) :   ' + str(trapezoidal_rule(data[0,:], da[0])))
# print('Population       at (1,0) :   ' + str(data[1,0]))
# print('Total Population at (1,a) :   ' + str(trapezoidal_rule(data[1,:], da[0])))
# print('Population       at (2,0) :   ' + str(data[2,0]))
# print('Total Population at (2,a) :   ' + str(trapezoidal_rule(data[2,:], da[0])))
# print('Population       at (3,0) :   ' + str(data[3,0]))
# print('Total Population at (3,a) :   ' + str(trapezoidal_rule(data[3,:], da[0])))
# print('Population       at (4,0) :   ' + str(data[4,0]))
# print('Total Population at (4,a) :   ' + str(trapezoidal_rule(data[4,:], da[0])))
# print('Population       at (5,0) :   ' + str(data[5,0]))
# print('Total Population at (5,a) :   ' + str(trapezoidal_rule(data[5,:], da[0])))
# print('Population       at (6,0) :   ' + str(data[6,0]))
# print('Total Population at (6,a) :   ' + str(trapezoidal_rule(data[6,:], da[0])))
# print('Population       at (7,0) :   ' + str(data[7,0]))
# print('Total Population at (7,a) :   ' + str(trapezoidal_rule(data[7,:], da[0])))
# print('Population       at (8,0) :   ' + str(data[8,0]))
# print('Total Population at (8,a) :   ' + str(trapezoidal_rule(data[8,:], da[0])))

# # array = np.zeros(len(time))

# # for i in range(len(time)):
# #     array[i] = trapezoidal_rule(data[i,:], da[0])

# # plt.plot(time, array)
# # plt.xlabel('time')
# # plt.ylabel('total population')
# # plt.show()

# plt.plot(age, data[0,:])
# plt.xlabel('age')
# plt.ylabel('population')
# plt.title('initial condition')
# plt.show()

# plt.plot(time, data[:,0])
# plt.xlabel('time')
# plt.ylabel('population')
# plt.title('boundary condition')
# plt.show()

