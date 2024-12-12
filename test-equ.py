import numpy as np
import matplotlib.pyplot as plt

age_num_points = int(30 / 0.1) + 1
age = np.linspace(0, 30, age_num_points)
Nage = len(age)                               # number of elements in age
print("Age:", age[-1]) 

time = 20
# term1=  5 * np.log( np.exp( -2/5 * (age-time)) * (np.exp(2/5 * time+8) + np.exp(2/5 * (age -time)) )) 

# term2=  5* np.log( np.exp(-2/5* (age-time))*(np.exp(2/5*(age-time)+ 8)+np.exp(8)) )

# beep = 0.5 * (term1 - 2 * time - term2)

# term1 = np.exp(8 - 2/5 * (age - 2 * time) )
# term2 = np.exp(8 - 2/5 * (age - time))

# beep = 5/2 * np.log(term1 + 1) + time - 5/2 * np.log(term2 + 1)

term1 = (np.exp(8-2*time/5) + np.exp(2*(age-time)/5)) / (np.exp(8) + np.exp(2*(age-time)/5))

beep = 5/2 * np.log(term1) + time

sol = np.exp(-(age - ( time + 5))**2) * np.exp(beep )

plt.plot(age,sol)
plt.show()