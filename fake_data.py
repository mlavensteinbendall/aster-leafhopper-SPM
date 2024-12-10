# import numpy                        as np
# import matplotlib.pyplot            as plt 
# import random   

# Amax = 30
# da = 0.1
# # initalize age array
# age_num_points = int(Amax / da) + 1
# age = np.linspace(0, Amax, age_num_points)


# y = np.exp(-1*(age-30)) 


# # Define the proportional error for synthetic data generation
# sigma = 0.1  # Set the proportional error to 10% of the true value
# # Generate synthetic data with proportional error
# # Each data point in 'y' is scaled by (1 + a random error term from a normal distribution scaled by sigma)
# prop_data = y * np.array([(1 + sigma * np.random.randn(y.shape[0]))]).T

# print(prop_data)


# # s_data1 = np.exp(-1*(age-30)) 

# plt.plot(prop_data)
# plt.show()
import matplotlib.pyplot as plt

# # Example survival data
# time_temp1 = [0, 1, 3, 5, 8, 10]
# survival_temp1 = [100, 100, 90, 90, 80, 80]

# time_temp2 = [0, 1, 3, 5, 7]
# survival_temp2 = [100, 80, 75, 60, 0]

# time_temp3 = [0, 1, 3, 5, 8, 9]
# survival_temp3 = [100, 60, 40, 35, 15, 5]

# time_temp4 = [0, 1, 3, 5]
# survival_temp4 = [100, 30, 10, 0]

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.step(time_temp1, survival_temp1, where='post', label='control', color='black', linewidth=3)
# plt.step(time_temp2, survival_temp2, where='post', label='Warm', color='green', linewidth=3)
# plt.step(time_temp3, survival_temp3, where='post', label='Hot', color='orange', linewidth=3)
# plt.step(time_temp4, survival_temp4, where='post', label='Hotter', color='red', linewidth=3)

# # Adding labels and title
# # plt.title("Survival at Different Temperatures", fontsize=14)
# plt.xlabel("Time Elapsed", fontsize=12)
# plt.ylabel("Survival Probability (%)", fontsize=12)
# # plt.ylim(0, 100)  # Ensure y-axis goes from 0 to 100
# # plt.legend(title="Temperature", fontsize=10)


# # Show plot
# plt.show()



time_temp1 = [0, 1, 3, 5, 8, 10]
survival_temp1 = [0, 15, 20, 20, 25, 25]

time_temp2 = [0, 1.2, 3.2, 5.2, 8.2, 10]
survival_temp2 = [0, 10, 10, 15, 20, 20]

time_temp3 = [0, 1.5, 3.5, 4.5, 8.5, 10]
survival_temp3 = [0, 6, 10, 12, 12, 10]

time_temp4 = [0, 1, 3, 5, 8]
survival_temp4 = [0, 0, 5, 5, 5]



# Plotting
plt.figure(figsize=(8, 6))
plt.step(time_temp1, survival_temp1, where='post', label='control', color='black', linewidth=3)
plt.step(time_temp2, survival_temp2, where='post', label='Warm', color='green', linewidth=3)
plt.step(time_temp3, survival_temp3, where='post', label='Hot', color='orange', linewidth=3)
plt.step(time_temp4, survival_temp4, where='post', label='Hotter', color='red', linewidth=3)

# Adding labels and title
# plt.title("Survival at Different Temperatures", fontsize=14)
# plt.xlabel("Time Elapsed", fontsize=12)
# plt.ylabel("Survival Probability (%)", fontsize=12)
# plt.ylim(0, 100)  # Ensure y-axis goes from 0 to 100
# plt.legend(title="Temperature", fontsize=10)


# Show plot
plt.show()
