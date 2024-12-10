import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import zipfile
import re

# Load the relevant data
# population_data = np.loadtxt('convergence/reproduction/no_mortality/constant_reproduction/rep_1/varied_dt/solutions/num_0.txt') 
# population_data = np.loadtxt('convergence/reproduction/no_mortality/linear_reproduction/varied_dt/solutions/num_0.txt') 
# population_data = np.loadtxt('/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/reproduction/no_mortality/gaussian_reproduction/varied_dt/solutions/num_0.txt')
# population_data = np.loadtxt('/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/reproduction/no_mortality/test/varied_dt/solutions/num_0.txt')
# population_data = np.loadtxt('/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/reproduction/no_mortality/constant_reproduction/rep_1/varied_dt/solutions/num_4.txt')

# gompertz function
# population_data = np.loadtxt('/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/mortality/gompertz_mortality/varied_dt/solutions/num_4.txt')

# gompertz function - fixed
# population_data = np.loadtxt('/Users/mlavensteinbendall/Documents/ALF_SPM_AGE/convergence/mortality/gompertz_mortality/fixed_dt/solutions/dt_0.0001/num_0.txt')


# constant function mortality
# population_data = np.loadtxt('convergence/mortality/constant_mortality/varied_dt/solutions/num_4.txt')
i = 4
da = 0.00625
dt = 0.5 * da

folder = 'convergence/'
zip_filename = folder + f"_results_{i}_da_{da}_dt_{dt}.zip"
output_dir = "unzipped_output"  # Directory to extract files

# Open the ZIP file
with zipfile.ZipFile(zip_filename, 'r') as zf:
    # List all files in the ZIP
    file_list = zf.namelist()
    
    # Sort the files (important if time steps should be in order)
    file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

    print(file_list)
    
    # Read each file into memory
    population_data = []
    for file in file_list:
        with zf.open(file) as f:
            population_data.append(np.load(f))  # Load the .npy file into a numpy array


data = np.stack(population_data, axis = 0)
print(data.shape)
age_max = 30
# da = 0.1

Tmax = 20
save_rate = 10

time_num_points = int(Tmax / dt) + 1
time = np.linspace(0, Tmax, time_num_points)

times = []

# for t,T in enumerate(time):
#     if t == 0 or t == len(time) - 1 or t % save_rate == 0:
#         times.append(T)

# times = np.array(times)

# print(len(times))

# # Number of time steps is the number of outer elements
time_steps = len(population_data)

print(time_steps)

# Number of age groups is the number of inner elements
age_groups = len(population_data[0])

print(age_groups)

# Generate time points corresponding to saved population data
times = [t for t, T in enumerate(time) if t == 0 or t == len(time) - 1 or t % save_rate == 0]
times = np.array([time[t] for t in times])  # Select the corresponding times from the original time array

print(len(times))

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, age_max)  # Rescale the x-axis to be from 0 to 15
ax.set_ylim(-0.1, np.max(population_data))  # Population numbers on the y-axis
line, = ax.plot([], [], lw=2.5, color = 'black')

# Create an array of age labels from 0 to age_max
age_labels = np.linspace(0, age_max, age_groups)

# Initialize the plot with empty data
def init():
    line.set_data([], [])
    return line,

# Update function that will be called at each time step
def update(frame):
    # X-axis: Corresponding age labels, Y-axis: Population numbers at the current time step
    y = population_data[frame]
    line.set_data(age_labels, y)  # Use the age labels for the x-values
    ax.set_title(f"Age Distribution of Population at time = {round(times[frame],0)}", fontsize=14)  # Update the title
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, time_steps), init_func=init, blit=False, interval=50)

# Set axis labels
plt.xlabel('Age', fontsize=12)
plt.ylabel('Population', fontsize=12)

# Set tick label sizes without changing tick positions
ax.tick_params(axis='x', labelsize=10)  # Set x-axis tick label size
ax.tick_params(axis='y', labelsize=10)  # Set y-axis tick label size


# Save the animation
# ani.save('population_dynamics_gompertz.gif', writer='pillow', fps=30)  # Save as GIF
# ani.save('population_dynamics_constant.gif', writer='pillow', fps=30)  # Save as GIF
ani.save('population_dynamics_2.gif', writer='pillow', fps=30)  # Uncomment to save as MP4
# ani.save('population_dynamics_logistic')


# Show the animation
plt.show()