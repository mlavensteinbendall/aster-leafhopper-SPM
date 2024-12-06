import numpy as np

# Parameters
da = 0.00625
dt1 = 0.00625  # First time step
dt2 = 0.1            # Second time step

# Time arrays
time1_num_points = int(10 / dt1) + 1
time1 = np.linspace(0, 10, time1_num_points)

time2_num_points = int(10 / dt2) + 1
time2 = np.linspace(0, 10, time2_num_points)

# print(f"Number of points in time1: {len(time1)/8}")
# print(f"Number of points in time2: {len(time2)}")


# Indices to save
beep1 = []  # Initialize as an empty list
for t, T in enumerate(time1):
    if t == 0 or t == len(time1) - 1 or t % 10 == 0:
        beep1.append(T)

beep2 = []  # Initialize as an empty list
for t, T in enumerate(time2):
    if t == 0 or t == len(time2) - 1 or t % 10 == 0:
        beep2.append(T)

# Convert to numpy arrays if needed
beep1 = np.array(beep1)
beep2 = np.array(beep2)

# Output
print("Beep1 indices:", beep1)
print("Beep2 indices:", beep2)
