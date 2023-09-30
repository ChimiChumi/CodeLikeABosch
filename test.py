import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data.csv')

# Initialize the vehicle's position
vehicle_position = np.array([0, 0])

# Create a new figure
fig, ax = plt.subplots()

# Loop over each row in the dataframe
for i, row in df.iterrows():
    # Clear the current plot
    ax.clear()

    # Update the vehicle's position based on its speed and yaw rate
    vehicle_speed = row['VehicleSpeed'] / 256  # Convert to m/s
    yaw_rate = np.deg2rad(row['YawRate'])  # Convert to radians
    vehicle_position = np.array([0.0, 0.0])
    vehicle_position += vehicle_speed * np.array([np.cos(yaw_rate), np.sin(yaw_rate)])

    # Plot the vehicle
    ax.scatter(*vehicle_position, color='red')
    ax.arrow(*vehicle_position, np.cos(yaw_rate), np.sin(yaw_rate), color='red')

    # Plot each object
    for obj in ['First', 'Second', 'Third', 'Fourth']:
        # Update the object's position based on its speed
        object_speed = np.array([row[f'{obj}ObjectSpeed_X'], row[f'{obj}ObjectSpeed_Y']]) / 256  # Convert to m/s
        object_position = vehicle_position + np.array([row[f'{obj}ObjectDistance_X'], row[f'{obj}ObjectDistance_Y']])
        object_position += object_speed

        ax.scatter(*object_position, color='blue')
        ax.arrow(*object_position, *object_speed, color='blue')

    # Set the x and y limits
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)

    # Show the plot
    plt.pause(0.01)

plt.show()