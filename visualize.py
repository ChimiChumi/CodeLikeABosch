import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys  # Import the sys module to handle keyboard interrupts

CAR_DIMENSIONS = (1000, 1000)
OBJ_DIMENSIONS = (1000, 1000)
SPEED_VECTOR_MULTIPLIER = 5

import numpy as np

def update(frame):
    ax.clear()

    # Car Speed
    car_speed_value = df["VehicleSpeed"].iloc[frame] / 256  # Convert to m/s
    car_speed = [0, df["VehicleSpeed"].iloc[frame]]
    car_rect = patches.Rectangle((-CAR_DIMENSIONS[1] / 2, -CAR_DIMENSIONS[0] / 2),
                                 CAR_DIMENSIONS[1], CAR_DIMENSIONS[0],
                                 linewidth=1, edgecolor='black', facecolor='red')

    # Calculate rotation based on the yaw rate
    yaw_rate = df["YawRate"].iloc[frame]
    rotation_matrix = Affine2D().rotate_deg(yaw_rate)

    # Apply the transformation
    car_rect.set_transform(rotation_matrix + ax.transData)

    ax.add_patch(car_rect)
    xs = [0, car_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [0, car_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    # Object 1 Speed
    obj1_speed_value = np.sqrt(df["FirstObjectSpeed_X"].iloc[frame]**2 + df["FirstObjectSpeed_Y"].iloc[frame]**2) / 256  # Convert to m/s
    draw_object([df["FirstObjectDistance_Y"].iloc[frame], df["FirstObjectDistance_X"].iloc[frame]],
                [df["FirstObjectSpeed_Y"].iloc[frame], df["FirstObjectSpeed_X"].iloc[frame]], 'b')

    # Object 2 Speed
    obj2_speed_value = np.sqrt(df["SecondObjectSpeed_X"].iloc[frame]**2 + df["SecondObjectSpeed_Y"].iloc[frame]**2) / 256  # Convert to m/s
    draw_object([df["SecondObjectDistance_Y"].iloc[frame], df["SecondObjectDistance_X"].iloc[frame]],
                [df["SecondObjectSpeed_Y"].iloc[frame], df["SecondObjectSpeed_X"].iloc[frame]], 'y')

    # Object 3 Speed
    obj3_speed_value = np.sqrt(df["ThirdObjectSpeed_X"].iloc[frame]**2 + df["ThirdObjectSpeed_Y"].iloc[frame]**2) / 256  # Convert to m/s
    draw_object([df["ThirdObjectDistance_Y"].iloc[frame], df["ThirdObjectDistance_X"].iloc[frame]],
                [df["ThirdObjectSpeed_Y"].iloc[frame], df["ThirdObjectSpeed_X"].iloc[frame]], 'g')

    # Object 4 Speed
    obj4_speed_value = np.sqrt(df["FourthObjectSpeed_X"].iloc[frame]**2 + df["FourthObjectSpeed_Y"].iloc[frame]**2) / 256  # Convert to m/s
    draw_object([df["FourthObjectDistance_Y"].iloc[frame], df["FourthObjectDistance_X"].iloc[frame]],
                [df["FourthObjectSpeed_Y"].iloc[frame], df["FourthObjectSpeed_X"].iloc[frame]], 'm')

    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_aspect('equal')

    # Legend
    legend_labels = [
        f'Car Speed: {car_speed_value:.2f} m/s',
        f'Object 1 Speed: {obj1_speed_value:.2f} m/s',
        f'Object 2 Speed: {obj2_speed_value:.2f} m/s',
        f'Object 3 Speed: {obj3_speed_value:.2f} m/s',
        f'Object 4 Speed: {obj4_speed_value:.2f} m/s',
    ]
    ax.legend(legend_labels)



def draw_object(position, speed, color):
    obj_rect = patches.Rectangle((position[0] - OBJ_DIMENSIONS[1] / 2,
                                  position[1] - OBJ_DIMENSIONS[0] / 2),
                                 OBJ_DIMENSIONS[1], OBJ_DIMENSIONS[0],
                                 linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(obj_rect)
    xs = [position[0], position[0] + speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [position[1], position[1] + speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

def on_close(event):
    sys.exit(0)

df = pd.read_csv('data.csv')
lineCount = df.shape[0]

fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=lineCount, interval=2)

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
