import math
import sys  # Import the sys module to handle keyboard interrupts

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

CAR_DIMENSIONS = (5, 5)
OBJ_DIMENSIONS = (5, 5)


def update(frame):
    ax.clear()

    global previous_timestamp
    current_timestamp = df['Timestamp'].iloc[frame]
    delta_time = current_timestamp - previous_timestamp
    previous_timestamp = current_timestamp

    car_speed = df["VehicleSpeed"].iloc[frame] / 256  # Convert to m/s
    car_yaw_rate = df["YawRate"].iloc[frame] * (180/math.pi)

    heading_angle = car_yaw_rate * delta_time
    delta_x = car_speed * math.cos(heading_angle) * delta_time
    delta_y = car_speed * math.sin(heading_angle) * delta_time

    global car_pos
    if frame == 0:
        car_pos = (0, 0)
    car_pos = (car_pos[0] + delta_x, car_pos[1] + delta_y)

    car_rect = patches.Rectangle((car_pos[1] - CAR_DIMENSIONS[1] / 2, car_pos[0] - CAR_DIMENSIONS[0] / 2),
                                 CAR_DIMENSIONS[1], CAR_DIMENSIONS[0],
                                 linewidth=1, edgecolor='black', facecolor='red')
    ax.add_patch(car_rect)

    obj1_relative_pos = (df["FirstObjectDistance_X"].iloc[frame] / 128,
                         df["FirstObjectDistance_Y"].iloc[frame] / 128)
    obj1_pos = (car_pos[0] + obj1_relative_pos[0], car_pos[1] + obj1_relative_pos[1])
    draw_object(obj1_pos, "g")

    obj2_relative_pos = (df["SecondObjectDistance_X"].iloc[frame] / 128,
                         df["SecondObjectDistance_Y"].iloc[frame] / 128)
    obj2_pos = (car_pos[0] + obj2_relative_pos[0], car_pos[1] + obj2_relative_pos[1])
    draw_object(obj2_pos, "b")

    obj3_relative_pos = (df["ThirdObjectDistance_X"].iloc[frame] / 128,
                         df["ThirdObjectDistance_Y"].iloc[frame] / 128)
    obj3_pos = (car_pos[0] + obj3_relative_pos[0], car_pos[1] + obj3_relative_pos[1])
    draw_object(obj3_pos, "y")

    obj4_relative_pos = (df["FourthObjectDistance_X"].iloc[frame] / 128,
                         df["FourthObjectDistance_Y"].iloc[frame] / 128)
    obj4_pos = (car_pos[0] + obj4_relative_pos[0], car_pos[1] + obj4_relative_pos[1])
    draw_object(obj4_pos, "m")

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 100)
    ax.set_aspect('equal')


def draw_object(position, color):
    obj_rect = patches.Rectangle((position[1] - OBJ_DIMENSIONS[1] / 2,
                                  position[0] - OBJ_DIMENSIONS[0] / 2),
                                 OBJ_DIMENSIONS[1], OBJ_DIMENSIONS[0],
                                 linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(obj_rect)


def on_close(event):
    sys.exit(0)


df = pd.read_csv('data.csv')
lineCount = df.shape[0]
previous_timestamp = df['Timestamp'].iloc[0]

car_pos = (0, 0)

fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=lineCount, interval=2)

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
