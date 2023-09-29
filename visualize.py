import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys  # Import the sys module to handle keyboard interrupts

CAR_DIMENSIONS = (1000, 1000)
OBJ_DIMENSIONS = (1000, 1000)
SPEED_VECTOR_MULTIPLIER = 5


def update(frame):
    ax.clear()

    car_speed = [df["VehicleSpeed"].iloc[frame], 0]
    car_rect = patches.Rectangle((-CAR_DIMENSIONS[0] / 2, -CAR_DIMENSIONS[1] / 2), CAR_DIMENSIONS[0], CAR_DIMENSIONS[1],
                                 linewidth=1, edgecolor='black', facecolor='red')
    ax.add_patch(car_rect)
    xs = [0, car_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [0, car_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)  # this speed vector does not consider the yawRate!!!

    # Draw detected objects
    obj1_position = [df["FirstObjectDistance_X"].iloc[frame], df["FirstObjectDistance_Y"].iloc[frame]]
    obj1_speed = [df["FirstObjectSpeed_X"].iloc[frame], df["FirstObjectSpeed_Y"].iloc[frame]]
    obj1_rect = patches.Rectangle((obj1_position[0] - OBJ_DIMENSIONS[0] / 2,
                                   obj1_position[1] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(obj1_rect)
    xs = [obj1_position[0], obj1_position[0] + obj1_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [obj1_position[1], obj1_position[1] + obj1_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    obj2_position = [df["SecondObjectDistance_X"].iloc[frame], df["SecondObjectDistance_Y"].iloc[frame]]
    obj2_speed = [df["SecondObjectSpeed_X"].iloc[frame], df["SecondObjectSpeed_Y"].iloc[frame]]
    obj2_rect = patches.Rectangle((obj2_position[0] - OBJ_DIMENSIONS[0] / 2,
                                   obj2_position[1] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(obj2_rect)
    xs = [obj2_position[0], obj2_position[0] + obj2_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [obj2_position[1], obj2_position[1] + obj2_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    obj3_position = [df["ThirdObjectDistance_X"].iloc[frame], df["ThirdObjectDistance_Y"].iloc[frame]]
    obj3_speed = [df["ThirdObjectSpeed_X"].iloc[frame], df["ThirdObjectSpeed_Y"].iloc[frame]]
    obj3_rect = patches.Rectangle((obj3_position[0] - OBJ_DIMENSIONS[0] / 2,
                                   obj3_position[1] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(obj3_rect)
    xs = [obj3_position[0], obj3_position[0] + obj3_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [obj3_position[1], obj3_position[1] + obj3_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    obj4_position = [df["FourthObjectDistance_X"].iloc[frame], df["FourthObjectDistance_Y"].iloc[frame]]
    obj4_speed = [df["FourthObjectSpeed_X"].iloc[frame], df["FourthObjectSpeed_Y"].iloc[frame]]
    obj4_rect = patches.Rectangle((obj4_position[0] - OBJ_DIMENSIONS[0] / 2,
                                   obj4_position[1] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='m', facecolor='none')
    ax.add_patch(obj4_rect)
    xs = [obj4_position[0], obj4_position[0] + obj4_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [obj4_position[1], obj4_position[1] + obj4_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_aspect('equal')  # Set fixed aspect ratio


def on_close(event):
    # This function will be called when the plot window is closed
    sys.exit(0)


df = pd.read_csv('data.csv')

lineCount = df.shape[0]

# Specify a fixed figure size (10x10)
fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=lineCount, interval=2)

# Attach an event handler to the plot window to handle window close events
fig.canvas.mpl_connect('close_event', on_close)

plt.show()
