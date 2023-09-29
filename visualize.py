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

    car_speed = [0, df["VehicleSpeed"].iloc[frame]]
    car_rect = patches.Rectangle((-CAR_DIMENSIONS[1] / 2, -CAR_DIMENSIONS[0] / 2),
                                 CAR_DIMENSIONS[1], CAR_DIMENSIONS[0],
                                 linewidth=1, edgecolor='black', facecolor='red')
    ax.add_patch(car_rect)
    xs = [0, car_speed[0] * SPEED_VECTOR_MULTIPLIER]
    ys = [0, car_speed[1] * SPEED_VECTOR_MULTIPLIER]
    ax.plot(xs, ys)

    draw_object([df["FirstObjectDistance_Y"].iloc[frame], df["FirstObjectDistance_X"].iloc[frame]],
                [df["FirstObjectSpeed_Y"].iloc[frame], df["FirstObjectSpeed_X"].iloc[frame]], 'b')

    draw_object([df["SecondObjectDistance_Y"].iloc[frame], df["SecondObjectDistance_X"].iloc[frame]],
                [df["SecondObjectSpeed_Y"].iloc[frame], df["SecondObjectSpeed_X"].iloc[frame]], 'y')

    draw_object([df["ThirdObjectDistance_Y"].iloc[frame], df["ThirdObjectDistance_X"].iloc[frame]],
                [df["ThirdObjectSpeed_Y"].iloc[frame], df["ThirdObjectSpeed_X"].iloc[frame]], 'g')

    draw_object([df["FourthObjectDistance_Y"].iloc[frame], df["FourthObjectDistance_X"].iloc[frame]],
                [df["FourthObjectSpeed_Y"].iloc[frame], df["FourthObjectSpeed_X"].iloc[frame]], 'm')

    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_aspect('equal')


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
