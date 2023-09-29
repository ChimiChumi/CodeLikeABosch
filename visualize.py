import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys  # Import the sys module to handle keyboard interrupts

CAR_DIMENSIONS = (1000, 1000)
OBJ_DIMENSIONS = (1000, 1000)

def update(frame):
    ax.clear()

    car_rect = patches.Rectangle((-CAR_DIMENSIONS[0]/2, -CAR_DIMENSIONS[1]/2), CAR_DIMENSIONS[0], CAR_DIMENSIONS[1], linewidth=1, edgecolor='black', facecolor='red')
    print(car_rect)
    ax.add_patch(car_rect)

    # Draw detected objects

    obj1_rect = patches.Rectangle((df["FirstObjectDistance_X"].iloc[frame] - OBJ_DIMENSIONS[0] / 2,
                                   df["FirstObjectDistance_Y"].iloc[frame] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(obj1_rect)
    obj2_rect = patches.Rectangle((df["SecondObjectDistance_X"].iloc[frame] - OBJ_DIMENSIONS[0] / 2,
                                   df["SecondObjectDistance_Y"].iloc[frame] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(obj2_rect)
    obj3_rect = patches.Rectangle((df["ThirdObjectDistance_X"].iloc[frame]-OBJ_DIMENSIONS[0]/2,
                                   df["ThirdObjectDistance_Y"].iloc[frame]-OBJ_DIMENSIONS[0]/2),
                                  OBJ_DIMENSIONS[0],OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(obj3_rect)
    obj4_rect = patches.Rectangle((df["FourthObjectDistance_X"].iloc[frame] - OBJ_DIMENSIONS[0] / 2,
                                   df["FourthObjectDistance_Y"].iloc[frame] - OBJ_DIMENSIONS[0] / 2),
                                  OBJ_DIMENSIONS[0], OBJ_DIMENSIONS[1],
                                  linewidth=1, edgecolor='m', facecolor='none')
    ax.add_patch(obj4_rect)

    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_aspect('equal')  # Set fixed aspect ratio
    plt.gca().invert_yaxis()

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
