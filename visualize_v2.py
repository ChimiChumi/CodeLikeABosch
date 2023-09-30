import math
import sys  # Import the sys module to handle keyboard interrupts

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats
from sklearn.impute import KNNImputer

CAR_DIMENSIONS = (2, 2)
OBJ_DIMENSIONS = (2, 2)

def fill_zeros(df, column):
    zero_indices = df[df[column] == 0].index
    for idx in zero_indices:
        if idx == 0 or idx == len(df) - 1:  # If the zero is at the start or end of the column
            continue
        else:
            prev_idx = idx - 1
            next_idx = idx + 1
            while df.loc[next_idx, column] == 0:  # Find the next non-zero value
                next_idx += 1
                if next_idx >= len(df):  # If we've reached the end of the column
                    break
            if next_idx >= len(df):  # If all remaining values are zero
                continue
            else:
                # Calculate the difference between the previous non-zero value and the next non-zero value
                diff = df.loc[next_idx, column] - df.loc[prev_idx, column]
                # Divide it by the number of zero values plus one (to include the next non-zero value)
                increment = diff / (next_idx - prev_idx)
                # Adjust the stop argument to align with the increment
                interpolated_values = [df.loc[prev_idx, column] + increment * i for i in range(1, next_idx - prev_idx)]
                # Replace the zero values
                df.loc[idx:next_idx-1, column] = interpolated_values
    return df

# Read your data
df = pd.read_csv('data.csv')

# Fill zeros for each column that you want to process
columns_to_fill = ['FirstObjectDistance_X', 'FirstObjectDistance_Y', 'SecondObjectDistance_X', 'SecondObjectDistance_Y', 'ThirdObjectDistance_X', 'ThirdObjectDistance_Y', 'FourthObjectDistance_X', 
'FourthObjectDistance_Y']
for column in columns_to_fill:
    # df = fill_zeros(df, column)
    z_scores = np.abs(stats.zscore(df[column]))
    # # Identify outliers
    outliers = z_scores > 2
    # # Replace outliers with median
    # df[column][outliers] = df[column].median()
    # imputer = KNNImputer(n_neighbors=5)
    # # Apply the imputer
    # df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))
    data = df[column]

    # Identify zero values
    mask = data[column] == 0

    # Apply the imputer
    imputer = KNNImputer(n_neighbors=5, missing_values=0)
    data.loc[mask, column] = imputer.fit_transform(data[column].values.reshape(-1, 1))[mask]

    # Update the DataFrame
    df.update(data)

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

    global previous_car_pos
    if frame == 0:
        previous_car_pos = (0, 0)
    car_pos = (previous_car_pos[0] + delta_x, previous_car_pos[1] + delta_y)
    predict_future_positions(car_pos, previous_car_pos)
    previous_car_pos = car_pos

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

    global obj1_prev_pos, obj2_prev_pos, obj3_prev_pos, obj4_prev_pos

    predict_future_positions(obj1_pos, obj1_prev_pos)
    obj1_prev_pos = obj1_pos

    predict_future_positions(obj2_pos, obj2_prev_pos)
    obj2_prev_pos = obj2_pos

    predict_future_positions(obj3_pos, obj3_prev_pos)
    obj3_prev_pos = obj3_pos

    predict_future_positions(obj4_pos, obj4_prev_pos)
    obj4_prev_pos = obj4_pos


    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 150)
    ax.set_aspect('equal')


def predict_future_positions(pos, prev_pos, predict_count = 15, step_size = 5):
    pos_delta = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
    for i in range(predict_count):
        pos_predicted = (pos[0] + pos_delta[0] * i * step_size, pos[1] + pos_delta[1] * i * step_size)
        draw_object(pos_predicted, "b")


def draw_object(position, color):
    obj_rect = patches.Rectangle((position[1] - OBJ_DIMENSIONS[1] / 2,
                                  position[0] - OBJ_DIMENSIONS[0] / 2),
                                 OBJ_DIMENSIONS[1], OBJ_DIMENSIONS[0],
                                 linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(obj_rect)


def on_close(event):
    sys.exit(0)


# df = pd.read_csv('data.csv')
lineCount = df.shape[0]
previous_timestamp = df['Timestamp'].iloc[0]

previous_car_pos = (0, 0)
obj1_prev_pos = (df["FirstObjectDistance_X"].iloc[0] / 128,
                 df["FirstObjectDistance_Y"].iloc[0] / 128)
obj2_prev_pos = (df["SecondObjectDistance_X"].iloc[0] / 128, df["SecondObjectDistance_Y"].iloc[0] / 128)
obj3_prev_pos = (df["ThirdObjectDistance_X"].iloc[0] / 128, df["ThirdObjectDistance_Y"].iloc[0] / 128)
obj4_prev_pos = (df["FourthObjectDistance_X"].iloc[0] / 128, df["FourthObjectDistance_Y"].iloc[0] / 128)


fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=lineCount, interval=2)

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
