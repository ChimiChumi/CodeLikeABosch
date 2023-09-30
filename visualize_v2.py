import math
import sys  # Import the sys module to handle keyboard interrupts

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats
from sklearn.impute import KNNImputer

CAR_DIMENSIONS = (5, 5)
OBJ_DIMENSIONS = (5, 5)


"""def fill_zeros(df, column):
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
"""
class DetectedObject:
    def __init__(self, column_name, car_object, edge_color, face_color='none'):
        self.column_name = column_name  # ex.: "ThirdObject"
        self.car_object = car_object
        self.pos_history = []
        self.edge_color = edge_color  # ex.: "g", "black"
        self.face_color = face_color  # ex.: "g", "black"
        self.pos = (0, 0)
        self.draw_dimensions = OBJ_DIMENSIONS

    def update(self, frame):
        relative_pos = (df[self.column_name + "Distance_X"].iloc[frame] / 128,
                        df[self.column_name + "Distance_Y"].iloc[frame] / 128)
        self.pos = (self.car_object.pos[0] + relative_pos[0],
                    self.car_object.pos[1] + relative_pos[1])
        if frame == 0:
            self.pos_history.clear()
        self.pos_history.append(self.pos)

    def draw(self):
        draw_object(self.pos, self.draw_dimensions, self.edge_color, self.face_color)

    def predict(self, predict_count, step_size, history_to_avg):
        predict_future_positions(self.pos, self.pos_history, history_to_avg, predict_count, step_size)


class Car:
    def __init__(self, edge_color, face_color='none'):
        self.edge_color = edge_color  # ex.: "g", "black"
        self.face_color = face_color  # ex.: "g", "black"
        self.speed = 0
        self.yaw_rate = 0
        self.pos_history = [(0, 0)]
        self.pos = (0, 0)
        self.draw_dimensions = CAR_DIMENSIONS

    def update(self, frame, delta_time):
        self.speed = df["VehicleSpeed"].iloc[frame] / 256  # Convert to m/s
        self.yaw_rate = df["YawRate"].iloc[frame] * (180 / math.pi)

        heading_angle = self.yaw_rate * delta_time
        delta_x = self.speed * math.cos(heading_angle) * delta_time
        delta_y = self.speed * math.sin(heading_angle) * delta_time

        if frame == 0:
            self.pos_history.clear()
            self.pos = (0, 0)
        else:
            self.pos = (self.pos_history[-1][0] + delta_x, self.pos_history[-1][1] + delta_y)
        self.pos_history.append(self.pos)

    def draw(self):
        draw_object(self.pos, self.draw_dimensions, self.edge_color, self.face_color)

    def predict(self, predict_count, step_size, history_to_avg):
        predict_future_positions(self.pos, self.pos_history, history_to_avg, predict_count, step_size)


def update(frame):
    ax.clear()

    global previous_timestamp
    current_timestamp = df['Timestamp'].iloc[frame]
    delta_time = current_timestamp - previous_timestamp
    previous_timestamp = current_timestamp

    car.update(frame, delta_time)
    car.draw()
    car.predict(5, 5, 5)

    for detectedObject in detectedObjects:
        detectedObject.update(frame)
        detectedObject.draw()
        detectedObject.predict(5, 5, 5)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 100)
    ax.set_aspect('equal')


def predict_future_positions(pos, pos_history, history_to_avg, predict_count, step_size):
    avg_pos_delta = (0, 0)
    if len(pos_history) > 2:
        pos_deltas = []
        for i in range(min(history_to_avg, len(pos_history)-1)):
            pos_deltas.append([pos_history[-1-i][0] - pos_history[-2-i][0],
                              pos_history[-1-i][1] - pos_history[-2-i][1]])
        avg_x = sum(x for x, y in pos_deltas) / len(pos_deltas)
        avg_y = sum(y for x, y in pos_deltas) / len(pos_deltas)
        avg_pos_delta = (avg_x, avg_y)
    for i in range(predict_count):
        pos_predicted = (pos[0] + avg_pos_delta[0] * i * step_size, pos[1] + avg_pos_delta[1] * i * step_size)
        draw_object(pos_predicted, OBJ_DIMENSIONS, "b")


def draw_object(position, draw_dimensions, edge_color, face_color='none'):
    obj_rect = patches.Rectangle((position[1] - draw_dimensions[1] / 2,
                                  position[0] - draw_dimensions[0] / 2),
                                 draw_dimensions[1], draw_dimensions[0],
                                 linewidth=1, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(obj_rect)


def on_close(event):
    sys.exit(0)


df = pd.read_csv('data.csv')
lineCount = df.shape[0]
previous_timestamp = df['Timestamp'].iloc[0]

car = Car('black', 'red')
detectedObjects = [
    DetectedObject("FirstObject", car, "green", "green"),
    DetectedObject("SecondObject", car, "yellow", "yellow"),
    DetectedObject("ThirdObject", car, "blue", "blue"),
    DetectedObject("FourthObject", car, "magenta", "magenta")
]

fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=lineCount, interval=2)

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
