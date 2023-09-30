import math
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats

CAR_DIMENSIONS = (5, 5)
OBJ_DIMENSIONS = (5, 5)


def fill_zeros(df, column):
    zero_indices = df[df[column] == 0].index
    for idx in zero_indices:
        if idx == 0 or idx == len(df) - 1:
            continue
        else:
            prev_idx = idx - 1
            next_idx = idx + 1
            while df.loc[next_idx, column] == 0:
                next_idx += 1
                if next_idx >= len(df):
                    break
            if next_idx >= len(df):
                continue
            else:
                diff = df.loc[next_idx, column] - df.loc[prev_idx, column]
                increment = diff / (next_idx - prev_idx)
                interpolated_values = [df.loc[prev_idx, column] + increment * i for i in range(1, next_idx - prev_idx)]
                df.loc[idx:next_idx - 1, column] = interpolated_values
    return df


class CommonObject:
    def __init__(self, edge_color, face_color='none'):
        self.edge_color = edge_color
        self.face_color = face_color
        self.pos = (0, 0)
        self.pos_history = []
        self.draw_dimensions = OBJ_DIMENSIONS
        self.predicted_pos_list = []

    def draw(self):
        draw_object(self.pos, self.draw_dimensions, self.edge_color, self.face_color)

    def predict(self, predict_count, step_size, history_to_avg):
        avg_pos_delta = (0, 0)
        if len(self.pos_history) > 2:
            pos_deltas = []
            for i in range(min(history_to_avg, len(self.pos_history) - 1)):
                pos_deltas.append([self.pos_history[-1 - i][0] - self.pos_history[-2 - i][0],
                                   self.pos_history[-1 - i][1] - self.pos_history[-2 - i][1]])
            avg_x = sum(x for x, y in pos_deltas) / len(pos_deltas)
            avg_y = sum(y for x, y in pos_deltas) / len(pos_deltas)
            avg_pos_delta = (avg_x, avg_y)
        self.predicted_pos_list.clear()
        for i in range(predict_count):
            pos_predicted = (self.pos[0] + avg_pos_delta[0] * i * step_size,
                             self.pos[1] + avg_pos_delta[1] * i * step_size)
            self.predicted_pos_list.append(pos_predicted)

    def draw_predicted(self, edge_color="b"):
        for pos_predicted in self.predicted_pos_list:
            draw_object(pos_predicted, self.draw_dimensions, edge_color)


class DetectedObject(CommonObject):
    def __init__(self, column_name, car_object, edge_color, face_color='none'):
        super().__init__(edge_color, face_color)
        self.column_name = column_name
        self.car_object = car_object
        self.draw_dimensions = OBJ_DIMENSIONS

    def update(self, frame):
        relative_pos = (df[self.column_name + "Distance_X"].iloc[frame] / 128,
                        df[self.column_name + "Distance_Y"].iloc[frame] / 128)
        self.pos = (self.car_object.pos[0] + relative_pos[0],
                    self.car_object.pos[1] + relative_pos[1])
        if frame == 0:
            self.pos_history.clear()
        self.pos_history.append(self.pos)


class Car(CommonObject):
    def __init__(self, edge_color, face_color='none'):
        super().__init__(edge_color, face_color)
        self.speed = 0
        self.yaw_rate = 0
        self.pos_history = [(0, 0)]
        self.draw_dimensions = CAR_DIMENSIONS

    def update(self, frame, delta_time):
        self.speed = df["VehicleSpeed"].iloc[frame] / 256
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


def update(frame):
    ax.clear()

    global previous_timestamp
    current_timestamp = df['Timestamp'].iloc[frame]
    delta_time = current_timestamp - previous_timestamp
    previous_timestamp = current_timestamp

    car.update(frame, delta_time)
    car.draw()
    car.predict(5, 5, 5)
    car.draw_predicted()

    collision_detected = False

    cpnco_detected = False
    cpta_detected = False
    cpla_detected = False

    for detectedObject in detected_objects:
        detectedObject.update(frame)
        detectedObject.draw()
        detectedObject.predict(5, 5, 5)
        detectedObject.draw_predicted()

        if check_collision(car, detectedObject):
            collision_detected = True
            if check_cpnco(car, detectedObject, adjacent_cars):
                cpnco_detected = True

                # Check for CPTA
            if check_cpta(car, detectedObject):
                cpta_detected = True

                # Check for CPLA
            if check_cpla(car, detectedObject):
                cpla_detected = True
            break

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 100)
    ax.set_aspect('equal')

    if collision_detected:
        print("Collision Detected!!!")
        if cpnco_detected:
            print("CPNCO - Car to Pedestrian Nearside Child Obstructed Detected!")

        if cpta_detected:
            print("CPTA - Car to Pedestrian Turn Adult Detected!")

        if cpla_detected:
            print("CPLA - Car to Pedestrian Longitudinal Adult Detected!")
        plt.pause(20000)
        plt.close()


def check_cpnco(car, pedestrian, adjacent_cars):
    # Check if the car is moving in a straight line (you can adjust the threshold)
    if abs(car.yaw_rate) < 10:
        # Check if there are adjacent parallel cars in the last 5 meters (you can adjust the distance)
        for adjacent_car in adjacent_cars:
            if abs(car.pos[1] - adjacent_car.pos[1]) < 5:
                # Check if the pedestrian is crossing in front of the car and is obstructed
                if (car.pos[0] - (car.draw_dimensions[0] / 2)
                        < pedestrian.pos[0] < car.pos[0] + (car.draw_dimensions[0] / 2) and
                        abs(pedestrian.pos[1] - car.pos[1]) < car.draw_dimensions[1] / 2):
                    # Check if the pedestrian's movement is perpendicular to the car's movement
                    car_direction_vector = (1, 0)  # Assuming the car moves in the x-direction
                    pedestrian_vector = (pedestrian.pos[0] - car.pos[0], pedestrian.pos[1] - car.pos[1])
                    dot_product = car_direction_vector[0] * pedestrian_vector[0] + \
                                  car_direction_vector[1] * pedestrian_vector[1]
                    if abs(dot_product) < 1e-3:  # Check if the dot product is close to zero
                        return True
    return False


def check_cpta(car, pedestrian):
    # Check if the car's yaw rate changes significantly over a long timestamp (you can adjust the thresholds)
    if abs(car.yaw_rate) > 8 and car.speed > 4:
        # Check if the pedestrian appears in front of the car and is perpendicular to its movement
        if (car.pos[0] - (car.draw_dimensions[0] / 2)
                < pedestrian.pos[0] < car.pos[0] + (car.draw_dimensions[0] / 2) and
                abs(pedestrian.pos[1] - car.pos[1]) < car.draw_dimensions[1] / 2):
            return True
    return False


def check_cpla(car, pedestrian):
    # Check if the car and pedestrian are moving roughly in the same direction
    if abs(car.yaw_rate) < 10 and car.speed > 5:
        # Check if the car and pedestrian positions overlap or are very close (adjust threshold)
        if abs(car.pos[1] - pedestrian.pos[1]) < car.draw_dimensions[1] / 2:
            return True
    return False


def draw_object(position, draw_dimensions, edge_color, face_color='none'):
    obj_rect = patches.Rectangle((position[1] - draw_dimensions[1] / 2,
                                  position[0] - draw_dimensions[0] / 2),
                                 draw_dimensions[1], draw_dimensions[0],
                                 linewidth=1, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(obj_rect)


def check_collision(car, detectedObject):
    car_x, car_y = car.pos
    object_x, object_y = detectedObject.pos
    car_width, car_height = CAR_DIMENSIONS
    object_width, object_height = OBJ_DIMENSIONS

    if car_x < 1 or car_y < 1 or object_x < 1 or object_y < 1:
        return False
    if (car_x + car_width / 2 < object_x - object_width / 2 or
            car_x - car_width / 2 > object_x + object_width / 2 or
            car_y + car_height / 2 < object_y - object_height / 2 or
            car_y - car_height / 2 > object_y + object_height / 2):
        return False

    return True


def on_close(event):
    sys.exit(0)


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    line_count = df.shape[0]

    previous_timestamp = df['Timestamp'].iloc[0]

    car = Car('black', 'red')
    detected_objects = [
        DetectedObject("FirstObject", car, "green", "green"),
        DetectedObject("SecondObject", car, "yellow", "yellow"),
        DetectedObject("ThirdObject", car, "blue", "blue"),
        DetectedObject("FourthObject", car, "magenta", "magenta")
    ]

    possible_scenarios = ["CPNCO", "CPTA", "CPLA"]
    adjacent_cars = [Car('blue', 'blue'), Car('blue', 'blue')]

    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(fig, update, frames=line_count, interval=2)

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()
