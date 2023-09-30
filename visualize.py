import math
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

CAR_DIMENSIONS = (5, 5)  # size of the car model
OBJ_DIMENSIONS = (3, 3)  # size of the tracked object model
ACCEPTABLE_DIFFERENCE_FROM_PREDICTION = 5  # threshold for noisy object tracking


class CommonObject:
    """Common base class for the car and tracked objects"""
    def __init__(self, edge_color, prediction_color="blue", face_color='none'):
        # visualization colors
        self.edge_color = edge_color
        self.face_color = face_color
        self.prediction_color = prediction_color

        # size of the object
        self.draw_dimensions = OBJ_DIMENSIONS

        # current position
        self.pos = (0, 0)

        # list of previously visited positions
        self.pos_history = []

        # list of predicted future positions
        self.predicted_pos_list = []

        # ignored object are not displayed and not checked for collisions
        # objects are ignored if they make impossible movements (detection noise)
        self.ignore_object = False

    def draw(self):
        """displays the object in its current position"""
        if self.ignore_object:
            return
        draw_object(self.pos, self.draw_dimensions, self.edge_color, self.face_color)

    def predict(self, predict_count, step_size, history_to_avg):
        """predicts the next positions of the object based on the average of previous position changes"""
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
        for i in range(1, predict_count + 1):
            pos_predicted = (self.pos[0] + avg_pos_delta[0] * i * step_size,
                             self.pos[1] + avg_pos_delta[1] * i * step_size)
            self.predicted_pos_list.append(pos_predicted)

    def draw_predicted(self):
        """displays the predicted future positions of the object"""
        if self.ignore_object:
            return
        for pos_predicted in self.predicted_pos_list:
            draw_object(pos_predicted, self.draw_dimensions, self.prediction_color)


class DetectedObject(CommonObject):
    """represents an object tracked by the car"""
    def __init__(self, column_name, car_object, edge_color, prediction_color="blue", face_color='none'):
        super().__init__(edge_color, prediction_color, face_color)

        # name of the object on the .csv file (ex.:"FirstObject")
        self.column_name = column_name

        # reference to the car that is tracking this object
        self.car_object = car_object

        # size of the object
        self.draw_dimensions = OBJ_DIMENSIONS

    def update(self, frame):
        """reads the position of the object from the CSV
        updates its position history
        checks if the object should be ignored or not"""
        relative_pos = (df[self.column_name + "Distance_X"].iloc[frame] / 128,
                        df[self.column_name + "Distance_Y"].iloc[frame] / 128)
        self.pos = (self.car_object.pos[0] + relative_pos[0],
                    self.car_object.pos[1] + relative_pos[1])
        if frame == 0:
            self.pos_history.clear()
        self.pos_history.append(self.pos)
        self.__update_ignored_status()

    def __update_ignored_status(self):
        """checks of the object should be ignored"""
        # object is ignored if there are big changes in its movement
        is_far_from_predicted = False
        if len(self.predicted_pos_list) > 0:
            is_far_from_predicted = (
                        abs(self.pos[0] - self.predicted_pos_list[0][0]) > ACCEPTABLE_DIFFERENCE_FROM_PREDICTION
                        or abs(self.pos[1] - self.predicted_pos_list[0][1]) > ACCEPTABLE_DIFFERENCE_FROM_PREDICTION)

        # object is ignored if it is "inside" the car (ex.: the 0 values in the CSV)
        is_too_close_to_car = (abs(self.pos[0] - self.car_object.pos[0]) < car.draw_dimensions[0] / 2
                               and abs(self.pos[1] - self.car_object.pos[1]) < car.draw_dimensions[1] / 2)

        if is_too_close_to_car or is_far_from_predicted:
            self.ignore_object = True
        else:
            self.ignore_object = False


class Car(CommonObject):
    """Represents the car"""
    def __init__(self, edge_color, prediction_color, face_color='none'):
        super().__init__(edge_color, prediction_color, face_color)
        self.speed = 0
        self.yaw_rate = 0
        self.pos_history = [(0, 0)]
        self.draw_dimensions = CAR_DIMENSIONS

    def update(self, frame, delta_time):
        """calculates the position from the values in the CSV
        updates its position history"""
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
    """function that runs for every row in the CSV
    updates the system"""

    ax.clear()

    global previous_timestamp
    current_timestamp = df['Timestamp'].iloc[frame]
    delta_time = current_timestamp - previous_timestamp
    previous_timestamp = current_timestamp

    car.update(frame, delta_time)
    car.draw()
    car.predict(5, 5, 5)
    car.draw_predicted()

    cpnco_detected = False
    cpta_detected = False
    cpla_detected = False

    for detectedObject in detected_objects:
        detectedObject.update(frame)
        detectedObject.draw()
        detectedObject.predict(5, 5, 5)
        detectedObject.draw_predicted()

    collision_predicted = False
    collision_object = None
    predicted_collision_index = 0
    for i in range(len(car.predicted_pos_list)):
        for detectedObject in detected_objects:
            if detectedObject.ignore_object:
                continue
            if is_overlapping(car.predicted_pos_list[i], car.draw_dimensions,
                              detectedObject.predicted_pos_list[i], detectedObject.draw_dimensions):
                collision_predicted = True
                collision_object = detectedObject
                predicted_collision_index = i
    if collision_predicted:
        car.prediction_color = "red"
    else:
        car.prediction_color = "green"
    if collision_predicted:
        if check_cpnco(car, collision_object, adjacent_cars, predicted_collision_index):
            cpnco_detected = True
            # Check for CPTA
        if check_cpta(car, collision_object, predicted_collision_index):
            cpta_detected = True

            # Check for CPLA
        if check_cpla(car, collision_object, predicted_collision_index):
            cpla_detected = True

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 100)
    ax.set_aspect('equal')
    plt.gca().invert_xaxis()

    if collision_predicted:
        print("Collision Predicted!!!")
        if cpnco_detected:
            print("CPNCO - Car to Pedestrian Nearside Child Obstructed Detected!")

        if cpta_detected:
            print("CPTA - Car to Pedestrian Turn Adult Detected!")

        if cpla_detected:
            print("CPLA - Car to Pedestrian Longitudinal Adult Detected!")
        plt.pause(20000)
        plt.close()


# Checking if the scenario is CPNCO and returning boolean
def check_cpnco(car, pedestrian, adjacent_cars, predicted_collision_index):
    # If the car is moving in straight line we proceed
    if abs(car.yaw_rate) < 2:
        for adjacent_car in adjacent_cars:
            if abs(car.predicted_pos_list[predicted_collision_index][1] - adjacent_car.pos[1]) < 5:
                if (is_overlapping(car.predicted_pos_list[predicted_collision_index],
                                   car.draw_dimensions,
                                   pedestrian.predicted_pos_list[predicted_collision_index],
                                   pedestrian.draw_dimensions)):
                    # Check if the pedestrian's movement is perpendicular to the car's movement
                    car_direction_vector = (car.predicted_pos_list[0][0] - car.pos[0],
                                            car.predicted_pos_list[0][1] - car.pos[1])
                    pedestrian_vector = (pedestrian.pos[0] - car.predicted_pos_list[predicted_collision_index][0],
                                         pedestrian.pos[1] - car.predicted_pos_list[predicted_collision_index][1],)
                    dot_product = car_direction_vector[0] * pedestrian_vector[0] + car_direction_vector[1] * pedestrian_vector[1]
                    if abs(dot_product) < 1e-3:  # Check if the dot product is close to zero
                        return True
    return False


# Checking for CPTA Scenario
def check_cpta(car, pedestrian, predicted_collision_index):
    # Proceeding if we have high speed and turning
    if abs(car.yaw_rate) > 8 and car.speed > 4:
        if is_overlapping(car.predicted_pos_list[predicted_collision_index],
                          car.draw_dimensions,
                          pedestrian.predicted_pos_list[predicted_collision_index],
                          pedestrian.draw_dimensions):
            return True
    return False


# Checking if the scenario is CPLA
def check_cpla(car, pedestrian, predicted_collision_index):
    # Proceeding if the car is not turning and has high speed
    if abs(car.yaw_rate) < 2 and car.speed > 4:
        if is_overlapping(car.predicted_pos_list[predicted_collision_index],
                          car.draw_dimensions,
                          pedestrian.predicted_pos_list[predicted_collision_index],
                          pedestrian.draw_dimensions):
            return True
    return False


def draw_object(position, draw_dimensions, edge_color, face_color='none'):
    obj_rect = patches.Rectangle((position[1] - draw_dimensions[1] / 2,
                                  position[0] - draw_dimensions[0] / 2),
                                 draw_dimensions[1], draw_dimensions[0],
                                 linewidth=1, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(obj_rect)


def is_overlapping(obj1_pos, obj1_dimensions, obj2_pos, obj2_dimensions):
    """checks if two objects with given positions and sizes overlap"""
    left_1 = obj1_pos[1] - obj1_dimensions[1] / 2
    right_1 = obj1_pos[1] + obj1_dimensions[1] / 2
    top_1 = obj1_pos[0] + obj1_dimensions[0] / 2
    bottom_1 = obj1_pos[0] - obj1_dimensions[0] / 2

    left_2 = obj2_pos[1] - obj2_dimensions[1] / 2
    right_2 = obj2_pos[1] + obj2_dimensions[1] / 2
    top_2 = obj2_pos[0] + obj2_dimensions[0] / 2
    bottom_2 = obj2_pos[0] - obj2_dimensions[0] / 2

    return (left_1 < right_2 and right_1 > left_2 and
            bottom_1 < top_2 and top_1 > bottom_2)


def on_close(event):
    sys.exit(0)


if __name__ == "__main__":
    # initialize file reading
    df = pd.read_csv('data.csv')
    line_count = df.shape[0]

    # initialize first timestamp
    previous_timestamp = df['Timestamp'].iloc[0]

    # initialize the car and the objects tracked by it
    car = Car('black', "green", 'red')
    detected_objects = [
        DetectedObject("FirstObject", car, "green", "green", "green"),
        DetectedObject("SecondObject", car, "purple", "purple", "purple"),
        DetectedObject("ThirdObject", car, "blue", "blue", "blue"),
        DetectedObject("FourthObject", car, "magenta", "magenta", "magenta")
    ]

    # list of possible scenarios the system detects
    possible_scenarios = ["CPNCO", "CPTA", "CPLA"]

    adjacent_cars = [Car('blue', 'blue'), Car('blue', 'blue')]

    # initialize the visualization plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(fig, update, frames=line_count, interval=2)

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()
