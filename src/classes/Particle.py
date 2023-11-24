import numpy as np
from src.classes.DataLoader import DataLoader
from src.classes.Model import Model
import math


def get_distance_between_points(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)


def get_closest_discrete_point(new_point, data_loader: DataLoader = None):
    # Round data point
    for i in range(len(new_point)):
        new_point[i] = math.ceil(new_point[i])

    if data_loader.df.loc[(data_loader.df[data_loader.independent_column_names[0]] == new_point[0]) & (
            data_loader.df[data_loader.independent_column_names[1]] == new_point[1])].any().all():
        return new_point
    else:
        closest_point = None
        closest_distance = None
        for i, row in data_loader.df.iterrows():
            cur_point = row.drop(data_loader.dependent_column_name).to_numpy()
            cur_distance = get_distance_between_points(new_point, cur_point)
            if (closest_point is None and closest_distance is None) or cur_distance < closest_distance:
                closest_point = cur_point
                closest_distance = cur_distance
        return closest_point


class Particle:

    # Initialize a single particle
    def __init__(self, pos_0, discrete=False):
        self.position = np.array(pos_0)
        self.velocity = np.random.randint(low=-1, high=1, size=len(pos_0))
        self.value = None
        self.value_best = None
        self.position_best = self.position
        self.history = []
        self.discrete = discrete

    # Evaluate the objective function at current value and update personal best position and value
    def evaluate(self, fn=None, data_loader: DataLoader = None, is_model=False, minimize=True):
        if not self.discrete:
            if is_model:
                model = Model(cost=self.position[0], gamma=self.position[1], test_size=self.position[2])
                self.value = model.get_model_accuracy()
            else:
                self.value = fn(self.position[0], self.position[1])
        else:
            if data_loader:
                condition = (data_loader.df[data_loader.independent_column_names[0]] == self.position[0])
                for i in range(1, len(data_loader.independent_column_names)):
                    condition = condition & (
                            data_loader.df[data_loader.independent_column_names[i]] == self.position[i])
                self.value = data_loader.df.loc[condition, data_loader.dependent_column_name].iloc[0]

        if minimize:
            if not self.value_best or self.value < self.value_best:
                self.value_best = self.value
                self.position_best = self.position
        else:
            if not self.value_best or self.value > self.value_best:
                self.value_best = self.value
                self.position_best = self.position

    # Update the particles position
    def update_position(self, min_bounds, max_bounds, discrete, is_model=False, data_loader: DataLoader = None):
        self.position = self.position + self.velocity
        if discrete:
            for d in range(len(data_loader.independent_column_names)):
                if self.position[d] < min(data_loader.ind_data[d]):
                    self.position[d] = min(data_loader.ind_data[d])
                if self.position[d] > max(data_loader.ind_data[d]):
                    self.position[d] = max(data_loader.ind_data[d])
        else:
            for d in range(len(min_bounds)):
                if self.position[d] < min_bounds[d]:
                    self.position[d] = min_bounds[d]
                if self.position[d] > max_bounds[d]:
                    self.position[d] = max_bounds[d]

        # if the data is discrete, match the new point to the closest discrete data
        if self.discrete and not is_model:
            self.position = get_closest_discrete_point(self.position, data_loader)

        # Add position to history
        self.history.append(self.position)

    # Update particles velocity
    def update_velocity(self, global_position_best, w, c1, c2):
        r1, r2 = np.random.random(), np.random.random()

        cognitive_velocity = c1 * r1 * (self.position_best - self.position)
        social_velocity = c2 * r2 * (global_position_best - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
