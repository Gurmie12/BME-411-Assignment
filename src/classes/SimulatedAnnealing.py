import numpy as np
import time
from src.classes import DataLoader, get_closest_discrete_point, Model


class SimulatedAnnealing:

    def __init__(self, max_bounds, min_bounds, fn, dimensions=2, discrete=False, iterations=10000, step_size=1, temp=10,
                 minimize=True, data_loader: DataLoader = None, is_model=False):
        self.max_bounds = max_bounds
        self.min_bounds = min_bounds
        self.dimensions = dimensions
        self.discrete = discrete
        self.iterations = iterations
        self.step_size = step_size
        self.temp = temp
        self.fn = fn
        self.minimize = minimize
        self.data_loader = data_loader
        self.is_model = is_model

        # Generate initial point
        pos_0 = []
        for d in range(dimensions):
            if discrete:
                if data_loader is None:
                    raise Exception("Must provide a data loader object if using SA in discrete mode.")
                else:
                    pos_0.append(np.random.choice(data_loader.ind_data[d], size=1)[0])
            else:
                if not is_model:
                    if d < 2:
                        values = np.logspace(start=self.min_bounds[d], stop=self.max_bounds[d], num=50)
                        pos_0.append(np.random.choice(values, size=1)[0])
                    else:
                        pos_0.append(np.random.uniform(low=self.min_bounds[d], high=self.max_bounds[d], size=1)[0])
                else:
                    pos_0.append(np.random.uniform(low=self.min_bounds[d], high=self.max_bounds[d], size=1)[0])
        if discrete:
            while not data_loader.df.loc[(data_loader.df[data_loader.independent_column_names[0]] == pos_0[0]) & (
                    data_loader.df[data_loader.independent_column_names[1]] == pos_0[1])].any().all():
                for d in range(dimensions):
                    pos_0[d] = np.random.choice(data_loader.ind_data[d], size=1)[0]

        self.best_point = pos_0
        if discrete:
            condition = (data_loader.df[data_loader.independent_column_names[0]] == self.best_point[0])
            for i in range(1, dimensions):
                condition = condition & (
                        data_loader.df[data_loader.independent_column_names[i]] == self.best_point[i])
            self.best_val = data_loader.df.loc[condition, data_loader.dependent_column_name].iloc[0]
        else:
            if not is_model:
                self.best_val = fn(self.best_point[0], self.best_point[1])
            else:
                model = Model(self.best_point[0], self.best_point[1], self.best_point[2])
                self.best_val = model.get_model_accuracy()
        self.cur_point = self.best_point
        self.cur_val = self.best_val
        self.execution_time = None
        self.val_history = []

    def run(self, verbose=False):
        start_time = time.time()
        for i in range(self.iterations):
            if verbose:
                print(f"Iteration {i + 1} --------------")
                print(f"Best Value: {self.best_point}")
                print(f"Best Position: {self.best_val}")

            new_point = []
            for d in range(self.dimensions):
                new_point.append(self.cur_point[d] + np.random.randn(1)[0] * self.step_size)

            # Check if new_point is in bounds
            if self.discrete:
                for d in range(self.dimensions):
                    if new_point[d] < min(self.data_loader.ind_data[d]):
                        new_point[d] = min(self.data_loader.ind_data[d])
                    if new_point[d] > max(self.data_loader.ind_data[d]):
                        new_point[d] = max(self.data_loader.ind_data[d])
            else:
                for d in range(self.dimensions):
                    if new_point[d] < self.min_bounds[d]:
                        new_point[d] = self.min_bounds[d]
                    if new_point[d] > self.max_bounds[d]:
                        new_point[d] = self.max_bounds[d]

            if self.discrete:
                new_point = get_closest_discrete_point(new_point, self.data_loader)

            if self.discrete:
                condition = (self.data_loader.df[self.data_loader.independent_column_names[0]] == new_point[0])
                for i in range(1, self.dimensions):
                    condition = condition & (
                            self.data_loader.df[self.data_loader.independent_column_names[i]] == new_point[i])
                new_val = self.data_loader.df.loc[condition, self.data_loader.dependent_column_name].iloc[0]
            else:
                if not self.is_model:
                    new_val = self.fn(new_point[0], new_point[1])
                else:
                    model = Model(new_point[0], new_point[1], new_point[2])
                    new_val = model.get_model_accuracy()

            if self.minimize:
                if new_val < self.best_val:
                    self.best_point = new_point
                    self.best_val = new_val
            else:
                if new_val > self.best_val:
                    self.best_point = new_point
                    self.best_val = new_val

            delta = new_val - self.cur_val
            t = self.temp / float(i + 1)
            metropolis = np.exp(-delta / t)
            if delta < 0 or np.random.rand() < metropolis:
                self.cur_point = new_point
                self.cur_val = new_val
            self.val_history.append(self.best_val)
        end_time = time.time()
        self.execution_time = round(end_time - start_time, 2)
        print(f"Final Results After {self.iterations} Iterations: -------------------")
        print(f"Iterations: {self.iterations}")
        print(f"Step Size: {self.step_size}")
        print(f"Temp: {self.temp}")
        print(f"Best Position: {self.best_point}")
        print(f"Best Value: {self.best_val}")
        print(f"Completed run in: {self.execution_time} seconds")
        print(f"----------------------------------------------------------------")
