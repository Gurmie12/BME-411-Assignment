import numpy as np
import time
from src.classes.Particle import Particle
from src.classes import DataLoader


class Swarm:

    def __init__(self, w, c1, c2, fn, max_bounds, min_bounds, iters=50, swarm_size=20, dimensions=2, discrete=False,
                 data_loader: DataLoader = None, is_model=False, minimize=True):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fn = fn
        self.global_position_best = None
        self.global_value_best = None

        # [x_max, y_max]
        self.max_bounds = max_bounds

        # [x_min, y_min]
        self.min_bounds = min_bounds

        self.iters = iters
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.discrete = discrete
        self.execution_time = None

        if discrete and data_loader is None:
            raise Exception(
                "When using swarm in discrete mode, must provide data_loader containing data.")
        self.data_loader = data_loader

        self.is_model = is_model

        # Initialize all your particles of your swarm
        self.particles = []
        for i in range(swarm_size):
            pos_0 = []
            for d in range(dimensions):
                if not discrete:
                    if self.is_model:
                        pos_0.append(np.random.uniform(low=self.min_bounds[d], high=self.max_bounds[d], size=1)[0])
                    else:
                        pos_0.append(np.random.randint(low=self.min_bounds[d], high=self.max_bounds[d] + 1, size=1)[0])
                else:
                    pos_0.append(np.random.choice(data_loader.ind_data[d], size=1)[0])

            if discrete:
                while not data_loader.df.loc[(data_loader.df[data_loader.independent_column_names[0]] == pos_0[0]) & (
                        data_loader.df[data_loader.independent_column_names[1]] == pos_0[1])].any().all():
                    for d in range(dimensions):
                        pos_0[d] = np.random.choice(data_loader.ind_data[d], size=1)[0]
            self.particles.append(Particle(pos_0=pos_0, discrete=discrete))

            self.val_history = []
            self.best_pos_history = []

            self.minimize = minimize

    def run(self, verbose=False):
        start_time = time.time()
        cur_iteration = 0
        while cur_iteration < self.iters:
            if verbose:
                print(f"Iteration {cur_iteration + 1} --------------")
                print(f"Best Value: {self.global_value_best}")
                print(f"Best Position: {self.global_position_best}")

            # Check if particles are better than global best
            for i in range(self.swarm_size):
                if not self.discrete:
                    self.particles[i].evaluate(fn=self.fn, is_model=self.is_model, minimize=self.minimize)
                else:
                    self.particles[i].evaluate(data_loader=self.data_loader, minimize=self.minimize)

                if self.minimize:
                    if not self.global_value_best or self.particles[i].value < self.global_value_best:
                        self.global_position_best = self.particles[i].position
                        self.global_value_best = self.particles[i].value
                else:
                    if not self.global_value_best or self.particles[i].value > self.global_value_best:
                        self.global_position_best = self.particles[i].position
                        self.global_value_best = self.particles[i].value

            # Update positions and velocities of each particle
            for i in range(self.swarm_size):
                self.particles[i].update_velocity(self.global_position_best, self.w, self.c1, self.c2)
                if self.discrete:
                    self.particles[i].update_position(self.min_bounds, self.max_bounds, data_loader=self.data_loader,
                                                      discrete=self.discrete,
                                                      is_model=self.is_model)
                else:
                    self.particles[i].update_position(self.min_bounds, self.max_bounds, discrete=self.discrete)

            # Update the global value and position history
            self.val_history.append(self.global_value_best)
            self.best_pos_history.append(self.global_position_best)
            cur_iteration += 1
        end_time = time.time()
        self.execution_time = round(end_time - start_time, 2)
        print(f"Final Results After {self.iters} Iterations: -------------------")
        print(f"Swarm Size: {self.swarm_size}")
        print(f"Best Position: {self.global_position_best}")
        print(f"Best Value: {self.global_value_best}")
        print(f"Completed run in: {self.execution_time} seconds")
        print(f"----------------------------------------------------------------")
