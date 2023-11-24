from src.classes import Swarm
from src.utils import ackley_fun
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


# Parameters of PSO that can be changed to study effects
MIN_X = -32
MAX_X = 32
MIN_Y = -32
MAX_Y = 32
NUM_ITERATIONS = 50
COLOR_MAP_GRANULARITY = 200
TEST_FN = ackley_fun
DIMENSIONS = 2
DISCRETE = False
VERBOSE = False

w_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
c1_vals = [0.1, 1.0, 2.0, 3.0, 4.0]
c2_vals = [0.1, 1.0, 2.0, 3.0, 4.0]
particle_vals = [20, 40, 60, 80, 100]

results = []
for w in w_vals:
    for c1 in c1_vals:
        for c2 in c2_vals:
            for num_particles in particle_vals:
                swarm = Swarm(w=w, c1=c1, c2=c2, fn=TEST_FN, max_bounds=[MAX_X, MAX_Y], min_bounds=[MIN_X, MIN_Y],
                                 iters=NUM_ITERATIONS, swarm_size=num_particles, dimensions=DIMENSIONS,
                                 data_loader=None, is_model=False, minimize=True)
                swarm.run(verbose=VERBOSE)
                result_obj = {
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'swarm_size': num_particles,
                    'runtime (s)': swarm.execution_time,
                    'RMSE': abs(0-swarm.global_value_best),
                    'model': swarm
                }
                results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(5*5*5*5)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_one_ackley_func.csv')
print(results.head())

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, facecolor='w')
results = json.loads(results.head().to_json(orient='records'))
for i, result in enumerate(results):
    ax.plot(np.linspace(0, NUM_ITERATIONS + 1, NUM_ITERATIONS), result['model']['val_history'], label=f"Model #{i+1}")
ax.set_title('Model Convergence Vs. PSO Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Minimum Value')
ax.legend()
fig.savefig('results/experiment_one_ackley_func.png')