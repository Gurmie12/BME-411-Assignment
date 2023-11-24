from src.classes import Swarm, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


# Parameters of PSO that can be changed to study effects
NUM_ITERATIONS = 50
COLOR_MAP_GRANULARITY = 200
DIMENSIONS = 2
DISCRETE = True
VERBOSE = False

# RUN SWARM
data_loader = DataLoader(path_to_file='data/rainfall_data.csv',
                         independent_column_names=['x', 'y'],
                         dependent_column_name='z')

w_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
c1_vals = [0.1, 1.0, 2.0, 3.0, 4.0]
c2_vals = [0.1, 1.0, 2.0, 3.0, 4.0]
particle_vals = [20, 40, 60, 80, 100]

results = []
for w in w_vals:
    for c1 in c1_vals:
        for c2 in c2_vals:
            for num_particles in particle_vals:
                swarm = Swarm(w=w, c1=c1, c2=c2, fn=None, max_bounds=None, min_bounds=None,
                          iters=NUM_ITERATIONS, swarm_size=num_particles, dimensions=DIMENSIONS, discrete=DISCRETE,
                          data_loader=data_loader, is_model=False, minimize=True)
                swarm.run(verbose=VERBOSE)
                result_obj = {
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'swarm_size': num_particles,
                    'runtime (s)': swarm.execution_time,
                    'RMSE': abs(62.2-swarm.global_value_best),
                    'model': swarm
                }
                results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(5*5*5*5)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_two.csv')
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
fig.savefig('results/experiment_two.png')