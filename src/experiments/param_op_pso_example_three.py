from src.classes import Swarm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Parameters of PSO that can be changed to study effects

NUM_ITERATIONS = 50
VERBOSE = False
costs_min_bound = 0.01
costs_max_bound = 100
gamma_min_bound = 0.01
gamma_max_bound = 100
test_size_min_bound = 0.1
test_size_max_bound = 0.3

MAX_BOUNDS = [costs_max_bound, gamma_max_bound, test_size_max_bound]
MIN_BOUNDS = [costs_min_bound, gamma_min_bound, test_size_min_bound]

w_vals = [0.2, 0.3, 0.4]
c1_vals = [0.1, 0.2, 0.3]
c2_vals = [0.5, 0.75, 1]
particle_vals = [20, 30, 40]

results = []
for w in w_vals:
    for c1 in c1_vals:
        for c2 in c2_vals:
            for num_particles in particle_vals:
                swarm = Swarm(w=w, c1=c1, c2=c2, fn=None, max_bounds=MAX_BOUNDS, min_bounds=MIN_BOUNDS,
                              iters=NUM_ITERATIONS,
                              swarm_size=num_particles, dimensions=3, discrete=False, data_loader=None, is_model=True,
                              minimize=False)
                swarm.run(verbose=VERBOSE)
                result_obj = {
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'swarm_size': num_particles,
                    'runtime (s)': swarm.execution_time,
                    'RMSE': abs(100 - swarm.global_value_best),
                    'model': swarm
                }
                results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(81)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_three.csv')
print(results.head())

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, facecolor='w')
results = json.loads(results.head().to_json(orient='records'))
for i, result in enumerate(results):
    ax.plot(np.linspace(0, NUM_ITERATIONS + 1, NUM_ITERATIONS), result['model']['val_history'], label=f"Model #{i + 1}")
ax.set_title('Model Convergence Vs. PSO Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Minimum Value')
ax.legend()
fig.savefig('results/experiment_three.png')
