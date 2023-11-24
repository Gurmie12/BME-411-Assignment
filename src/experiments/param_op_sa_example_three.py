from src.classes import SimulatedAnnealing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Parameters of PSO that can be changed to study effects

costs_min_bound = 0.01
costs_max_bound = 100
gamma_min_bound = 0.01
gamma_max_bound = 100
test_size_min_bound = 0.1
test_size_max_bound = 0.3

MAX_BOUNDS = [costs_max_bound, gamma_max_bound, test_size_max_bound]
MIN_BOUNDS = [costs_min_bound, gamma_min_bound, test_size_min_bound]

step_sizes = [0.3, 0.6, 0.9]
temp_vals = [5, 10, 15]
iteration_vals = [2500]

results = []
for step_size in step_sizes:
    for temp in temp_vals:
        for iters in iteration_vals:
            sa = SimulatedAnnealing(max_bounds=MAX_BOUNDS, min_bounds=MIN_BOUNDS, fn=None, dimensions=3, discrete=False,
                                    iterations=iters, step_size=step_size, temp=temp, minimize=False,
                                    data_loader=None, is_model=True)
            sa.run(verbose=False)
            result_obj = {
                'step_size': step_size,
                'temp': temp,
                'iterations': iters,
                'runtime (s)': sa.execution_time,
                'RMSE': abs(100.0 - sa.best_val),
                'model': sa
            }
            results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(9)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_three_sa.csv')
print(results.head())

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, facecolor='w')
results = json.loads(results.head().to_json(orient='records'))
for i, result in enumerate(results):
    ax.plot(np.linspace(0, iteration_vals[0] + 1, iteration_vals[0]), result['model']['val_history'],
            label=f"Model #{i + 1}")
ax.set_title('Model Convergence Vs. SA Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Maximum Accuracy (%)')
ax.legend()
fig.savefig('results/experiment_three_sa.png')
