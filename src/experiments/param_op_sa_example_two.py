from src.classes import DataLoader, SimulatedAnnealing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

data_loader = DataLoader(path_to_file='data/rainfall_data.csv',
                         independent_column_names=['x', 'y'],
                         dependent_column_name='z')

step_sizes = [20, 40, 60, 80, 100, 120, 140, 160]
temp_vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
iteration_vals = [5000]

results = []
for step_size in step_sizes:
    for temp in temp_vals:
        for iters in iteration_vals:
            sa = SimulatedAnnealing(max_bounds=None, min_bounds=None, fn=None, dimensions=2, discrete=True,
                                    iterations=iters, step_size=step_size, temp=temp, minimize=True,
                                    data_loader=data_loader)
            sa.run(verbose=False)
            result_obj = {
                'step_size': step_size,
                'temp': temp,
                'iterations': iters,
                'runtime (s)': sa.execution_time,
                'RMSE': abs(62.2 - sa.best_val),
                'model': sa
            }
            results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(8 * 10)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_two_sa.csv')
print(results.head())

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, facecolor='w')
results = json.loads(results.head().to_json(orient='records'))
for i, result in enumerate(results):
    ax.plot(np.linspace(0, iteration_vals[0] + 1, iteration_vals[0]), result['model']['val_history'],
            label=f"Model #{i + 1}")
ax.set_title('Model Convergence Vs. SA Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Minimum Value')
ax.legend()
fig.savefig('results/experiment_two_sa.png')
