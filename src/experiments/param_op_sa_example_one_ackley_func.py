from src.classes import SimulatedAnnealing
from src.utils import ackley_fun
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# Parameters of SA that can be changed to study effects
MIN_X = -32
MAX_X = 32
MIN_Y = -32
MAX_Y = 32
STEP_SIZE = 1
TEMP = 10
TEST_FN = ackley_fun

step_sizes = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 10.0]
temp_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
iteration_vals = [5000]

results = []
for step_size in step_sizes:
    for temp in temp_vals:
        for iters in iteration_vals:
            sa = SimulatedAnnealing(max_bounds=[MAX_X, MAX_Y], min_bounds=[MIN_X, MIN_Y], fn=ackley_fun, dimensions=2,
                                    discrete=False, iterations=iters, step_size=step_size, minimize=True, temp=temp)
            sa.run()
            result_obj = {
                'step_size': step_size,
                'temp': temp,
                'iterations': iters,
                'runtime (s)': sa.execution_time,
                'RMSE': abs(0 - sa.best_val),
                'model': sa
            }
            results.append(result_obj)
results = pd.DataFrame(results, index=[i for i in range(8*10)])
results.sort_values('RMSE', axis=0, inplace=True)
results.to_csv('results/experiment_one_sa_ackley_func.csv')
print(results.head())

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, facecolor='w')
results = json.loads(results.head().to_json(orient='records'))
for i, result in enumerate(results):
    ax.plot(np.linspace(0, max(iteration_vals) + 1, max(iteration_vals)), result['model']['val_history'],
            label=f"Model #{i + 1}")
ax.set_title('Model Convergence Vs. SA Iterations')
ax.set_xlabel('Iterations')
ax.set_ylabel('Minimum Value')
ax.legend()
fig.savefig('results/experiment_one_sa_ackley_func.png')
