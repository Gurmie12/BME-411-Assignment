from src.classes import Swarm, SimulatedAnnealing, DataLoader
import matplotlib.pyplot as plt
import numpy as np

BEST_SWARM_PARAMS = {
    'w': 1.0,
    'c1': 4.0,
    'c2': 4.0,
    'swarm_size': 100,
    'iters': 50
}

BEST_SA_PARAMS = {
    'iters': 5000,
    'step_size': 20,
    'temp': 100
}

data_loader = DataLoader(path_to_file='data/rainfall_data.csv',
                         independent_column_names=['x', 'y'],
                         dependent_column_name='z')

swarm = Swarm(w=BEST_SWARM_PARAMS['w'], c1=BEST_SWARM_PARAMS['c1'], c2=BEST_SWARM_PARAMS['c2'], fn=None,
              max_bounds=None, min_bounds=None, iters=BEST_SWARM_PARAMS['iters'],
              swarm_size=BEST_SWARM_PARAMS['swarm_size'], dimensions=2, discrete=True, data_loader=data_loader,
              is_model=False, minimize=True)
swarm.run()

sa = SimulatedAnnealing(max_bounds=None, min_bounds=None, fn=None, dimensions=2,
                        discrete=True, iterations=BEST_SA_PARAMS['iters'], step_size=BEST_SA_PARAMS['step_size'],
                        temp=BEST_SA_PARAMS['temp'], minimize=True, data_loader=data_loader, is_model=False)
sa.run()

fig = plt.figure(figsize=(13, 6))
fig.suptitle('PSA Vs. SA Comparison (Example Two)')
ax1 = fig.add_subplot(121, facecolor='w')
ax2 = fig.add_subplot(122, facecolor='w')
ax1.plot(np.linspace(0, BEST_SWARM_PARAMS['iters'] + 1, BEST_SWARM_PARAMS['iters']), swarm.val_history, label=f"PSO",
         c='b')
ax1.axhline(y=62.2, color='r', linestyle='-', label='Actual Minimum')
ax2.plot(np.linspace(0, BEST_SA_PARAMS['iters'] + 1, BEST_SA_PARAMS['iters']), sa.val_history, label=f"SA", c='b')
ax2.axhline(y=62.2, color='r', linestyle='-', label='Actual Minimum')
ax1.set_title('Best PSO Model Convergence Vs. PSO Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Minimum Value')
ax1.legend()

ax2.set_title('Best SA Model Convergence Vs. SA Iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Minimum Value')
ax2.legend()
fig.savefig('results/comparison_example_two.png')
