from src.classes import Swarm, SimulatedAnnealing
import matplotlib.pyplot as plt
import numpy as np

BEST_SWARM_PARAMS = {
    'w': 0.4,
    'c1': 0.3,
    'c2': 0.75,
    'swarm_size': 20,
    'iters': 50
}

# TODO: Update then run
BEST_SA_PARAMS = {
    'iters': 5000,
    'step_size': 20,
    'temp': 100
}

costs_min_bound = 0.01
costs_max_bound = 100
gamma_min_bound = 0.01
gamma_max_bound = 100
test_size_min_bound = 0.1
test_size_max_bound = 0.3

MAX_BOUNDS = [costs_max_bound, gamma_max_bound, test_size_max_bound]
MIN_BOUNDS = [costs_min_bound, gamma_min_bound, test_size_min_bound]

swarm = Swarm(w=BEST_SWARM_PARAMS['w'], c1=BEST_SWARM_PARAMS['c1'], c2=BEST_SWARM_PARAMS['c2'], fn=None,
              max_bounds=MAX_BOUNDS, min_bounds=MIN_BOUNDS, iters=BEST_SWARM_PARAMS['iters'],
              swarm_size=BEST_SWARM_PARAMS['swarm_size'], dimensions=3, discrete=False, data_loader=None,
              is_model=True, minimize=False)
swarm.run()

sa = SimulatedAnnealing(max_bounds=MAX_BOUNDS, min_bounds=MIN_BOUNDS, fn=None, dimensions=3,
                        discrete=False, iterations=BEST_SA_PARAMS['iters'], step_size=BEST_SA_PARAMS['step_size'],
                        temp=BEST_SA_PARAMS['temp'], minimize=False, data_loader=None, is_model=True)
sa.run()

fig = plt.figure(figsize=(13, 6))
fig.suptitle('PSA Vs. SA Comparison (Example Three)')
ax1 = fig.add_subplot(121, facecolor='w')
ax2 = fig.add_subplot(122, facecolor='w')
ax1.plot(np.linspace(0, BEST_SWARM_PARAMS['iters'] + 1, BEST_SWARM_PARAMS['iters']), swarm.val_history, label=f"PSO",
         c='b')
ax1.axhline(y=100.0, color='r', linestyle='-', label='Global Max Acc.')
ax2.plot(np.linspace(0, BEST_SA_PARAMS['iters'] + 1, BEST_SA_PARAMS['iters']), sa.val_history, label=f"SA", c='b')
ax2.axhline(y=100.0, color='r', linestyle='-', label='Global Max Acc.')
ax1.set_title('Best PSO Model Convergence Vs. PSO Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Maximum Accuracy (%)')
ax1.legend()

ax2.set_title('Best SA Model Convergence Vs. SA Iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Maximum Accuracy (%)')
ax2.legend()
fig.savefig('results/comparison_example_three.png')
