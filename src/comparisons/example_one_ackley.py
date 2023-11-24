from src.classes import Swarm, SimulatedAnnealing
from src.utils import ackley_fun
import matplotlib.pyplot as plt
import numpy as np

BEST_SWARM_PARAMS = {
    'w': 0.2,
    'c1': 0.1,
    'c2': 1.0,
    'swarm_size': 80,
    'iters': 50
}

BEST_SA_PARAMS = {
    'iters': 5000,
    'step_size': 0.5,
    'temp': 8
}

MIN_X = -32
MAX_X = 32
MIN_Y = -32
MAX_Y = 32

swarm = Swarm(w=BEST_SWARM_PARAMS['w'], c1=BEST_SWARM_PARAMS['c1'], c2=BEST_SWARM_PARAMS['c2'], fn=ackley_fun,
              max_bounds=[MAX_X, MAX_Y], min_bounds=[MIN_X, MIN_Y], iters=BEST_SWARM_PARAMS['iters'],
              swarm_size=BEST_SWARM_PARAMS['swarm_size'], dimensions=2, discrete=False, data_loader=None,
              is_model=False, minimize=True)
swarm.run()

sa = SimulatedAnnealing(max_bounds=[MAX_X, MAX_Y], min_bounds=[MIN_X, MIN_Y], fn=ackley_fun, dimensions=2,
                        discrete=False, iterations=BEST_SA_PARAMS['iters'], step_size=BEST_SA_PARAMS['step_size'],
                        temp=BEST_SA_PARAMS['temp'], minimize=True, data_loader=None, is_model=False)
sa.run()

fig = plt.figure(figsize=(13, 6))
fig.suptitle('PSA Vs. SA Comparison (Example One)')
ax1 = fig.add_subplot(121, facecolor='w')
ax2 = fig.add_subplot(122, facecolor='w')
ax1.plot(np.linspace(0, BEST_SWARM_PARAMS['iters'] + 1, BEST_SWARM_PARAMS['iters']), swarm.val_history, label=f"PSO",
         c='b')
ax1.axhline(y=0, color='r', linestyle='-', label='Actual Minimum')
ax2.plot(np.linspace(0, BEST_SA_PARAMS['iters'] + 1, BEST_SA_PARAMS['iters']), sa.val_history, label=f"SA", c='b')
ax2.axhline(y=0, color='r', linestyle='-', label='Actual Minimum')
ax1.set_title('Best PSO Model Convergence Vs. PSO Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Minimum Value')
ax1.legend()

ax2.set_title('Best SA Model Convergence Vs. SA Iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Minimum Value')
ax2.legend()
fig.savefig('results/comparison_example_one.png')
