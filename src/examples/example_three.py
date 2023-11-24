import numpy as np

from src.classes import Swarm
import matplotlib.pyplot as plt

# Parameters of PSO that can be changed to study effects
W = 0.8
c1 = 0.1
c2 = 0.1
NUM_ITERATIONS = 50
SWARM_SIZE = 1
VERBOSE = 1
costs_min_bound = 0.01
costs_max_bound = 100
gamma_min_bound = 0.01
gamma_max_bound = 100
test_size_min_bound = 0.1
test_size_max_bound = 0.3

MAX_BOUNDS = [costs_max_bound, gamma_max_bound, test_size_max_bound]
MIN_BOUNDS = [costs_min_bound, gamma_min_bound, test_size_min_bound]

if __name__ == '__main__':
    model_param_op_swarm = Swarm(w=W, c1=c1, c2=c2, fn=None, max_bounds=MAX_BOUNDS, min_bounds=MIN_BOUNDS,
                                 iters=NUM_ITERATIONS,
                                 swarm_size=SWARM_SIZE, dimensions=3, discrete=False, data_loader=None, is_model=True,
                                 minimize=False)
    model_param_op_swarm.run(verbose=VERBOSE)

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111, facecolor='w')
    ax.plot(np.linspace(0, model_param_op_swarm.iters + 1, model_param_op_swarm.iters), model_param_op_swarm.val_history)
    ax.set_title('Model Accuracy Vs. PSO Iterations')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy (%)')
    fig.savefig('results/example_three.png')
