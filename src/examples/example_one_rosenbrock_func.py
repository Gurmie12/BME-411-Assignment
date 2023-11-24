from src.classes import Swarm
from src.utils import rosenbrock_fun
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

# Parameters of PSO that can be changed to study effects
MIN_X = -32
MAX_X = 32
MIN_Y = -32
MAX_Y = 32
NUM_PARTICLES = 50
NUM_ITERATIONS = 100
COLOR_MAP_GRANULARITY = 200
TEST_FN = rosenbrock_fun
DIMENSIONS = 2
DISCRETE = False
VERBOSE = False
w = 0.8
c1 = 0.1
c2 = 0.1

random_function_swarm_op = Swarm(w=w, c1=c1, c2=c2, fn=TEST_FN, max_bounds=[MAX_X, MAX_Y], min_bounds=[MIN_X, MIN_Y],
                                 iters=NUM_ITERATIONS, swarm_size=NUM_PARTICLES, dimensions=DIMENSIONS,
                                 data_loader=None, is_model=False, minimize=True)
random_function_swarm_op.run(verbose=VERBOSE)

# Plotting
x = np.linspace(MIN_X, MAX_X, COLOR_MAP_GRANULARITY)
y = np.linspace(MIN_Y, MAX_Y, COLOR_MAP_GRANULARITY)
X, Y = np.meshgrid(x, y)
Z = TEST_FN(X, Y)

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121, facecolor='w')
ax2 = fig.add_subplot(122, facecolor='w')


def animate(frame):
    fig.suptitle(
        f'Particle Swarm Optimization (PSO) \nIteration: {frame}, Best Position: {random_function_swarm_op.best_pos_history[frame]}, Best Value: {random_function_swarm_op.val_history[frame]} \nW={random_function_swarm_op.w}, c1={random_function_swarm_op.c1}, c2={random_function_swarm_op.c2}',
        fontsize=12)
    ax1.cla()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim(MIN_X, MAX_X)
    ax1.set_ylim(MIN_Y, MAX_Y)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Min Value')
    ax2.set_xlim(0, random_function_swarm_op.iters)
    ax2.set_ylim(random_function_swarm_op.val_history[-1] - 5, random_function_swarm_op.val_history[0] + 5)

    # data to be plot
    data = np.array([particle.history[frame] for particle in random_function_swarm_op.particles])
    global_best = random_function_swarm_op.val_history
    contour = ax1.contourf(X, Y, Z, levels=200)

    ax1.scatter(data[:, 0], data[:, 1], marker='x', color='white', label="Particles")
    ax1.scatter(random_function_swarm_op.best_pos_history[frame][0],
                random_function_swarm_op.best_pos_history[frame][1], marker='o', color='black',
                label="Best Particle")

    # plot current global best
    x_range = np.arange(0, frame + 1)
    ax2.plot(x_range, global_best[:frame + 1])
    ax1.legend()


ani = animation.FuncAnimation(fig, animate, frames=random_function_swarm_op.iters, interval=250, repeat=False,
                              blit=False)
ani.save('results/example_one_rosenbrock_func.gif', writer="imagemagick")
