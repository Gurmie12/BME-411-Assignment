import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from src.classes import Swarm, DataLoader

# Parameters of PSO that can be changed to study effects
NUM_PARTICLES = 50
NUM_ITERATIONS = 100
COLOR_MAP_GRANULARITY = 200
DIMENSIONS = 2
DISCRETE = True
VERBOSE = True
w = 0.8
c1 = 0.1
c2 = 0.1

# RUN SWARM
data_loader = DataLoader(path_to_file='data/rainfall_data.csv',
                         independent_column_names=['x', 'y'],
                         dependent_column_name='z')
discrete_swarm_op = Swarm(w=w, c1=c1, c2=c2, fn=None, max_bounds=None, min_bounds=None,
                          iters=NUM_ITERATIONS, swarm_size=NUM_PARTICLES, dimensions=DIMENSIONS, discrete=DISCRETE,
                          data_loader=data_loader, is_model=False, minimize=True)
discrete_swarm_op.run(verbose=VERBOSE)

# Plotting
x = data_loader.df['x'].to_numpy()
y = data_loader.df['y'].to_numpy()
z = data_loader.df['z'].to_numpy()

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121, facecolor='w')
ax2 = fig.add_subplot(122, facecolor='w')

def animate(frame):
    fig.suptitle(
        f'Particle Swarm Optimization (PSO) \nIteration: {frame}, Best Position: {discrete_swarm_op.best_pos_history[frame]}, Best Value: {discrete_swarm_op.val_history[frame]} \nW={discrete_swarm_op.w}, c1={discrete_swarm_op.c1}, c2={discrete_swarm_op.c2}',
        fontsize=12)
    ax1.cla()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(min(y), max(y))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Min Value')
    ax2.set_xlim(0, discrete_swarm_op.iters)
    ax2.set_ylim(discrete_swarm_op.val_history[-1] - 5, discrete_swarm_op.val_history[0] + 5)

    # data to be plot
    data = np.array([particle.history[frame] for particle in discrete_swarm_op.particles])
    global_best = discrete_swarm_op.val_history

    ax1.scatter(x, y, c=z, cmap=plt.cm.coolwarm)
    ax1.scatter(data[:, 0], data[:, 1], marker='x', color='white', label="Particles")
    ax1.scatter(discrete_swarm_op.best_pos_history[frame][0], discrete_swarm_op.best_pos_history[frame][1], marker='o', color='black',
                label="Best Particle")

    # plot current global best
    x_range = np.arange(0, frame + 1)
    ax2.plot(x_range, global_best[:frame + 1])
    ax1.legend()

ani = animation.FuncAnimation(fig, animate, frames=discrete_swarm_op.iters, interval=250, repeat=False, blit=False)
ani.save('results/example_two.gif', writer="imagemagick")
