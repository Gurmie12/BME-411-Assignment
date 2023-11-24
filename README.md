## BME 411 - Particle Swarm Optimization Vs. Simulated Annealing

#### Group #20

### Setup

to setup your environment to run these files, follow these steps:

- Create a new python virtual environment: `python3 -m virtualenv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install all package requirements: `pip install -r requirements.txt`

### File Structure

For the purposes of this assignment all files have been seperated into 3 categories:

- classes
- comparisons
- examples
- experiments
- utils

#### Classes

The `classes` directory in the `src` folder contains all of our custom class implementations for PSO and SA (in addition
to some helper classes for data loading).
All of these files do not need to be interacted with as they are called directly from the scripts found in the other
folders.

#### Comparisons

The `comaprisons` directory in the `src` folder contains all of the scripts that were run to compare the PSO to SA. The
results from these scripts along with all other scripts can be found in the `results` folder.

#### Examples

The `examples` directory in the `src` folder contains all of the scripts that were used to test and run the PSO against
our three examples scenarios and data. The result of executing these files is a graph that contains the convergence of
the algorithm as well as a GIF describing how the particles are moving around the problem space.

#### Experiments

The `experiments` directory in the `src` folder contains all of the experiments we ran on PSO and SA to test the effect
of different PSO and SA parameters on their ability to find a global min or max.
By running these files you will get an output in the `results` folder which is a CSV, containing the params, execution
time and RMSE for all combinations of params in the search space.

#### Utils

The `utils` directory in the `src` folder contains function definitions for continuous functions we used for testing.
You
do not need to call these functions directly.

#### Data

The `data` directory contains all datasets that were used for all examples.

### Execution

In order to execute comparisons, examples or experiments, follow the following command structure:
`python3 -m src.<comparisons | examples | experiments>.<name of file in folder you want to run>`