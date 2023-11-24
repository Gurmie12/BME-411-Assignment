import numpy as np


def ackley_fun(x, y):
    """Ackley function
    Domain: -32 < xi < 32
    Global minimum: f_min(0,..,0)=0
    """
    return -20 * np.exp(-.2 * np.sqrt(1/2 * (x ** 2 + y ** 2))) - np.exp(
        1/2 * (np.cos(np.pi * 2 * x) + np.cos(np.pi * 2 * y))) + np.exp(1) + 20


def rosenbrock_fun(x, y):
    """Rosenbrock function
    Domain: -5 < xi < 5
    Global minimun: f_min(1,..,1)=0
    """
    return 100 * (y - x ** 2) ** 2 + (x - 1) ** 2
