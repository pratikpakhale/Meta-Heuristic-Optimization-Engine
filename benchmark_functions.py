import numpy as np

def goldstein_price_2d(x):
    x1, x2 = x
    return (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) * \
           (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

def schwefel_2d(x):
    return - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def schaffer_2d(x):
    x1, x2 = x
    num = np.sin(x1**2 - x2**2)**2 - 0.5
    den = (1 + 0.001*(x1**2 + x2**2))**2
    return 0.5 + num / den

def rosenbrock_10d(x):
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere_10d(x):
    x = np.array(x)
    return np.sum(x**2)

def ackley_10d(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - \
           np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin_10d(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def martin_gaddy_2d(x):
    x1, x2 = x
    return (x1 - x2)**2 + ((x1 + x2 - 10) / 3)**2

def easom_2d(x):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def griewank_10d(x):
    x = np.array(x)
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

benchmark_functions = {
    "Goldstein-Price": {
        "function": goldstein_price_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -2,
        "UPPER_BOUND": 2,
        "GLOBAL_OPTIMUM": 3
    },
    "Schwefel": {
        "function": schwefel_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -500,
        "UPPER_BOUND": 500,
        "GLOBAL_OPTIMUM": -837.658
    },
    "Schaffer": {
        "function": schaffer_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -100,
        "UPPER_BOUND": 100,
        "GLOBAL_OPTIMUM": 0
    },
    "Rosenbrock": {
        "function": rosenbrock_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -1.2,
        "UPPER_BOUND": 1.2,
        "GLOBAL_OPTIMUM": 0
    },
    "Sphere": {
        "function": sphere_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Ackley": {
        "function": ackley_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -32.768,
        "UPPER_BOUND": 32.768,
        "GLOBAL_OPTIMUM": 0
    },
    "Rastrigin": {
        "function": rastrigin_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Martin-Gaddy": {
        "function": martin_gaddy_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": 0,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    },
    "Easom": {
        "function": easom_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -100,
        "UPPER_BOUND": 100,
        "GLOBAL_OPTIMUM": -1
    },
    "Griewank": {
        "function": griewank_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -600,
        "UPPER_BOUND": 600,
        "GLOBAL_OPTIMUM": 0
    }
}