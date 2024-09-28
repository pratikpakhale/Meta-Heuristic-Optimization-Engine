import numpy as np

def schwefel(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def griewank(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

def michalewicz(x):
    m = 10
    return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x**2) / np.pi)**(2*m))

def zakharov(x):
    return np.sum(x**2) + (0.5 * np.sum(np.arange(1, len(x)+1) * x))**2 + (0.5 * np.sum(np.arange(1, len(x)+1) * x))**4

def dixon_price(x):
    return (x[0] - 1)**2 + np.sum((np.arange(2, len(x)+1) * (2 * x[1:]**2 - x[:-1])**2))

benchmark_functions = {
    "Schwefel": {
        "function": schwefel,
        "DIMENSION": 2,
        "LOWER_BOUND": -500,
        "UPPER_BOUND": 500,
        "GLOBAL_OPTIMUM": -837.9658
    },
    "Sphere": {
        "function": sphere,
        "DIMENSION": 2,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Rosenbrock": {
        "function": rosenbrock,
        "DIMENSION": 2,
        "LOWER_BOUND": -5,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    },
    "Rastrigin": {
        "function": rastrigin,
        "DIMENSION": 2,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Ackley": {
        "function": ackley,
        "DIMENSION": 2,
        "LOWER_BOUND": -32.768,
        "UPPER_BOUND": 32.768,
        "GLOBAL_OPTIMUM": 0
    },
    "Griewank": {
        "function": griewank,
        "DIMENSION": 2,
        "LOWER_BOUND": -600,
        "UPPER_BOUND": 600,
        "GLOBAL_OPTIMUM": 0
    },
    "Levy": {
        "function": levy,
        "DIMENSION": 2,
        "LOWER_BOUND": -10,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    },
    "Michalewicz": {
        "function": michalewicz,
        "DIMENSION": 2,
        "LOWER_BOUND": 0,
        "UPPER_BOUND": np.pi,
        "GLOBAL_OPTIMUM": -1.8013 # Approximate for 2D
    },
    "Zakharov": {
        "function": zakharov,
        "DIMENSION": 2,
        "LOWER_BOUND": -5,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    },
    "Dixon-Price": {
        "function": dixon_price,
        "DIMENSION": 2,
        "LOWER_BOUND": -10,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    }
}
