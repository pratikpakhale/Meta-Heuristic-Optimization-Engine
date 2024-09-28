import numpy as np

def schwefel(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


benchmark_functions = {
    "Schwefel": {
        "function": schwefel,
        "DIMENSION": 2,
        "LOWER_BOUND": -500,
        "UPPER_BOUND": 500,
        "GLOBAL_OPTIMUM": -837.658
    },
    
}
