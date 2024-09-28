# main.py
from optimization_engine import OptimizationEngine
from benchmark_functions import benchmark_functions
from algorithms import socio_evolution_learning_optimization

if __name__ == "__main__":
    engine = OptimizationEngine(benchmark_functions, socio_evolution_learning_optimization, "SELO")
    
    engine.run_optimization()
