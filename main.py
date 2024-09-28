from optimization_engine import OptimizationEngine
from benchmark_functions import benchmark_functions
from algorithms import socio_evolution_learning_optimization, nomadic_people_optimizer

if __name__ == "__main__":
    algorithms = {
        "SELO": socio_evolution_learning_optimization,
        "NPO": nomadic_people_optimizer
    }

    for algo_name, algo_func in algorithms.items():
        engine = OptimizationEngine(benchmark_functions, algo_func, algo_name)
        engine.run_optimization()

