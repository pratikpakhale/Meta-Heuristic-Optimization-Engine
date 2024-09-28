from optimization_engine import OptimizationEngine
from benchmark_functions import benchmark_functions
from algorithms import socio_evolution_learning_optimization, nomadic_people_optimizer, socio_nomadic_learning_optimizer, particle_swarm_optimization

if __name__ == "__main__":
    algorithms = {
        # "SELO": socio_evolution_learning_optimization,
        # "NPO": nomadic_people_optimizer,
        "SNLO": socio_nomadic_learning_optimizer,
        # "PSO": particle_swarm_optimization
    }

    for algo_name, algo_func in algorithms.items():
        engine = OptimizationEngine(benchmark_functions, algo_func, algo_name, plot_interval=1)
        engine.run_optimization()

