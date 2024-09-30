from optimization_engine import OptimizationEngine
from benchmark_functions import benchmark_functions
from algorithms import socio_evolution_learning_optimization, nomadic_people_optimizer, socio_nomadic_learning_optimizer, particle_swarm_optimization, artificial_bee_colony

if __name__ == "__main__":
    algorithms = {
        "ABC": artificial_bee_colony,
        "SELO": socio_evolution_learning_optimization,
        "NPO": nomadic_people_optimizer,
        "SNLO": socio_nomadic_learning_optimizer,
        "PSO": particle_swarm_optimization
    }

    engine = OptimizationEngine(benchmark_functions, algorithms, plot_interval=1)
    engine.run_optimization()
