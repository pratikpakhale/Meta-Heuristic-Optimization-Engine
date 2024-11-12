from optimization_engine import OptimizationEngine
from benchmark_functions import benchmark_functions
from algorithms import socio_evolution_learning_optimization, nomadic_people_optimizer, socio_nomadic_learning_optimizer, particle_swarm_optimization, artificial_bee_colony, hunger_games_search, sine_cosine_algorithm, grey_wolf_optimizer, differential_evolution, firefly_algorithm, salp_swarm_algorithm

if __name__ == "__main__":
    algorithms = {
        # "ABC": artificial_bee_colony,
        # "SELO": socio_evolution_learning_optimization,
        # "NPO": nomadic_people_optimizer,
        # "SNLO": socio_nomadic_learning_optimizer,
        "PSO": particle_swarm_optimization,
        "HGS": hunger_games_search,
        "SCA": sine_cosine_algorithm,
        "GWO": grey_wolf_optimizer,
        "DE": differential_evolution,
        "FA": firefly_algorithm,
        "SSA": salp_swarm_algorithm
    }

    engine = OptimizationEngine(benchmark_functions, algorithms, plot_interval=1)
    engine.run_optimization()
