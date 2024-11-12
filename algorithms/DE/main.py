import numpy as np

def differential_evolution(func, dim, lower_bound, upper_bound, population_size=50, max_iter=1000,
                         F=0.5, CR=0.5, callback=None, global_optimum=None, **kwargs):
    """
    Differential Evolution with constraint handling for engineering optimization
    """
    # Initialize
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Initialize population
    population = np.random.uniform(lower_bound, upper_bound, (population_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Storage for statistics
    convergence_history = []
    all_fitness_values = []
    
    for iteration in range(max_iter):
        for i in range(population_size):
            # Select three random individuals
            candidates = [idx for idx in range(population_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector
            mutant = population[a] + F * (population[b] - population[c])
            
            # Crossover
            crossover_mask = np.random.rand(dim) < CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial = np.where(crossover_mask, mutant, population[i])
            
            # Boundary handling
            trial = np.clip(trial, lower_bound, upper_bound)
            
            # Selection with constraint handling
            trial_fitness = func(trial)
            
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
        
        # Store history
        convergence_history.append(best_fitness)
        all_fitness_values.append(fitness.copy())
        
        if callback:
            callback(iteration=iteration, best_fitness=best_fitness)
    
    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "convergence_history": convergence_history,
        "mean_fitness": np.mean(all_fitness_values),
        "median_fitness": np.median(all_fitness_values),
        "std_dev_fitness": np.std(all_fitness_values),
        "worst_fitness": np.max(all_fitness_values),
        "all_fitness_values": all_fitness_values
    }