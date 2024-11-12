import numpy as np

def salp_swarm_algorithm(func, dim, lower_bound, upper_bound, num_salps=50, max_iter=1000,
                        callback=None, global_optimum=None, **kwargs):
    """
    Salp Swarm Algorithm with improved constraint handling for engineering optimization
    """
    # Convert bounds to numpy arrays if they aren't already
    lower_bound = np.array(lower_bound) if isinstance(lower_bound, (list, np.ndarray)) else np.full(dim, lower_bound)
    upper_bound = np.array(upper_bound) if isinstance(upper_bound, (list, np.ndarray)) else np.full(dim, upper_bound)
    
    # Initialize population
    salps = np.random.uniform(lower_bound, upper_bound, (num_salps, dim))
    
    # Initialize constraint handling parameters
    phi = 0.5  # Control parameter
    epsilon = 1e-10  # Small constant
    
    # Evaluate initial population
    fitness = np.array([func(s) for s in salps])
    
    # Find the best salp (food source)
    leader_idx = np.argmin(fitness)
    food_position = salps[leader_idx].copy()
    food_fitness = fitness[leader_idx]
    
    # Storage for statistics
    convergence_history = []
    all_fitness_values = []
    
    for iteration in range(max_iter):
        # Update c1 parameter (decreases exponentially)
        c1 = 2 * np.exp(-(4 * iteration / max_iter) ** 2)
        
        # Update leader (first salp)
        c2, c3 = np.random.rand(2)
        for j in range(dim):
            if c3 < 0.5:
                salps[0, j] = food_position[j] + c1 * ((upper_bound[j] - lower_bound[j]) * c2)
            else:
                salps[0, j] = food_position[j] - c1 * ((upper_bound[j] - lower_bound[j]) * c2)
        
        # Update followers
        for i in range(1, num_salps):
            salps[i] = 0.5 * (salps[i] + salps[i-1])
        
        # Boundary handling
        salps = np.clip(salps, lower_bound, upper_bound)
        
        # Evaluate salps with constraint handling
        for i in range(num_salps):
            current_fitness = func(salps[i])
            
            # Apply adaptive constraint handling
            if current_fitness < float('inf'):
                fitness[i] = current_fitness
            else:
                # Solution is infeasible - apply soft penalty
                penalty = phi * (1 - np.exp(-iteration/max_iter))
                fitness[i] = current_fitness * penalty + epsilon
            
            if fitness[i] < food_fitness:
                food_position = salps[i].copy()
                food_fitness = fitness[i]
        
        # Store history
        convergence_history.append(food_fitness)
        all_fitness_values.append(fitness.copy())
        
        if callback:
            callback(iteration=iteration, best_fitness=food_fitness)
    
    return {
        "best_solution": food_position,
        "best_fitness": food_fitness,
        "convergence_history": convergence_history,
        "mean_fitness": np.mean(all_fitness_values),
        "median_fitness": np.median(all_fitness_values),
        "std_dev_fitness": np.std(all_fitness_values),
        "worst_fitness": np.max(all_fitness_values),
        "all_fitness_values": all_fitness_values
    }