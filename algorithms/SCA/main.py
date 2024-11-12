import numpy as np

def sine_cosine_algorithm(func, dim, lower_bound, upper_bound, num_agents=30, max_iter=1000, 
                         callback=None, global_optimum=None, A=2, **kwargs):
    """
    Sine Cosine Algorithm with constraint handling for engineering optimization
    """
    # Initialize
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Population initialization
    X = np.random.uniform(lower_bound, upper_bound, (num_agents, dim))
    
    # Initialize best solution
    fitness = np.array([func(x) for x in X])
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Storage for statistics
    convergence_history = []
    all_fitness_values = []
    
    # Main loop
    for t in range(max_iter):
        # Update a (decreases linearly from A to 0)
        a = A * (1 - t/max_iter)
        
        for i in range(num_agents):
            # Update each dimension
            for j in range(dim):
                # Calculate r1, r2, r3, r4
                r1 = a * np.random.random()
                r2 = 2 * np.pi * np.random.random()
                r3 = np.random.random()
                r4 = np.random.random()
                
                # Position update using sine or cosine
                if r4 < 0.5:
                    # Sine update
                    X[i,j] = X[i,j] + r1 * np.sin(r2) * abs(r3 * best_solution[j] - X[i,j])
                else:
                    # Cosine update
                    X[i,j] = X[i,j] + r1 * np.cos(r2) * abs(r3 * best_solution[j] - X[i,j])
            
            # Boundary handling
            X[i] = np.clip(X[i], lower_bound, upper_bound)
            
            # Evaluate new solution
            new_fitness = func(X[i])
            
            # Update best solution if better
            if new_fitness < best_fitness:
                best_solution = X[i].copy()
                best_fitness = new_fitness
        
        # Store history
        convergence_history.append(best_fitness)
        all_fitness_values.append(fitness.copy())
        
        if callback:
            callback(iteration=t, best_fitness=best_fitness)
    
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