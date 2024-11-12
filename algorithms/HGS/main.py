import numpy as np

def hunger_games_search(func, dim, lower_bound, upper_bound, num_agents=30, max_iter=1000, 
                       callback=None, global_optimum=None, tolerance=1e-6, **kwargs):
    """
    Modified Hunger Games Search (HGS) algorithm with improved constraint handling
    """
    # Initialize as before
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Initialize population with feasible solutions
    X = np.random.uniform(lower_bound, upper_bound, (num_agents, dim))
    
    # Modified constraint handling parameters
    phi = 0.5  # Control parameter for constraint handling
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    # Rest of the initialization remains same
    best_position = np.zeros(dim)
    temp_position = np.zeros((num_agents, dim))
    destination_fitness = float('inf')
    worst_fitness = float('-inf')
    all_fitness = np.full(num_agents, float('inf'))
    hungry = np.zeros(num_agents)
    
    convergence_history = []
    all_fitness_values = []
    
    for iteration in range(max_iter):
        vc2 = 0.03
        count = 0
        
        # Evaluate solutions with improved constraint handling
        for i in range(num_agents):
            X[i] = np.clip(X[i], lower_bound, upper_bound)
            current_fitness = func(X[i])
            
            # Apply adaptive constraint handling
            if current_fitness < float('inf'):
                # Solution is feasible or partially feasible
                all_fitness[i] = current_fitness
            else:
                # Solution is infeasible - apply soft penalty
                penalty = phi * (1 - np.exp(-iteration/max_iter))
                all_fitness[i] = current_fitness * penalty + epsilon
        
        # Update best and worst solutions
        current_best_idx = np.argmin(all_fitness)
        current_best_fitness = all_fitness[current_best_idx]
        
        if current_best_fitness < destination_fitness:
            best_position = X[current_best_idx].copy()
            destination_fitness = current_best_fitness
            count = 0
            
        # Rest of the algorithm remains same...
        
        # Modified position updates with constraint awareness
        shrink = 2 * (1 - iteration/max_iter)
        for i in range(num_agents):
            if np.random.random() < vc2:
                # Add constraint-aware perturbation
                perturbation = np.random.randn(dim) * shrink
                X[i] = X[i] + perturbation * (upper_bound - lower_bound)
            else:
                # Original position update with bounds checking
                A = np.random.randint(0, max(1, count))
                r = np.random.random()
                vb = 2 * shrink * r - shrink
                
                if r > 1/(np.cosh(min(700, abs(all_fitness[i] - destination_fitness))) + epsilon):
                    X[i] = temp_position[A] + vb * abs(temp_position[A] - X[i])
                else:
                    X[i] = temp_position[A] - vb * abs(temp_position[A] - X[i])
            
            # Ensure bounds
            X[i] = np.clip(X[i], lower_bound, upper_bound)
        
        # Store history
        convergence_history.append(destination_fitness)
        all_fitness_values.append(all_fitness.copy())
        
        if callback:
            callback(iteration=iteration, best_fitness=destination_fitness)
    
    return {
        "best_solution": best_position,
        "best_fitness": destination_fitness,
        "convergence_history": convergence_history,
        "mean_fitness": np.mean(all_fitness_values),
        "median_fitness": np.median(all_fitness_values),
        "std_dev_fitness": np.std(all_fitness_values),
        "worst_fitness": np.max(all_fitness_values),
        "all_fitness_values": all_fitness_values
    }