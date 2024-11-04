import numpy as np

def hunger_games_search(func, dim, lower_bound, upper_bound, num_agents=30, max_iter=1000, 
                       callback=None, global_optimum=None, tolerance=1e-6, **kwargs):
    """
    Modified Hunger Games Search (HGS) algorithm implementation.
    """
    # Convert bounds to numpy arrays if they aren't already
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Initialize population
    X = np.random.uniform(lower_bound, upper_bound, (num_agents, dim))
    
    # Initialize algorithm variables
    best_position = np.zeros(dim)
    temp_position = np.zeros((num_agents, dim))
    destination_fitness = float('inf')
    worst_fitness = float('-inf')
    all_fitness = np.full(num_agents, float('inf'))
    hungry = np.zeros(num_agents)
    
    # Lists to store optimization history
    convergence_history = []
    all_fitness_values = []
    
    # Main loop
    for iteration in range(max_iter):
        vc2 = 0.03
        count = 0
        
        # Evaluate all solutions
        for i in range(num_agents):
            X[i] = np.clip(X[i], lower_bound, upper_bound)
            all_fitness[i] = func(X[i])
        
        # Update best and worst fitness
        current_best_idx = np.argmin(all_fitness)
        current_best_fitness = all_fitness[current_best_idx]
        current_worst_fitness = np.max(all_fitness)
        
        if current_best_fitness < destination_fitness:
            best_position = X[current_best_idx].copy()
            destination_fitness = current_best_fitness
            count = 0
        
        if current_worst_fitness > worst_fitness:
            worst_fitness = current_worst_fitness
        
        # Calculate hungry values and update positions
        for i in range(num_agents):
            if destination_fitness == all_fitness[i]:
                hungry[i] = 0
                count += 1
                if count <= num_agents:
                    temp_position[count-1] = X[i].copy()
            else:
                temp_rand = np.random.random()
                c = ((all_fitness[i] - destination_fitness) / 
                     (worst_fitness - destination_fitness + 1e-10) * temp_rand * 2 * 
                     (upper_bound - lower_bound))  # Now works with numpy arrays
                b = 100 * (1 + temp_rand) if np.mean(c) < 100 else np.mean(c)
                hungry[i] += b
        
        # Update positions
        shrink = 2 * (1 - iteration/max_iter)
        for i in range(num_agents):
            if np.random.random() < vc2:
                X[i] = X[i] * (1 + np.random.randn(dim))
            else:
                A = np.random.randint(0, max(1, count))
                r = np.random.random()
                vb = 2 * shrink * r - shrink
                
                cosh_value = np.cosh(min(700, abs(all_fitness[i] - destination_fitness)))
                if r > 1/cosh_value:
                    X[i] = temp_position[A] + vb * abs(temp_position[A] - X[i])
                else:
                    X[i] = temp_position[A] - vb * abs(temp_position[A] - X[i])
            
            # Ensure bounds
            X[i] = np.clip(X[i], lower_bound, upper_bound)
        
        # Store history and check convergence
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
