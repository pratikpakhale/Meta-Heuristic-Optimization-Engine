import numpy as np

def grey_wolf_optimizer(func, dim, lower_bound, upper_bound, num_wolves=30, max_iter=1000,
                       callback=None, global_optimum=None, **kwargs):
    """
    Grey Wolf Optimizer with constraint handling for engineering optimization
    """
    # Initialize
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Population initialization
    positions = np.random.uniform(lower_bound, upper_bound, (num_wolves, dim))
    
    # Evaluate initial population
    fitness = np.array([func(pos) for pos in positions])
    
    # Sort wolves
    sorted_idx = np.argsort(fitness)
    alpha_pos = positions[sorted_idx[0]].copy()
    beta_pos = positions[sorted_idx[1]].copy()
    delta_pos = positions[sorted_idx[2]].copy()
    alpha_score = fitness[sorted_idx[0]]
    
    # Storage for statistics
    convergence_history = []
    all_fitness_values = []
    
    # Main loop
    for iteration in range(max_iter):
        # Update a linearly from 2 to 0
        a = 2 * (1 - iteration/max_iter)
        
        for i in range(num_wolves):
            for j in range(dim):
                # Update coefficients
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                # Calculate new position
                D_alpha = abs(C1 * alpha_pos[j] - positions[i,j])
                D_beta = abs(C2 * beta_pos[j] - positions[i,j])
                D_delta = abs(C3 * delta_pos[j] - positions[i,j])
                
                X1 = alpha_pos[j] - A1 * D_alpha
                X2 = beta_pos[j] - A2 * D_beta
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i,j] = (X1 + X2 + X3) / 3
            
            # Boundary handling
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)
            
            # Evaluate new position
            new_fitness = func(positions[i])
            
            # Update alpha, beta, delta
            if new_fitness < alpha_score:
                delta_pos = beta_pos.copy()
                beta_pos = alpha_pos.copy()
                alpha_pos = positions[i].copy()
                alpha_score = new_fitness
            elif new_fitness < fitness[sorted_idx[1]]:
                delta_pos = beta_pos.copy()
                beta_pos = positions[i].copy()
            elif new_fitness < fitness[sorted_idx[2]]:
                delta_pos = positions[i].copy()
        
        # Store history
        convergence_history.append(alpha_score)
        all_fitness_values.append(fitness.copy())
        
        if callback:
            callback(iteration=iteration, best_fitness=alpha_score)
    
    return {
        "best_solution": alpha_pos,
        "best_fitness": alpha_score,
        "convergence_history": convergence_history,
        "mean_fitness": np.mean(all_fitness_values),
        "median_fitness": np.median(all_fitness_values),
        "std_dev_fitness": np.std(all_fitness_values),
        "worst_fitness": np.max(all_fitness_values),
        "all_fitness_values": all_fitness_values
    }