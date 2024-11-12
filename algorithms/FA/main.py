import numpy as np

def firefly_algorithm(func, dim, lower_bound, upper_bound, num_fireflies=50, max_iter=1000,
                     alpha=0.5, beta=0.2, gamma=1.0, callback=None, global_optimum=None, **kwargs):
    """
    Firefly Algorithm with constraint handling for engineering optimization
    """
    # Initialize
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Initialize fireflies
    fireflies = np.random.uniform(lower_bound, upper_bound, (num_fireflies, dim))
    intensity = np.array([func(f) for f in fireflies])
    
    best_idx = np.argmin(intensity)
    best_solution = fireflies[best_idx].copy()
    best_fitness = intensity[best_idx]
    
    # Storage for statistics
    convergence_history = []
    all_fitness_values = []
    
    for iteration in range(max_iter):
        # Update alpha (decreases with iteration)
        alpha_t = alpha * (0.1 + 0.9 * (1 - iteration/max_iter))
        
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if intensity[j] < intensity[i]:
                    # Calculate distance
                    r = np.sqrt(np.sum((fireflies[i] - fireflies[j])**2))
                    
                    # Calculate attractiveness
                    beta_r = beta * np.exp(-gamma * r**2)
                    
                    # Update position
                    rand = np.random.uniform(-0.5, 0.5, dim)
                    fireflies[i] += beta_r * (fireflies[j] - fireflies[i]) + alpha_t * rand
                    
                    # Boundary handling
                    fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)
                    
                    # Update intensity
                    new_intensity = func(fireflies[i])
                    if new_intensity < intensity[i]:
                        intensity[i] = new_intensity
                        
                        if new_intensity < best_fitness:
                            best_solution = fireflies[i].copy()
                            best_fitness = new_intensity
        
        # Store history
        convergence_history.append(best_fitness)
        all_fitness_values.append(intensity.copy())
        
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