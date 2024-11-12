import numpy as np

def particle_swarm_optimization(func, dim, lower_bound, upper_bound, num_particles=50, max_iter=1000, 
                              w=0.7, c1=1.5, c2=1.5, callback=None, global_optimum=None, **kwargs):
    """
    Modified PSO with improved constraint handling for engineering optimization problems
    """
    # Convert bounds to numpy arrays
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    # Initialize particles and velocities
    particles = np.random.uniform(lower_bound, upper_bound, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    
    # Initialize constraint handling parameters
    phi = 0.5  # Control parameter for constraint handling
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    # Initialize best positions
    personal_best = particles.copy()
    personal_best_fitness = np.array([func(p) for p in personal_best])
    
    # Initialize global best
    global_best = personal_best[np.argmin(personal_best_fitness)].copy()
    global_best_fitness = np.min(personal_best_fitness)
    
    # Storage for statistics
    all_best_fitness = []
    all_mean_fitness = []
    all_median_fitness = []
    all_std_fitness = []
    all_worst_fitness = []
    all_fitness_values = []
    
    # Velocity bounds based on search space
    v_max = 0.2 * (upper_bound - lower_bound)
    v_min = -v_max

    for iteration in range(max_iter):
        # Update inertia weight dynamically
        w = 0.9 - 0.5 * (iteration / max_iter)
        
        # Calculate current fitness with adaptive constraint handling
        current_fitness = np.array([func(p) for p in particles])
        
        # Apply adaptive constraint handling
        for i in range(num_particles):
            if current_fitness[i] < float('inf'):
                # Solution is feasible or partially feasible
                pass
            else:
                # Solution is infeasible - apply soft penalty
                penalty = phi * (1 - np.exp(-iteration/max_iter))
                current_fitness[i] = current_fitness[i] * penalty + epsilon
        
        # Update personal best
        improved = current_fitness < personal_best_fitness
        personal_best[improved] = particles[improved]
        personal_best_fitness[improved] = current_fitness[improved]
        
        # Update global best
        current_best_idx = np.argmin(current_fitness)
        if current_fitness[current_best_idx] < global_best_fitness:
            global_best = particles[current_best_idx].copy()
            global_best_fitness = current_fitness[current_best_idx]
        
        # Update velocities with constraint awareness
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (personal_best - particles)
        social = c2 * r2 * (global_best - particles)
        
        velocities = w * velocities + cognitive + social
        
        # Apply velocity clamping
        velocities = np.clip(velocities, v_min, v_max)
        
        # Update positions with boundary handling
        particles += velocities
        
        # Apply boundary handling - reflection method
        out_of_bounds = (particles < lower_bound) | (particles > upper_bound)
        velocities[out_of_bounds] = -velocities[out_of_bounds]  # Reverse velocity if hitting boundary
        particles = np.clip(particles, lower_bound, upper_bound)
        
        # Store statistics
        mean_fitness = np.mean(current_fitness)
        median_fitness = np.median(current_fitness)
        std_fitness = np.std(current_fitness)
        worst_fitness = np.max(current_fitness)
        
        all_best_fitness.append(global_best_fitness)
        all_mean_fitness.append(mean_fitness)
        all_median_fitness.append(median_fitness)
        all_std_fitness.append(std_fitness)
        all_worst_fitness.append(worst_fitness)
        all_fitness_values.append(current_fitness.copy())
        
        if callback:
            callback(
                iteration=iteration,
                best_fitness=global_best_fitness
            )    
    # Calculate overall statistics
    all_fitness_values_array = np.concatenate(all_fitness_values)
    
    return {
        "best_solution": global_best,
        "best_fitness": global_best_fitness,
        "mean_fitness": np.mean(all_fitness_values_array),
        "median_fitness": np.median(all_fitness_values_array),
        "std_dev_fitness": np.std(all_fitness_values_array),
        "worst_fitness": np.max(all_fitness_values_array),
        "all_fitness_values": all_fitness_values
    }