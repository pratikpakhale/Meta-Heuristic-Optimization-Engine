import numpy as np

def particle_swarm_optimization(func, dim, lower_bound, upper_bound, num_particles, max_iter, w, c1, c2, callback=None, global_optimum=None, **kwargs):
    # Initialize particles and velocities
    particles = np.random.uniform(lower_bound, upper_bound, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    
    personal_best = particles.copy()
    personal_best_fitness = np.array([func(p) for p in personal_best])
    global_best = personal_best[np.argmin(personal_best_fitness)]
    global_best_fitness = np.min(personal_best_fitness)

    # Lists to store fitness statistics for each iteration
    all_best_fitness = []
    all_mean_fitness = []
    all_median_fitness = []
    all_std_fitness = []
    all_worst_fitness = []
    all_fitness_values = []

    for iteration in range(max_iter):
        r1, r2 = np.random.rand(2)
        velocities = (w * velocities + 
                      c1 * r1 * (personal_best - particles) + 
                      c2 * r2 * (global_best - particles))
        particles += velocities

        np.clip(particles, lower_bound, upper_bound, out=particles)

        fitness = np.array([func(p) for p in particles])
        
        improved = fitness < personal_best_fitness
        personal_best[improved] = particles[improved]
        personal_best_fitness[improved] = fitness[improved]

        if np.min(fitness) < global_best_fitness:
            global_best = particles[np.argmin(fitness)]
            global_best_fitness = np.min(fitness)

        # Store fitness statistics for this iteration
        mean_fitness = np.mean(fitness)
        median_fitness = np.median(fitness)
        std_fitness = np.std(fitness)
        worst_fitness = np.max(fitness)

        all_best_fitness.append(global_best_fitness)
        all_mean_fitness.append(mean_fitness)
        all_median_fitness.append(median_fitness)
        all_std_fitness.append(std_fitness)
        all_worst_fitness.append(worst_fitness)
        all_fitness_values.append(fitness.copy())

        if callback:
            callback(iteration, global_best_fitness)

    # Calculate overall statistics
    all_fitness_values_array = np.concatenate(all_fitness_values)
    overall_mean_fitness = np.mean(all_fitness_values_array)
    overall_median_fitness = np.median(all_fitness_values_array)
    overall_std_fitness = np.std(all_fitness_values_array)
    overall_worst_fitness = np.max(all_fitness_values_array)

    return {
        "best_solution": global_best,
        "best_fitness": global_best_fitness,
        "mean_fitness": overall_mean_fitness,
        "median_fitness": overall_median_fitness,
        "std_dev_fitness": overall_std_fitness,
        "worst_fitness": overall_worst_fitness,
        "all_fitness_values": all_fitness_values  # List of arrays for each iteration
    }
