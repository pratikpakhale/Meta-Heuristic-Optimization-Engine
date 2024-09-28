
import numpy as np

def particle_swarm_optimization(func, dim, lower_bound, upper_bound, num_particles, max_iter, w, c1, c2):
    
    particles = np.random.uniform(lower_bound, upper_bound, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    
    
    personal_best = particles.copy()
    personal_best_fitness = np.array([func(p) for p in personal_best])
    global_best = personal_best[np.argmin(personal_best_fitness)]
    global_best_fitness = np.min(personal_best_fitness)

    for _ in range(max_iter):
        
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

    return {
        "best_solution": global_best,
        "best_fitness": global_best_fitness
    }
