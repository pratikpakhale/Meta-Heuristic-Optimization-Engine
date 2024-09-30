import numpy as np
from scipy.special import gamma

def nomadic_people_optimizer(
    objective_function,
    dimension,
    lower_bound,
    upper_bound,
    max_iterations=100,
    num_clans=5,
    families_per_clan=10,
    callback=None,
    global_optimum=None, **kwargs
):
    """
    Nomadic People Optimizer (NPO) Algorithm Implementation.

    Parameters:
    - objective_function: The objective function to be minimized.
    - dimension: Dimensionality of the problem.
    - lower_bound: Lower bound array or scalar.
    - upper_bound: Upper bound array or scalar.
    - max_iterations: Maximum number of iterations.
    - num_clans: Number of clans (swarms).
    - families_per_clan: Number of families per clan.

    Returns:
    - A dictionary with the best solution found and its fitness.
      Additionally, returns statistical information and fitness history.
    """

    # Ensure bounds are arrays
    if np.isscalar(lower_bound):
        lower_bound = np.full(dimension, lower_bound)
    else:
        lower_bound = np.array(lower_bound)
    if np.isscalar(upper_bound):
        upper_bound = np.full(dimension, upper_bound)
    else:
        upper_bound = np.array(upper_bound)

    # Initialize leaders (Sheikhs) for each clan
    leaders_positions = []  # Positions of leaders
    leaders_fitness = []    # Fitness of leaders

    for _ in range(num_clans):
        position = np.random.uniform(lower_bound, upper_bound)
        fitness = objective_function(position)
        leaders_positions.append(position)
        leaders_fitness.append(fitness)

    leaders_positions = np.array(leaders_positions)
    leaders_fitness = np.array(leaders_fitness)

    # Initialize global best leader
    best_leader_index = np.argmin(leaders_fitness)
    best_leader_position = leaders_positions[best_leader_index].copy()
    best_leader_fitness = leaders_fitness[best_leader_index]

    # Lists to store fitness values for each iteration
    all_best_fitness = []
    all_mean_fitness = []
    all_median_fitness = []
    all_std_fitness = []
    all_worst_fitness = []
    all_fitness_values = []

    # Iterative optimization process
    for iteration in range(1, max_iterations + 1):
        # Fitness values for this iteration
        iteration_fitness_values = []

        # For each clan
        for c in range(num_clans):
            leader_position = leaders_positions[c]
            leader_fitness = leaders_fitness[c]

            # Semicircular Distribution (Local Exploitation)
            family_positions = []
            family_fitnesses = []

            for _ in range(families_per_clan):
                # Generate a random angle theta between 0 and pi
                theta = np.random.uniform(0, np.pi)
                R = np.random.uniform(0, 1)  # Radius can be adjusted

                # Generate a random direction in D dimensions
                direction = np.random.normal(0, 1, dimension)
                direction /= np.linalg.norm(direction)  # Normalize

                # Project direction onto a hemisphere (semicircle in higher dimensions)
                # Ensure the direction is in the same half-space as a reference vector
                reference_vector = np.ones(dimension)
                if np.dot(direction, reference_vector) < 0:
                    direction = -direction

                # Update family position
                new_position = leader_position + R * direction

                # Ensure within bounds
                new_position = np.clip(new_position, lower_bound, upper_bound)
                fitness = objective_function(new_position)

                family_positions.append(new_position)
                family_fitnesses.append(fitness)

            family_positions = np.array(family_positions)
            family_fitnesses = np.array(family_fitnesses)

            # Store fitness values
            iteration_fitness_values.extend(family_fitnesses)

            # Leadership Transition
            min_family_index = np.argmin(family_fitnesses)
            min_family_fitness = family_fitnesses[min_family_index]
            if min_family_fitness < leader_fitness:
                # Update leader
                leaders_positions[c] = family_positions[min_family_index]
                leaders_fitness[c] = min_family_fitness
                leader_position = leaders_positions[c]
                leader_fitness = leaders_fitness[c]
            else:
                # Families Searching (Global Exploration)
                # Calculate alpha_c (average distance between families and leader)
                distances = np.linalg.norm(family_positions - leader_position, axis=1)
                alpha_c = np.mean(distances)

                beta = 1.5
                sigma_u = (
                    gamma(1 + beta)
                    * np.sin(np.pi * beta / 2)
                    / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
                ) ** (1 / beta)

                for i in range(families_per_clan):
                    # Lévy flight step
                    u = np.random.normal(0, sigma_u, size=dimension)
                    v = np.random.normal(0, 1, size=dimension)
                    step = u / (np.abs(v) ** (1 / beta))

                    # Step size
                    step_size = alpha_c * step

                    # Update position
                    new_position = family_positions[i] + step_size
                    new_position = np.clip(new_position, lower_bound, upper_bound)
                    fitness = objective_function(new_position)

                    family_positions[i] = new_position
                    family_fitnesses[i] = fitness

                # Store updated fitness values
                iteration_fitness_values.extend(family_fitnesses)

                # Leadership Transition after exploration
                min_family_index = np.argmin(family_fitnesses)
                min_family_fitness = family_fitnesses[min_family_index]
                if min_family_fitness < leader_fitness:
                    # Update leader
                    leaders_positions[c] = family_positions[min_family_index]
                    leaders_fitness[c] = min_family_fitness
                    leader_position = leaders_positions[c]
                    leader_fitness = leaders_fitness[c]

            # Update global best leader if necessary
            if leader_fitness < best_leader_fitness:
                best_leader_fitness = leader_fitness
                best_leader_position = leader_position.copy()
                best_leader_index = c

        # Periodical Meetings (Cooperation Among Clans)
        if iteration % 5 == 0:
            # Identify the best leader
            best_leader_index = np.argmin(leaders_fitness)
            best_leader_position = leaders_positions[best_leader_index]
            best_leader_fitness = leaders_fitness[best_leader_index]
            # Update each clan leader
            for c in range(num_clans):
                if c == best_leader_index:
                    continue  # Skip the best leader
                r_N = leaders_positions[c]
                f_r_N = leaders_fitness[c]

                # Compute direction 
                if best_leader_fitness < f_r_N:
                    Psi = +1
                else:
                    Psi = -1
                # Compute normalized distance ΔPos
                D_pos = best_leader_position - r_N
                D_norm = np.linalg.norm(D_pos)
                if D_norm == 0:
                    D_norm = 1e-8  # Prevent division by zero
                D_pos_normalized = D_pos / D_norm

                # Update leader position
                # Random weight for influence
                weight = np.random.uniform(0, 1)
                # Influence diminishes over time
                influence = Psi * D_pos_normalized * weight * (iteration / max_iterations)
                r_new = r_N + influence
                r_new = np.clip(r_new, lower_bound, upper_bound)
                f_r_new = objective_function(r_new)

                if f_r_new < f_r_N:
                    leaders_positions[c] = r_new
                    leaders_fitness[c] = f_r_new
                    # Update global best if necessary
                    if f_r_new < best_leader_fitness:
                        best_leader_fitness = f_r_new
                        best_leader_position = r_new.copy()
                        best_leader_index = c

                # Store leader fitness after meetings
                iteration_fitness_values.append(leaders_fitness[c])

        # Aggregate all fitness values (families and leaders) for this iteration
        total_fitness_values = np.array(iteration_fitness_values + leaders_fitness.tolist())

        # Calculate statistics for this iteration
        mean_fitness = np.mean(total_fitness_values)
        median_fitness = np.median(total_fitness_values)
        std_fitness = np.std(total_fitness_values)
        worst_fitness = np.max(total_fitness_values)

        all_best_fitness.append(best_leader_fitness)
        all_mean_fitness.append(mean_fitness)
        all_median_fitness.append(median_fitness)
        all_std_fitness.append(std_fitness)
        all_worst_fitness.append(worst_fitness)
        all_fitness_values.append(total_fitness_values)

        if callback:
            callback(iteration, best_leader_fitness)

    # Calculate overall statistics
    all_fitness_values_array = np.concatenate(all_fitness_values)
    overall_mean_fitness = np.mean(all_fitness_values_array)
    overall_median_fitness = np.median(all_fitness_values_array)
    overall_std_fitness = np.std(all_fitness_values_array)
    overall_worst_fitness = np.max(all_fitness_values_array)

    result = {
        'best_solution': best_leader_position,
        'best_fitness': best_leader_fitness,
        'mean_fitness': overall_mean_fitness,
        'median_fitness': overall_median_fitness,
        'std_dev_fitness': overall_std_fitness,
        'worst_fitness': overall_worst_fitness,
        'fitness_history': {
            'best': all_best_fitness,
            'mean': all_mean_fitness,
            'median': all_median_fitness,
            'std_dev': all_std_fitness,
            'worst': all_worst_fitness,
            'all_fitness_values': all_fitness_values  # List of arrays for each iteration
        }
    }
    return result
