# snlo.py

import numpy as np
import random
import uuid  # To generate unique identifiers

def socio_nomadic_learning_optimizer(
    objective_function,
    dimension,
    lower_bound,
    upper_bound,
    callback=None,
    **kwargs
):
    """
    Socio-Nomadic Learning Optimizer (SNLO)

    Combines the Nomadic People Optimizer (NPO) and Socio Evolution & Learning Optimization Algorithm (SELO).

    Parameters:
    - objective_function: The objective function to be minimized.
    - dimension: Dimensionality of the problem.
    - lower_bound: Lower bound array or scalar.
    - upper_bound: Upper bound array or scalar.
    - **kwargs: Additional hyperparameters.

    Returns:
    - A dictionary with the best solution found and its fitness.
    """

    # Set default hyperparameters and override with kwargs if provided
    max_iterations = kwargs.get('max_iterations', 100)
    num_clans = kwargs.get('num_clans', 5)
    families_per_clan = kwargs.get('families_per_clan', 5)
    num_parents = kwargs.get('num_parents', 2)
    num_children = kwargs.get('num_children', 3)
    alpha = kwargs.get('alpha', 0.9)
    beta = kwargs.get('beta', 0.1)
    gamma_coef = kwargs.get('gamma_coef', 0.9)
    decay_factor = kwargs.get('decay_factor', 0.95)
    increment = kwargs.get('increment', 0.05)

    # Ensure bounds are arrays
    if np.isscalar(lower_bound):
        lower_bound = np.full(dimension, lower_bound)
    else:
        lower_bound = np.array(lower_bound)
    if np.isscalar(upper_bound):
        upper_bound = np.full(dimension, upper_bound)
    else:
        upper_bound = np.array(upper_bound)

    # Initialize clans
    clans = []
    individual_id_counter = 0  # To assign unique IDs

    for _ in range(num_clans):
        clan = {}
        # Initialize clan leader
        leader_position = np.random.uniform(lower_bound, upper_bound)
        leader_fitness = objective_function(leader_position)
        clan['leader'] = {
            'id': individual_id_counter,
            'position': leader_position,
            'fitness': leader_fitness,
            'SI': np.array([lower_bound.copy(), upper_bound.copy()]).T
        }
        individual_id_counter += 1

        # Initialize families within the clan
        families = []
        for _ in range(families_per_clan):
            family = {}
            # Initialize parents
            parents = []
            for _ in range(num_parents):
                parent = {}
                x = np.random.uniform(lower_bound, upper_bound, dimension)
                parent['id'] = individual_id_counter
                individual_id_counter += 1
                parent['position'] = x
                parent['fitness'] = objective_function(x)
                parent['SI'] = np.array([lower_bound.copy(), upper_bound.copy()]).T
                parents.append(parent)
            family['parents'] = parents

            # Initialize children
            children = []
            for _ in range(num_children):
                child = {}
                random_parent = random.choice(parents)
                child['id'] = individual_id_counter
                individual_id_counter += 1
                child['SI'] = random_parent['SI'].copy()
                x = np.array([np.random.uniform(si[0], si[1]) for si in child['SI']])
                child['position'] = x
                child['fitness'] = objective_function(x)
                children.append(child)
            family['children'] = children

            families.append(family)
        clan['families'] = families

        clans.append(clan)

    # Initialize global best
    best_solution = None
    best_fitness = np.inf
    all_best_fitness = []
    all_mean_fitness = []
    all_median_fitness = []
    all_std_fitness = []
    all_worst_fitness = []
    all_fitness_values = []
    iteration_fitness_values = []
    # Iterative optimization process
    for iteration in range(1, max_iterations + 1):
        # For each clan
        for c_index, clan in enumerate(clans):
            leader = clan['leader']

            # Social Learning within Families (SELO)
            families = clan['families']
            for f_index, family in enumerate(families):
                parents = family['parents']
                children = family['children']

                # Parent Influence
                all_parents = [p for fam in clan['families'] for p in fam['parents']]
                for parent in parents:
                    r_parent = random.random()
                    if r_parent < alpha:
                        # Learn from other parents
                        fitness_list = np.array([p['fitness'] for p in all_parents])
                        max_fitness = np.max(fitness_list)
                        min_fitness = np.min(fitness_list)
                        if max_fitness == min_fitness:
                            selection_probs = np.ones(len(all_parents)) / len(all_parents)
                        else:
                            shifted_fitness = fitness_list - min_fitness + 1e-6
                            inverted_fitness = 1.0 / shifted_fitness
                            selection_probs = inverted_fitness / np.sum(inverted_fitness)
                        selected_parent = np.random.choice(all_parents, p=selection_probs)

                        # Update Social Influence (SI)
                        parent['SI'][:, 0] = selected_parent['position'] - (selected_parent['SI'][:, 1] - selected_parent['SI'][:, 0]) * gamma_coef / 2.0
                        parent['SI'][:, 1] = selected_parent['position'] + (selected_parent['SI'][:, 1] - selected_parent['SI'][:, 0]) * gamma_coef / 2.0
                    else:
                        # Self-improvement
                        parent['SI'][:, 0] = parent['position'] - (parent['SI'][:, 1] - parent['SI'][:, 0]) * gamma_coef / 2.0
                        parent['SI'][:, 1] = parent['position'] + (parent['SI'][:, 1] - parent['SI'][:, 0]) * gamma_coef / 2.0

                    # Bounds check
                    parent['SI'][:, 0] = np.maximum(parent['SI'][:, 0], lower_bound)
                    parent['SI'][:, 1] = np.minimum(parent['SI'][:, 1], upper_bound)

                    # Update position
                    x_new = np.array([np.random.uniform(si[0], si[1]) for si in parent['SI']])
                    x_new = np.clip(x_new, lower_bound, upper_bound)
                    parent['position'] = x_new
                    parent['fitness'] = objective_function(x_new)

                # Child Influence
                for child in children:
                    r_child = random.random()
                    if r_child < alpha:
                        # Learn from parents
                        fitness_list = np.array([p['fitness'] for p in parents])
                        max_fitness = np.max(fitness_list)
                        min_fitness = np.min(fitness_list)
                        if max_fitness == min_fitness:
                            selection_probs = np.ones(len(parents)) / len(parents)
                        else:
                            shifted_fitness = fitness_list - min_fitness + 1e-6
                            inverted_fitness = 1.0 / shifted_fitness
                            selection_probs = inverted_fitness / np.sum(inverted_fitness)
                        selected_individual = np.random.choice(parents, p=selection_probs)
                    elif r_child < alpha + beta:
                        # Learn from siblings
                        siblings = [sibling for sibling in children if sibling['id'] != child['id']]
                        if len(siblings) > 0:
                            fitness_list = np.array([sibling['fitness'] for sibling in siblings])
                            max_fitness = np.max(fitness_list)
                            min_fitness = np.min(fitness_list)
                            if max_fitness == min_fitness:
                                selection_probs = np.ones(len(siblings)) / len(siblings)
                            else:
                                shifted_fitness = fitness_list - min_fitness + 1e-6
                                inverted_fitness = 1.0 / shifted_fitness
                                selection_probs = inverted_fitness / np.sum(inverted_fitness)
                            selected_individual = np.random.choice(siblings, p=selection_probs)
                        else:
                            selected_individual = random.choice(parents)
                    else:
                        # Learn from peers in other clans
                        other_clans = [other_clan for idx, other_clan in enumerate(clans) if idx != c_index]
                        other_individuals = []
                        for other_clan in other_clans:
                            for other_family in other_clan['families']:
                                other_individuals.extend(other_family['parents'] + other_family['children'])
                        if len(other_individuals) > 0:
                            fitness_list = np.array([ind['fitness'] for ind in other_individuals])
                            max_fitness = np.max(fitness_list)
                            min_fitness = np.min(fitness_list)
                            if max_fitness == min_fitness:
                                selection_probs = np.ones(len(other_individuals)) / len(other_individuals)
                            else:
                                shifted_fitness = fitness_list - min_fitness + 1e-6
                                inverted_fitness = 1.0 / shifted_fitness
                                selection_probs = inverted_fitness / np.sum(inverted_fitness)
                            selected_individual = np.random.choice(other_individuals, p=selection_probs)
                        else:
                            selected_individual = random.choice(parents)

                    # Update SI
                    child['SI'][:, 0] = selected_individual['position'] - (selected_individual['SI'][:, 1] - selected_individual['SI'][:, 0]) * gamma_coef / 2.0
                    child['SI'][:, 1] = selected_individual['position'] + (selected_individual['SI'][:, 1] - selected_individual['SI'][:, 0]) * gamma_coef / 2.0

                    # Bounds check
                    child['SI'][:, 0] = np.maximum(child['SI'][:, 0], lower_bound)
                    child['SI'][:, 1] = np.minimum(child['SI'][:, 1], upper_bound)

                    # Update position
                    x_new = np.array([np.random.uniform(si[0], si[1]) for si in child['SI']])
                    x_new = np.clip(x_new, lower_bound, upper_bound)
                    fitness_new = objective_function(x_new)

                    # Behavior Correction if necessary
                    if fitness_new > child['fitness']:
                        options = []
                        x_option1 = np.array([np.random.uniform(si[0], si[1]) for si in child['SI']])
                        fitness_option1 = objective_function(x_option1)
                        options.append((x_option1, fitness_option1))

                        for parent in parents:
                            x_option_p = np.array([np.random.uniform(si[0], si[1]) for si in parent['SI']])
                            fitness_option_p = objective_function(x_option_p)
                            options.append((x_option_p, fitness_option_p))

                        options.append((x_new, fitness_new))

                        best_option = min(options, key=lambda x: x[1])
                        child['position'], child['fitness'] = best_option
                    else:
                        child['position'] = x_new
                        child['fitness'] = fitness_new

            # Movement Strategies from NPO
            # Semicircular Distribution (Local Search)
            family_positions = []
            family_fitnesses = []
            for family in families:
                for parent in family['parents']:
                    family_positions.append(parent['position'])
                    family_fitnesses.append(parent['fitness'])
                for child in family['children']:
                    family_positions.append(child['position'])
                    family_fitnesses.append(child['fitness'])

            family_positions = np.array(family_positions)
            family_fitnesses = np.array(family_fitnesses)

            # Apply Semicircular Distribution
            new_positions = []
            new_fitnesses = []
            for i in range(len(family_positions)):
                theta = np.random.uniform(0, np.pi)
                R = np.random.uniform(0, 1)  # Radius

                # Generate random direction
                direction = np.random.normal(0, 1, dimension)
                direction /= np.linalg.norm(direction)

                # Ensure direction is in the same half-space as leader
                if np.dot(direction, leader['position'] - family_positions[i]) < 0:
                    direction = -direction

                new_position = leader['position'] + R * direction
                new_position = np.clip(new_position, lower_bound, upper_bound)
                new_fitness = objective_function(new_position)

                new_positions.append(new_position)
                new_fitnesses.append(new_fitness)

            # Replace positions if better
            index = 0
            for family in families:
                for parent in family['parents']:
                    if new_fitnesses[index] < parent['fitness']:
                        parent['position'] = new_positions[index]
                        parent['fitness'] = new_fitnesses[index]
                    index += 1
                for child in family['children']:
                    if new_fitnesses[index] < child['fitness']:
                        child['position'] = new_positions[index]
                        child['fitness'] = new_fitnesses[index]
                    index += 1

            # Leadership Transition
            # Find best individual in the clan
            clan_individuals = []
            for family in families:
                clan_individuals.extend(family['parents'] + family['children'])
            best_clan_individual = min(clan_individuals, key=lambda x: x['fitness'])
            if best_clan_individual['fitness'] < leader['fitness']:
                leader['position'] = best_clan_individual['position'].copy()
                leader['fitness'] = best_clan_individual['fitness']
                leader['SI'] = best_clan_individual['SI'].copy()

            # Update global best
            if leader['fitness'] < best_fitness:
                best_fitness = leader['fitness']
                best_solution = leader['position'].copy()
        for clan in clans:
            iteration_fitness_values.append(clan['leader']['fitness'])
            for family in clan['families']:
                for parent in family['parents']:
                    iteration_fitness_values.append(parent['fitness'])
                for child in family['children']:
                    iteration_fitness_values.append(child['fitness'])
        # Periodical Meetings (Inter-Clan Learning)
        if iteration % 5 == 0:
            # Identify the best leader among clans
            clan_leaders = [clan['leader'] for clan in clans]
            best_leader = min(clan_leaders, key=lambda x: x['fitness'])
            # Update clan leaders
            for clan in clans:
                if clan['leader']['id'] != best_leader['id']:
                    r_N = clan['leader']['position']
                    f_r_N = clan['leader']['fitness']

                    # Compute direction
                    D_pos = best_leader['position'] - r_N
                    D_norm = np.linalg.norm(D_pos)
                    if D_norm == 0:
                        D_norm = 1e-8  # Prevent division by zero
                    D_pos_normalized = D_pos / D_norm

                    # Update leader position
                    Psi = +1 if best_leader['fitness'] < f_r_N else -1
                    weight = np.random.uniform(0, 1)
                    influence = Psi * D_pos_normalized * weight * (iteration / max_iterations)
                    r_new = r_N + influence
                    r_new = np.clip(r_new, lower_bound, upper_bound)
                    f_r_new = objective_function(r_new)

                    if f_r_new < clan['leader']['fitness']:
                        clan['leader']['position'] = r_new
                        clan['leader']['fitness'] = f_r_new

                        # Update global best if necessary
                        if f_r_new < best_fitness:
                            best_fitness = f_r_new
                            best_solution = r_new.copy()
                        if callback:
                            callback(iteration, best_fitness)
                        
        # Calculate statistics for this iteration
        best_fitness = min(iteration_fitness_values)
        mean_fitness = np.mean(iteration_fitness_values)
        median_fitness = np.median(iteration_fitness_values)
        std_fitness = np.std(iteration_fitness_values)
        worst_fitness = max(iteration_fitness_values)

        # Store the statistics
        all_best_fitness.append(best_fitness)
        all_mean_fitness.append(mean_fitness)
        all_median_fitness.append(median_fitness)
        all_std_fitness.append(std_fitness)
        all_worst_fitness.append(worst_fitness)
        all_fitness_values.append(iteration_fitness_values)
        # Adjust learning parameters
        alpha *= decay_factor
        beta = min(beta + increment, 1 - alpha)
        alpha = max(alpha, 0.0)
        beta = min(beta, 1.0 - alpha)
        
        


    # Calculate overall statistics
    all_fitness_values_flat = [item for sublist in all_fitness_values for item in sublist]
    overall_mean_fitness = np.mean(all_fitness_values_flat)
    overall_median_fitness = np.median(all_fitness_values_flat)
    overall_std_fitness = np.std(all_fitness_values_flat)
    overall_worst_fitness = max(all_fitness_values_flat)

    result = {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'mean_fitness': overall_mean_fitness,
        'median_fitness': overall_median_fitness,
        'std_dev_fitness': overall_std_fitness,
        'worst_fitness': overall_worst_fitness,
        'all_fitness_values': all_fitness_values
        
    }
    return result
