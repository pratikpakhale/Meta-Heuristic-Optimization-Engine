

import numpy as np
import random

def socio_evolution_learning_optimization(objective_function, dimension, lower_bound, upper_bound, callback=None,
    global_optimum=None,
    **kwargs):
    """
    Socio Evolution & Learning Optimization Algorithm (SELO)

    Args:
        objective_function (callable): The objective function to minimize.
        dimension (int): The number of dimensions of the problem.
        lower_bound (float or np.ndarray): The lower bound(s) of the variables.
        upper_bound (float or np.ndarray): The upper bound(s) of the variables.
        **kwargs: Additional hyperparameters.

    Returns:
        dict: A dictionary containing the best solution and its fitness.
    """

    
    F = kwargs.get('num_families', 5)          
    P = kwargs.get('num_parents', 2)           
    C = kwargs.get('num_children', 3)          
    alpha = kwargs.get('alpha', 0.9)           
    beta = kwargs.get('beta', 0.1)             
    gamma = kwargs.get('gamma', 0.9)           
    max_iter = kwargs.get('max_iter', 100)     
    decay_factor = kwargs.get('decay_factor', 0.95)  
    increment = kwargs.get('increment', 0.05)        

    
    if np.isscalar(lower_bound):
        lower_bound = np.full(dimension, lower_bound)
    if np.isscalar(upper_bound):
        upper_bound = np.full(dimension, upper_bound)

    
    families = []
    for _ in range(F):
        family = {}
        
        parents = []
        for _ in range(P):
            parent = {}
            x = np.random.uniform(lower_bound, upper_bound, dimension)
            parent['x'] = x
            parent['fitness'] = objective_function(x)
            
            parent['SI'] = np.array([lower_bound.copy(), upper_bound.copy()]).T  
            parents.append(parent)
        family['parents'] = parents

        
        children = []
        for _ in range(C):
            child = {}
            random_parent = random.choice(parents)
            
            child['SI'] = random_parent['SI'].copy()
            
            x = np.array([np.random.uniform(si[0], si[1]) for si in child['SI']])
            child['x'] = x
            child['fitness'] = objective_function(x)
            children.append(child)
        family['children'] = children

        families.append(family)

    all_best_fitness = []
    all_mean_fitness = []
    all_median_fitness = []
    all_std_fitness = []
    all_worst_fitness = []
    all_fitness_values = []
    for iteration in range(max_iter):
        
        all_parents = [parent for family in families for parent in family['parents']]
        for f_index, family in enumerate(families):
            for p_index, parent in enumerate(family['parents']):
                r_parent = random.random()
                if r_parent < alpha:
                    
                    
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
                    
                    parent['SI'][:, 0] = selected_parent['x'] - (selected_parent['SI'][:, 1] - selected_parent['SI'][:, 0]) * gamma / 2.0
                    parent['SI'][:, 1] = selected_parent['x'] + (selected_parent['SI'][:, 1] - selected_parent['SI'][:, 0]) * gamma / 2.0
                    parent['SI'][:, 0] = np.maximum(parent['SI'][:, 0], lower_bound)
                    parent['SI'][:, 1] = np.minimum(parent['SI'][:, 1], upper_bound)
                else:
                    
                    parent['SI'][:, 0] = parent['x'] - (parent['SI'][:, 1] - parent['SI'][:, 0]) * gamma / 2.0
                    parent['SI'][:, 1] = parent['x'] + (parent['SI'][:, 1] - parent['SI'][:, 0]) * gamma / 2.0
                    parent['SI'][:, 0] = np.maximum(parent['SI'][:, 0], lower_bound)
                    parent['SI'][:, 1] = np.minimum(parent['SI'][:, 1], upper_bound)
                
                x_new = np.array([np.random.uniform(si[0], si[1]) for si in parent['SI']])
                parent['x'] = x_new
                parent['fitness'] = objective_function(x_new)

        
        for f_index, family in enumerate(families):
            parents = family['parents']
            children = family['children']
            for c_index, child in enumerate(children):
                r_child = random.random()
                if r_child < alpha:
                    
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
                    
                    siblings = [sibling for idx, sibling in enumerate(children) if idx != c_index]
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
                    
                    other_families = [fam for idx, fam in enumerate(families) if idx != f_index]
                    other_individuals = []
                    for other_family in other_families:
                        other_individuals.extend(other_family['children'] + other_family['parents'])
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
                
                child['SI'][:, 0] = selected_individual['x'] - (selected_individual['SI'][:, 1] - selected_individual['SI'][:, 0]) * gamma / 2.0
                child['SI'][:, 1] = selected_individual['x'] + (selected_individual['SI'][:, 1] - selected_individual['SI'][:, 0]) * gamma / 2.0
                child['SI'][:, 0] = np.maximum(child['SI'][:, 0], lower_bound)
                child['SI'][:, 1] = np.minimum(child['SI'][:, 1], upper_bound)
                
                x_new = np.array([np.random.uniform(si[0], si[1]) for si in child['SI']])
                fitness_new = objective_function(x_new)
                
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
                    child['x'], child['fitness'] = best_option
                else:
                    
                    child['x'] = x_new
                    child['fitness'] = fitness_new
        
        alpha *= decay_factor
        beta = min(beta + increment, 1 - alpha)
        
        alpha = max(alpha, 0)
        beta = min(beta, 1 - alpha)

        iteration_fitness_values = []
        for family in families:
            for parent in family['parents']:
                iteration_fitness_values.append(parent['fitness'])
            for child in family['children']:
                iteration_fitness_values.append(child['fitness'])

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

        if callback is not None:
            callback(iteration, best_fitness)



    
    # Calculate overall statistics
    all_fitness_values_flat = [item for sublist in all_fitness_values for item in sublist]
    overall_mean_fitness = np.mean(all_fitness_values_flat)
    overall_median_fitness = np.median(all_fitness_values_flat)
    overall_std_fitness = np.std(all_fitness_values_flat)
    overall_worst_fitness = max(all_fitness_values_flat)

    # Find the best solution
    all_individuals = []
    for family in families:
        all_individuals.extend(family['parents'] + family['children'])
    best_individual = min(all_individuals, key=lambda x: x['fitness'])

    result = {
        'best_solution': best_individual['x'],
        'best_fitness': best_individual['fitness'],
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
            'all_fitness_values': all_fitness_values
        }
    }
    return result
