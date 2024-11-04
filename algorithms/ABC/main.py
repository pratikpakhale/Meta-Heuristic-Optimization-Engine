import numpy as np

def artificial_bee_colony(func, dim, lower_bound, upper_bound, callback=None, **kwargs):
    # Extract hyperparameters from kwargs
    num_bees = kwargs.get('num_bees', 50)
    max_iter = kwargs.get('max_iter', 1000)
    limit = kwargs.get('limit', 20)

    # Initialize food sources
    food_sources = np.random.uniform(lower_bound, upper_bound, (num_bees, dim))
    fitness = np.array([func(fs) for fs in food_sources])
    trials = np.zeros(num_bees)

    # Find the initial best solution
    best_idx = np.argmin(fitness)
    best_solution = food_sources[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Initialize list to store all fitness values for each iteration
    all_fitness_values = []

    for iteration in range(max_iter):
        # Employed Bee Phase
        for i in range(num_bees):
            new_solution = food_sources[i].copy()
            j = np.random.randint(dim)
            k = np.random.choice([x for x in range(num_bees) if x != i])
            phi = np.random.uniform(-1, 1)
            new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            
            new_fitness = func(new_solution)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Calculate fitness values for probability calculation
        fitness_values = np.where(fitness >= 0, 1 / (1 + fitness), 1 + np.abs(fitness))
        probabilities = fitness_values / np.sum(fitness_values)
        
        # Onlooker Bee Phase
        for _ in range(num_bees):
            i = np.random.choice(num_bees, p=probabilities)
            new_solution = food_sources[i].copy()
            j = np.random.randint(dim)
            k = np.random.choice([x for x in range(num_bees) if x != i])
            phi = np.random.uniform(-1, 1)
            new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            
            new_fitness = func(new_solution)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        abandoned = np.where(trials >= limit)[0]
        for i in abandoned:
            food_sources[i] = np.random.uniform(lower_bound, upper_bound, dim)
            fitness[i] = func(food_sources[i])
            trials[i] = 0

        # Update best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = food_sources[best_idx].copy()
            best_fitness = fitness[best_idx]

        # Store all fitness values for this iteration
        all_fitness_values.append(fitness.copy())

        # Compute average fitness for this iteration
        avg_fitness = np.mean(fitness)

        # Get current solutions
        all_solutions = food_sources.copy()

        # Callback with updated parameters
        if callback:
            callback(iteration, best_fitness, best_solution, avg_fitness, all_solutions)

    # Calculate additional statistics
    all_fitness_array = np.array(all_fitness_values)
    mean_fitness = np.mean(all_fitness_array)
    std_dev_fitness = np.std(all_fitness_array)
    worst_fitness = np.max(all_fitness_array)
    median_fitness = np.median(all_fitness_array)

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "std_dev_fitness": std_dev_fitness,
        "worst_fitness": worst_fitness,
        "median_fitness": median_fitness,
        'all_fitness_values': all_fitness_values  # List of arrays for each iteration
    }
