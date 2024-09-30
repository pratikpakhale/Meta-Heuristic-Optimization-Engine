import numpy as np
import json
import logging
from typing import Callable, Dict, Any, List
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from benchmark_functions import benchmark_functions
from sklearn.decomposition import PCA

class OptimizationEngine:
    def __init__(self, benchmark_functions: Dict[str, Dict[str, Any]], algorithms: Dict[str, Callable], plot_interval=1):
        self.benchmark_functions = benchmark_functions
        self.algorithms = algorithms
        self.plot_interval = plot_interval
        self.setup_directories()
        self.setup_logging()
        self.load_hyperparameters()
        self.setup_data_structures()

    def setup_directories(self):
        self.plots_dir = "plots"
        self.logs_dir = "logs"
        self.csv_dir = "csv_results"
        self.json_dir = "json_results"
        for directory in [self.plots_dir, self.logs_dir, self.csv_dir, self.json_dir]:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.logs_dir, f"optimization_results_{self.timestamp}.log")
        
        self.logger = logging.getLogger("OptimizationEngine")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers = []

        file_handler = logging.FileHandler(self.log_filename)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        for handler in [file_handler, console_handler]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_hyperparameters(self):
        self.hyperparameters = {}
        for algo_name in self.algorithms.keys():
            file_path = f"algorithms/{algo_name}/hyperparameters.json"
            with open(file_path, 'r') as f:
                self.hyperparameters[algo_name] = json.load(f)

    def setup_data_structures(self):
        self.iteration_history = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}
        self.fitness_history = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}
        self.avg_fitness_history = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}
        self.best_solution_history = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}
        self.trajectory_first_dimension = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}
        self.all_solutions_history = {algo: {func: [] for func in self.benchmark_functions} for algo in self.algorithms}

    def callback(self, algo_name, func_name, iteration, best_fitness, best_solution, avg_fitness, all_solutions):
        if iteration % self.plot_interval == 0:
            self.iteration_history[algo_name][func_name].append(iteration)
            self.fitness_history[algo_name][func_name].append(best_fitness)
            self.avg_fitness_history[algo_name][func_name].append(avg_fitness)
            self.best_solution_history[algo_name][func_name].append(best_solution.copy())
            self.trajectory_first_dimension[algo_name][func_name].append(best_solution[0])
            self.all_solutions_history[algo_name][func_name].append(all_solutions.copy())



    def save_all_iterations(self):
        # Prepare data to be saved
        all_data = {
            "iterations": self.iteration_history,
            "fitness": self.fitness_history,
            "avg_fitness": self.avg_fitness_history,
            "best_solution": self.best_solution_history,
            "trajectory_first_dimension": self.trajectory_first_dimension
        }
        
        # Convert any numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(v) for v in obj]
            else:
                return obj

        all_data = convert_numpy_to_list(all_data)
        
        # Define the JSON file path
        json_file_path = os.path.join(self.json_dir, "all_iterations_fitness.json")
        
        # Write data to JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(all_data, json_file, indent=4)

    def plot_convergence(self, algo_name, func_name):
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_history[algo_name][func_name], self.fitness_history[algo_name][func_name], 'b-', marker='o')
        plt.title(f"Convergence Plot - {algo_name} - {func_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")

        if all(f > 0 for f in self.fitness_history[algo_name][func_name]):
            plt.yscale('log')
        else:
            min_fitness = min(self.fitness_history[algo_name][func_name])
            max_fitness = max(self.fitness_history[algo_name][func_name])
            plt.ylim(min_fitness - 0.1 * abs(min_fitness), max_fitness + 0.1 * abs(max_fitness))

        plt.grid(True)
        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "convergence.png"))
        plt.close()

    def plot_average_fitness_history(self, algo_name, func_name):
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_history[algo_name][func_name], self.avg_fitness_history[algo_name][func_name], 'g-', marker='o')
        plt.title(f"Average Fitness History - {algo_name} - {func_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Average Fitness")
        plt.grid(True)
        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "average_fitness_history.png"))
        plt.close()

    def plot_trajectory_first_dimension(self, algo_name, func_name):
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_history[algo_name][func_name], self.trajectory_first_dimension[algo_name][func_name], 'r-', marker='o')
        plt.title(f"Trajectory in First Dimension - {algo_name} - {func_name}")
        plt.xlabel("Iteration")
        plt.ylabel("First Dimension of Best Solution")
        plt.grid(True)
        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "trajectory_first_dimension.png"))
        plt.close()

    def plot_search_history(self, algo_name, func_name, func_details):
        all_solutions_list = self.all_solutions_history[algo_name][func_name]
        all_solutions = np.vstack(all_solutions_list)

        
        if func_details['DIMENSION'] == 2:
            x_range = [func_details["LOWER_BOUND"], func_details["UPPER_BOUND"]]
            y_range = [func_details["LOWER_BOUND"], func_details["UPPER_BOUND"]]

            x = np.linspace(x_range[0], x_range[1], 100)
            y = np.linspace(y_range[0], y_range[1], 100)
            X, Y = np.meshgrid(x, y)

            Z = np.array([[func_details["function"]([x_i, y_j]) for x_i in x] for y_j in y])

            plt.figure(figsize=(10, 8))
            plt.contour(X, Y, Z, levels=50, cmap='viridis')
            plt.scatter(all_solutions[:, 0], all_solutions[:, 1], c='red', s=10, alpha=0.5)
            plt.colorbar(label='Function Value')
            plt.title(f"Search History - {algo_name} - {func_name}")
            plt.xlabel("X")
            plt.ylabel("Y")
        else:
            pca = PCA(n_components=2)
            reduced_solutions = pca.fit_transform(all_solutions)

            x_min, x_max = reduced_solutions[:, 0].min() - 1, reduced_solutions[:, 0].max() + 1
            y_min, y_max = reduced_solutions[:, 1].min() - 1, reduced_solutions[:, 1].max() + 1

            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = pca.inverse_transform([X[i, j], Y[i, j]])
                    Z[i, j] = func_details["function"](point)

            plt.figure(figsize=(10, 8))
            plt.contour(X, Y, Z, levels=50, cmap='viridis')
            plt.scatter(reduced_solutions[:, 0], reduced_solutions[:, 1], c='red', s=10, alpha=0.5)
            plt.colorbar(label='Function Value')
            plt.title(f"Search History (PCA Reduced) - {algo_name} - {func_name}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

        plt.grid(True)
        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "search_history.png"))
        plt.close()


    def plot_comparative_results(self):
        max_iterations = 100  # Adjust this if needed
        for func_name in self.benchmark_functions:
            plt.figure(figsize=(12, 8))
            all_positive = True
            min_fitness = float('inf')
            max_fitness = float('-inf')
            has_data = False

            for algo_name in self.algorithms:
                if func_name in self.fitness_history[algo_name] and self.fitness_history[algo_name][func_name]:
                    fitness_values = self.fitness_history[algo_name][func_name]

                    # Adjust fitness_values to have length max_iterations
                    fitness_values = list(fitness_values)
                    if len(fitness_values) < max_iterations:
                        last_value = fitness_values[-1]
                        fitness_values.extend([last_value] * (max_iterations - len(fitness_values)))
                    else:
                        fitness_values = fitness_values[:max_iterations]

                    iterations = list(range(max_iterations))
                    plt.plot(iterations, fitness_values, label=algo_name)

                    if any(f <= 0 for f in fitness_values):
                        all_positive = False
                    min_fitness = min(min_fitness, min(fitness_values))
                    max_fitness = max(max_fitness, max(fitness_values))
                    has_data = True

            if not has_data:
                self.logger.warning(f"No data available for function: {func_name}")
                plt.close()
                continue

            plt.title(f"Comparative Convergence Plot - {func_name}")
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.legend()
            plt.grid(True)

            # Adjust y-axis limits to add extra space at the bottom
            ymin, ymax = plt.ylim()
            y_margin = 0.1 * abs(ymax - ymin)  # 10% of the y-range
            plt.ylim(ymin - y_margin, ymax + y_margin)

            # Optionally, adjust margins around the plot
            plt.margins(x=0, y=0.1)  # Adds a 10% margin on the y-axis

            # Use tight layout to adjust spacing
            plt.tight_layout()

            comparative_dir = os.path.join(self.plots_dir, 'comparative')
            os.makedirs(comparative_dir, exist_ok=True)

            plt.savefig(os.path.join(comparative_dir, f"comparative_{func_name}.png"))
            plt.close()

    def plot_box_results(self):
        for func_name in self.benchmark_functions:
            plt.figure(figsize=(12, 8))
            data_to_plot = []

            for algo_name in self.algorithms:
                if func_name in self.fitness_history[algo_name] and self.fitness_history[algo_name][func_name]:
                    fitness_values = self.fitness_history[algo_name][func_name]
                    data_to_plot.append(fitness_values)

            if not data_to_plot:
                self.logger.warning(f"No data available for function: {func_name}")
                plt.close()
                continue

            plt.boxplot(data_to_plot, labels=list(self.algorithms.keys()))
            plt.title(f"Box Plot of Fitness Values - {func_name}")
            plt.xlabel("Algorithms")
            plt.ylabel("Fitness Values")
            plt.grid(True)

            # Use tight layout to adjust spacing
            plt.tight_layout()

            boxplot_dir = os.path.join(self.plots_dir, 'boxplots')
            os.makedirs(boxplot_dir, exist_ok=True)

            plt.savefig(os.path.join(boxplot_dir, f"boxplot_{func_name}.png"))
            plt.close()

    def plot_3d_surface(self, func_details, algo_name, func_name):
        """
        Plots a 3D surface for a given 2D function and saves the plot as an image.
        """
        if func_details['DIMENSION'] != 2:
            self.logger.warning(f"Cannot plot 3D surface for {func_name} as it is not a 2D function.")
            return
        
        x_range = [func_details["LOWER_BOUND"], func_details["UPPER_BOUND"]]
        y_range = [func_details["LOWER_BOUND"], func_details["UPPER_BOUND"]]

        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)

        Z = np.array([[func_details["function"]([x_i, y_j]) for x_i in x] for y_j in y])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
        ax.set_title(f"{func_name} Function")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Save the plot
        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{func_name.lower().replace('-', '_')}_surface.png"), dpi=300)
        plt.close() 

    def run_single_optimization(self, algo_name, func_name, func_details):
        self.logger.info(f"\n--- Starting optimization for {algo_name} on {func_name} ---")
        self.logger.info(f"Dimension: {func_details['DIMENSION']}")
        self.logger.info(f"Bounds: [{func_details['LOWER_BOUND']}, {func_details['UPPER_BOUND']}]")
        self.logger.info(f"Global Optimum: {func_details['GLOBAL_OPTIMUM']}")

        random_seed = self.hyperparameters[algo_name].get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            self.logger.info(f"Random seed set to: {random_seed}")

        start_time = datetime.now()
        result = self.algorithms[algo_name](
            func_details['function'],
            func_details['DIMENSION'],
            func_details['LOWER_BOUND'],
            func_details['UPPER_BOUND'],
            callback=lambda iteration, best_fitness, best_solution, avg_fitness, all_solutions: self.callback(algo_name, func_name, iteration, best_fitness, best_solution, avg_fitness, all_solutions),
            global_optimum=func_details['GLOBAL_OPTIMUM'],
            **self.hyperparameters[algo_name]
        )
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        best_fitness = result['best_fitness']
        mean_fitness = result['mean_fitness']
        median_fitness = result['median_fitness']
        std_dev_fitness = result['std_dev_fitness']
        worst_fitness = result['worst_fitness']
        closeness_to_optimum = abs(best_fitness - func_details['GLOBAL_OPTIMUM'])
        best_solution = np.array2string(result['best_solution'], precision=4, suppress_small=True)
        all_fitness_values = result['all_fitness_values']

        self.logger.info("\nOptimization completed")
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Best fitness: {best_fitness:.6f}")
        self.logger.info(f"Mean fitness: {mean_fitness:.6f}")
        self.logger.info(f"Median fitness: {median_fitness:.6f}")
        self.logger.info(f"Std dev fitness: {std_dev_fitness:.6f}")
        self.logger.info(f"Worst fitness: {worst_fitness:.6f}")
        self.logger.info(f"Closeness from global optimum: {closeness_to_optimum:.6e}")
        self.logger.info(f"Best solution: {best_solution}")
        self.logger.info("-" * 40)

        plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot the 3D surface if applicable
        self.plot_3d_surface(func_details, algo_name, func_name)

        return {
            'Algorithm': algo_name,
            'Function': func_name,
            'Dimension': func_details['DIMENSION'],
            'Lower Bound': func_details['LOWER_BOUND'],
            'Upper Bound': func_details['UPPER_BOUND'],
            'Global Optimum': func_details['GLOBAL_OPTIMUM'],
            'Best Fitness': best_fitness,
            'Mean Fitness': mean_fitness,
            'Median Fitness': median_fitness,
            'Std Dev Fitness': std_dev_fitness,
            'Worst Fitness': worst_fitness,
            'Closeness to Global Optimum': closeness_to_optimum,
            'Execution Time (s)': execution_time,
            'Best Solution': best_solution,
            'Random Seed': random_seed,
            'All fitness values': all_fitness_values
        }

    def run_optimization(self):
        self.logger.info("=== Optimization Engine Started ===")
        for algo_name, algo_func in self.algorithms.items():
            self.logger.info(f"Algorithm: {algo_name}")
            self.logger.info(f"Hyperparameters: {json.dumps(self.hyperparameters[algo_name], indent=2)}")
            self.logger.info("=" * 40)

        results = []
        for algo_name in self.algorithms:
            for func_name, func_details in self.benchmark_functions.items():
                # Create the plot directory for this algorithm and function
                plot_dir = os.path.join(self.plots_dir, algo_name, func_name)
                os.makedirs(plot_dir, exist_ok=True)

                result = self.run_single_optimization(algo_name, func_name, func_details)
                results.append(result)

        self.save_results_to_csv(results)
        self.save_all_iterations()
        self.plot_box_results()
        self.plot_comparative_results()
        self.plot_all_histories()
        self.logger.info("\n=== Optimization Engine Finished ===")

    def save_results_to_csv(self, results: List[Dict[str, Any]]):
        csv_filename = os.path.join(self.csv_dir, f"results_{self.timestamp}.csv")
        fieldnames = ['Timestamp', 'Log File', 'Algorithm', 'Function', 'Dimension', 
                    'Lower Bound', 'Upper Bound', 'Global Optimum', 'Best Fitness', 
                    'Mean Fitness', 'Median Fitness', 'Std Dev Fitness', 'Worst Fitness',
                    'Closeness to Global Optimum', 'Execution Time (s)', 'Best Solution', 'Random Seed']
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {
                    'Timestamp': self.timestamp,
                    'Log File': os.path.basename(self.log_filename),
                    'Algorithm': result['Algorithm'],
                    'Function': result['Function'],
                    'Dimension': result['Dimension'],
                    'Lower Bound': result['Lower Bound'],
                    'Upper Bound': result['Upper Bound'],
                    'Global Optimum': result['Global Optimum'],
                    'Best Fitness': result['Best Fitness'],
                    'Mean Fitness': result['Mean Fitness'],
                    'Median Fitness': result['Median Fitness'],
                    'Std Dev Fitness': result['Std Dev Fitness'],
                    'Worst Fitness': result['Worst Fitness'],
                    'Closeness to Global Optimum': result['Closeness to Global Optimum'],
                    'Execution Time (s)': result['Execution Time (s)'],
                    'Best Solution': str(result['Best Solution']),
                    'Random Seed': result['Random Seed']
                }
                writer.writerow(row)

    def plot_all_histories(self):
        for algo_name in self.algorithms:
            for func_name, func_details in self.benchmark_functions.items():
                self.plot_convergence(algo_name, func_name)
                self.plot_average_fitness_history(algo_name, func_name)
                self.plot_trajectory_first_dimension(algo_name, func_name)
                self.plot_search_history(algo_name, func_name, func_details)
