import numpy as np
import json
import logging
from typing import Callable, Dict, Any, List
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

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
        for directory in [self.plots_dir, self.logs_dir, self.csv_dir]:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.logs_dir, f"optimization_results_{self.timestamp}.log")
        
        self.logger = logging.getLogger("OptimizationEngine")
        self.logger.setLevel(logging.INFO)

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

    def callback(self, algo_name, func_name, iteration, best_fitness):
        

        if iteration % self.plot_interval == 0:
            
            self.iteration_history[algo_name][func_name].append(iteration)
            self.fitness_history[algo_name][func_name].append(best_fitness)


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
        plt.savefig(os.path.join(self.plots_dir, f"{algo_name}_{func_name}.png"))
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

            plt.savefig(os.path.join(self.plots_dir, f"comparative_{func_name}.png"))
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
            callback=lambda iteration, best_fitness: self.callback(algo_name, func_name, iteration, best_fitness),
            global_optimum=func_details['GLOBAL_OPTIMUM'],
            **self.hyperparameters[algo_name]
        )
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        best_fitness = result['best_fitness']
        closeness_to_optimum = abs(best_fitness - func_details['GLOBAL_OPTIMUM'])
        best_solution = np.array2string(result['best_solution'], precision=4, suppress_small=True)

        self.logger.info("\nOptimization completed")
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Best fitness: {best_fitness:.6f}")
        self.logger.info(f"Closeness from global optimum: {closeness_to_optimum:.6e}")
        self.logger.info(f"Best solution: {best_solution}")
        self.logger.info("-" * 40)

        return {
            'Algorithm': algo_name,
            'Function': func_name,
            'Dimension': func_details['DIMENSION'],
            'Lower Bound': func_details['LOWER_BOUND'],
            'Upper Bound': func_details['UPPER_BOUND'],
            'Global Optimum': func_details['GLOBAL_OPTIMUM'],
            'Best Fitness': best_fitness,
            'Closeness to Global Optimum': closeness_to_optimum,
            'Execution Time (s)': execution_time,
            'Best Solution': best_solution,
            'Random Seed': random_seed
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
                result = self.run_single_optimization(algo_name, func_name, func_details)
                results.append(result)

        self.save_results_to_csv(results)
        self.plot_all_convergences()
        self.plot_comparative_results()
        self.logger.info("\n=== Optimization Engine Finished ===")

    
    def save_results_to_csv(self, results: List[Dict[str, Any]]):
        csv_filename = os.path.join(self.csv_dir, f"results_{self.timestamp}.csv")
        fieldnames = ['Timestamp', 'Log File', 'Algorithm', 'Function', 'Dimension', 
                      'Lower Bound', 'Upper Bound', 'Global Optimum', 'Best Fitness', 
                      'Closeness to Global Optimum', 'Execution Time (s)', 'Best Solution', 'Random Seed']
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {
                    'Timestamp': self.timestamp,
                    'Log File': os.path.basename(self.log_filename),
                    **result
                }
                writer.writerow(row)

    def plot_all_convergences(self):
        for algo_name in self.algorithms:
            for func_name in self.benchmark_functions:
                self.plot_convergence(algo_name, func_name)
