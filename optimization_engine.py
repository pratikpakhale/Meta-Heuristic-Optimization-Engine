import numpy as np
import json
import logging
from typing import Callable, Dict, Any
from datetime import datetime
import os
import csv

class OptimizationEngine:
    def __init__(self, benchmark_functions: Dict[str, Dict[str, Any]], algorithm: Callable, algorithm_name: str):
        self.benchmark_functions = benchmark_functions
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        hyperparameters_file = f"algorithms/{algorithm_name}/hyperparameters.json"
        self.hyperparameters = self.load_hyperparameters(hyperparameters_file)
        self.setup_logging()

    def load_hyperparameters(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

    def setup_logging(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        self.log_filename = os.path.join(log_dir, f"optimization_results_{self.timestamp}.log")
        
        # Create a logger
        self.logger = logging.getLogger(self.algorithm_name)
        self.logger.setLevel(logging.INFO)

        # Remove all handlers associated with the logger object
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # CSV setup
        self.csv_filename = os.path.join(log_dir, "results.csv")
        file_exists = os.path.isfile(self.csv_filename)
        
        self.csv_file = open(self.csv_filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'Log File', 'Algorithm', 'Function', 'Dimension', 
                                      'Lower Bound', 'Upper Bound', 'Global Optimum', 'Best Fitness', 
                                      'Closeness to Global Optimum', 'Execution Time (s)', 'Best Solution', 'Random Seed'])

    def run_optimization(self):
        self.logger.info("=== Optimization Engine Started ===")
        self.logger.info(f"Algorithm: {self.algorithm_name}")
        self.logger.info(f"Hyperparameters: {json.dumps(self.hyperparameters, indent=2)}")
        self.logger.info("=" * 40)

        for func_name, func_details in self.benchmark_functions.items():
            self.logger.info(f"\n--- Starting optimization for {func_name} ---")
            self.logger.info(f"Dimension: {func_details['DIMENSION']}")
            self.logger.info(f"Bounds: [{func_details['LOWER_BOUND']}, {func_details['UPPER_BOUND']}]")
            self.logger.info(f"Global Optimum: {func_details['GLOBAL_OPTIMUM']}")

            # Set the random seed
            random_seed = self.hyperparameters.get('random_seed', None)
            if random_seed is not None:
                np.random.seed(random_seed)
                self.logger.info(f"Random seed set to: {random_seed}")

            start_time = datetime.now()
            result = self.algorithm(
                func_details['function'],
                func_details['DIMENSION'],
                func_details['LOWER_BOUND'],
                func_details['UPPER_BOUND'],
                **self.hyperparameters
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

            self.csv_writer.writerow([
                self.timestamp,
                os.path.basename(self.log_filename),
                self.algorithm_name,
                func_name,
                func_details['DIMENSION'],
                func_details['LOWER_BOUND'],
                func_details['UPPER_BOUND'],
                func_details['GLOBAL_OPTIMUM'],
                best_fitness,
                closeness_to_optimum,
                execution_time,
                best_solution,
                random_seed
            ])
            self.csv_file.flush()  

        self.logger.info("\n=== Optimization Engine Finished ===")
        self.csv_file.close()
