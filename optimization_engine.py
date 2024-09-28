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
        logging.basicConfig(filename=self.log_filename, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

        
        self.csv_filename = os.path.join(log_dir, "results.csv")
        file_exists = os.path.isfile(self.csv_filename)
        
        self.csv_file = open(self.csv_filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'Log File', 'Algorithm', 'Function', 'Dimension', 
                                      'Lower Bound', 'Upper Bound', 'Global Optimum', 'Best Fitness', 
                                      'Closeness to Global Optimum', 'Execution Time (s)', 'Best Solution'])

    def run_optimization(self):
        logging.info("=== Optimization Engine Started ===")
        logging.info(f"Algorithm: {self.algorithm_name}")
        logging.info(f"Hyperparameters: {json.dumps(self.hyperparameters, indent=2)}")
        logging.info("=" * 40)

        for func_name, func_details in self.benchmark_functions.items():
            logging.info(f"\n--- Starting optimization for {func_name} ---")
            logging.info(f"Dimension: {func_details['DIMENSION']}")
            logging.info(f"Bounds: [{func_details['LOWER_BOUND']}, {func_details['UPPER_BOUND']}]")
            logging.info(f"Global Optimum: {func_details['GLOBAL_OPTIMUM']}")

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

            logging.info("\nOptimization completed")
            logging.info(f"Execution time: {execution_time:.2f} seconds")
            logging.info(f"Best fitness: {best_fitness:.6f}")
            logging.info(f"Closeness from global optimum: {closeness_to_optimum:.6e}")
            logging.info(f"Best solution: {best_solution}")
            logging.info("-" * 40)

            
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
                best_solution
            ])
            self.csv_file.flush()  

        logging.info("\n=== Optimization Engine Finished ===")
        self.csv_file.close()
