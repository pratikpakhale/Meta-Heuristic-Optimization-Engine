import os
import json
import csv

def convert_json_to_csv(json_file_path, base_csv_dir):
    # Load JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Ensure the base CSV directory exists
    os.makedirs(base_csv_dir, exist_ok=True)

    # Function to write data to CSV
    def write_to_csv(data_type, func_name, func_data):
        # Prepare the directory for each data type
        type_dir = os.path.join(base_csv_dir, data_type)
        os.makedirs(type_dir, exist_ok=True)

        # Prepare CSV file path
        csv_file_path = os.path.join(type_dir, f"{func_name}.csv")
        
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Get the list of algorithms
            algorithms = list(func_data.keys())
            
            # Assuming all algorithms have the same iteration structure
            iterations = range(1, len(next(iter(func_data.values()))) + 1)
            
            # Write header: Iteration, Algorithm1, Algorithm2, ...
            header = ['Iteration'] + algorithms
            writer.writerow(header)
            
            # Write data for iterations
            for iteration in iterations:
                row = [iteration]
                for algo in algorithms:
                    row.append(func_data[algo][iteration - 1])  # Subtract 1 to match 0-based index
                writer.writerow(row)

    # Convert each type of data into its own CSV file
    for data_type, data_content in data.items():
        # Get all unique function names across all algorithms
        all_functions = set()
        for algo_data in data_content.values():
            all_functions.update(algo_data.keys())

        for func_name in all_functions:
            func_data = {algo: data_content[algo].get(func_name, []) for algo in data_content}
            write_to_csv(data_type, func_name, func_data)

# Example usage
json_file_path = os.path.join('json_results', 'all_iterations_fitness.json')
csv_dir = 'organized_csv_results'
convert_json_to_csv(json_file_path, csv_dir)
