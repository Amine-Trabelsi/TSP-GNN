import random

# List of num_nodes values and file types
num_nodes_list = [10, 20, 30, 50, 100]
file_types = ['train', 'val', 'test']

# Output file names
output_files = {
    'train': 'combined_train.txt',
    'val': 'combined_val.txt',
    'test': 'combined_test.txt'
}

# Function to sample lines with a bias towards files with smaller num_nodes
def combine_files(file_type):
    input_files = [f"tsp{num_nodes}_{file_type}_concorde.txt" for num_nodes in num_nodes_list]
    combined_lines = []

    # Read all lines from the input files
    lines_by_file = []
    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            lines_by_file.append(lines)

    # Calculate weights based on num_nodes (smaller num_nodes get higher weight)
    total_lines = len(lines_by_file[0])  # Target number of lines (assuming all files have the same total lines)
    inverse_weights = [1 / num_nodes for num_nodes in num_nodes_list]
    total_weight = sum(inverse_weights)
    normalized_weights = [w / total_weight for w in inverse_weights]

    # Calculate the number of lines to take from each file
    lines_to_take = [int(total_lines * weight) for weight in normalized_weights]

    # Randomly sample lines from each file based on calculated weights
    for i, lines in enumerate(lines_by_file):
        sampled_lines = random.sample(lines, lines_to_take[i])
        combined_lines.extend(sampled_lines)

    # Shuffle combined lines to mix them
    random.shuffle(combined_lines)

    # Write combined lines to the output file
    with open(output_files[file_type], 'w') as f:
        f.writelines(combined_lines[:total_lines])

# Combine files for each type
for file_type in file_types:
    combine_files(file_type)

print("Files combined successfully with bias towards smaller num_nodes!")
