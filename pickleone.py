import json
import pickle
import os

# Set dataset paths
data_dir = "leaf/data/femnist/data"
train_path = os.path.join(data_dir, "train/all_data_0.json")
test_path = os.path.join(data_dir, "test/all_data_0.json")
output_file = "FEMNIST_full_partitioned_by_authors.pickle"

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_data = load_json(train_path)
test_data = load_json(test_path)

# Combine training and testing data
full_data = {"train": train_data, "test": test_data}

# Save as pickle
with open(output_file, 'wb') as f:
    pickle.dump(full_data, f)

print(f"Dataset saved as {output_file}")
