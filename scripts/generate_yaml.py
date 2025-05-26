import os
import yaml

# Set root directory path
root_dir = '/home/handsomexd/EventAD/data/detector/ROL'

# Define subdirectory paths
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')

def get_folder_names(directory):
    """Get a list of folder names in the specified directory, converting numeric names to strings"""
    folder_names = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            # If folder name is purely numeric, convert to string format
            if folder.isdigit():
                folder_names.append(f'{folder}')
            else:
                folder_names.append(folder)
    return sorted(folder_names)

# Get folder names for train, val, test
train_folders = get_folder_names(train_dir)
val_folders = get_folder_names(val_dir)
test_folders = get_folder_names(test_dir)

# Create a dictionary to store train, val, test folder names
data = {
    'train': train_folders,
    'val': val_folders,
    'test': test_folders
}

# Output to YAML file
yaml_file_path = '/home/handsomexd/EventAD/config/rol_split.yaml'
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

print(f"YAML file has been saved to: {yaml_file_path}")