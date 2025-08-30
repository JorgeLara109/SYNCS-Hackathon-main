import os
import shutil

# Define the source and target directories
source_dir = 'PreprocessData/elec'
target_dir = 'html_graphs/Elec'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get list of folders in the source directory
try:
    source_folders = [f for f in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, f))]
    
    # Create corresponding folders in target directory
    for folder_name in source_folders:
        target_path = os.path.join(target_dir, folder_name)
        os.makedirs(target_path, exist_ok=True)
        print(f"Created folder: {target_path}")
        
    print(f"Successfully created {len(source_folders)} folders in {target_dir}")
    
except FileNotFoundError:
    print(f"Source directory '{source_dir}' not found")
except PermissionError:
    print(f"Permission denied to access '{source_dir}' or create folders in '{target_dir}'")