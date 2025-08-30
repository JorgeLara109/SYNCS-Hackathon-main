import pandas as pd
import os
import shutil
from pathlib import Path
import re

## Change comparison for 1.csv, only drop nas in columns that are not measured by rainfall days
# Period over which rainfall was measured (days)
# Days of accumulation of minimum temperature
# Days of accumulation of maximum temperature


def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # only if file_path ends with 1.csv, filter year >= 2015
    if file_path.endswith("1.csv") or file_path.endswith("2.csv") or file_path.endswith("3.csv") or file_path.endswith("4.csv"):
        df = df[df['Year'] >= 2015]
    
    df["index"] = range(len(df))


    # Define protected columns
    protected_columns = [
        'Period over which rainfall was measured (days)',
        'Days of accumulation of minimum temperature',
        'Days of accumulation of maximum temperature'
    ]
    
    # Find which protected columns actually exist in the dataframe
    existing_protected_cols = [col for col in protected_columns if col in df.columns]
    
    # Drop rows with missing values, but only for columns that are NOT protected
    if existing_protected_cols:
        # Create a subset of columns to check for NaN (all columns except protected ones)
        columns_to_check = [col for col in df.columns if col not in existing_protected_cols]
        df.dropna(subset=columns_to_check, inplace=True)
    else:
        # If no protected columns exist, drop NaN from all columns
        df.dropna(inplace=True)
    
    # Convert categorical variables to numerical using one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)

def process_elec_data():
    # Define paths
    base_dir = "SYNCS-Hackathon-main"
    input_base = os.path.join("ElecData", "data")
    output_base = os.path.join("PreprocessData", "elec")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)
    
    # Get all folder names in the input directory
    try:
        folder_names = [name for name in os.listdir(input_base) 
                       if os.path.isdir(os.path.join(input_base, name))]
        print(f"Found {len(folder_names)} folders: {folder_names}")
    except FileNotFoundError:
        print(f"Error: Directory '{input_base}' not found.")
        return
    
    # Create the same folders in the output directory
    for folder_name in folder_names:
        output_folder = os.path.join(output_base, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
    
    # Process files in each folder
    for folder_name in folder_names:
        input_folder = os.path.join(input_base, folder_name)
        output_folder = os.path.join(output_base, folder_name)
        
        print(f"\nProcessing folder: {folder_name}")
        
        # Get all CSV files in the input folder
        all_files = os.listdir(input_folder)
        
        # Filter files: 1.csv, 2.csv, 3.csv, 4.csv, and files matching the pattern
        target_files = []
        pattern = re.compile(r".*132_11kV FY\d{4}\.csv$")
        
        for file in all_files:
            if file.endswith('.csv'):
                if file in ['1.csv', '2.csv', '3.csv', '4.csv']:
                    target_files.append(file)
                elif pattern.match(file):
                    target_files.append(file)
        
        print(f"Found {len(target_files)} target files: {target_files}")
        
        # Process each target file
        for file in target_files:
            input_file_path = os.path.join(input_folder, file)
            output_file_path = os.path.join(output_folder, file)
            
            try:
                print(f"Processing: {file}")
                # Preprocess the data
                df_processed = preprocess_data(input_file_path)
                
                # Save the preprocessed data
                save_preprocessed_data(df_processed, output_file_path)
                print(f"Saved preprocessed data to: {output_file_path}")
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")

if __name__ == "__main__":
    process_elec_data()
    print("Data preprocessing completed!")