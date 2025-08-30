import pandas as pd
from pathlib import Path

# Base directory
base_dir = Path("PreprocessData/elec")

# Iterate through all subfolders
for folder in base_dir.iterdir():
    if folder.is_dir():
        # Iterate through all CSV files in the folder
        for csv_file in folder.glob("*132_11kV FY[0-9][0-9][0-9][0-9].csv"):
            print(f"Processing {csv_file}")
            
            # Read the CSV
            df = pd.read_csv(csv_file)
            
            # Drop columns starting with 'Date'
            df = df.loc[:, ~df.columns.str.startswith("Date")]
            
            # Compute average for each row ignoring 'year'
            cols_to_avg = [col for col in df.columns if col != "year"]
            df["averagemwh"] = df[cols_to_avg].mean(axis=1)
            
            # Save back to the same file (or you can save to a new folder)
            df.to_csv(csv_file, index=False)
