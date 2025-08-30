import os
import pandas as pd
import glob
import re

def process_elec_data():
    # Define the base directory
    base_dir = "PreprocessData/elec"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Get all subdirectories in the elec folder
    subdirectories = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirectories:
        subdir_path = os.path.join(base_dir, subdir)
        print(f"Processing directory: {subdir}")
        
        # Find all CSV files matching the pattern
        pattern = os.path.join(subdir_path, "* 132_11kV FY*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"  No matching CSV files found in {subdir}")
            continue
        
        # List to store all processed data
        all_data = []
        
        for csv_file in csv_files:
            try:
                # Extract year from filename using regex
                filename = os.path.basename(csv_file)
                year_match = re.search(r'FY(\d{4})\.csv$', filename)
                
                if not year_match:
                    print(f"  Could not extract year from filename: {filename}")
                    continue
                
                year = int(year_match.group(1))
                
                # Only process files from 2015 onwards
                if year < 2015:
                    print(f"  Skipping {filename} (year {year} < 2015)")
                    continue
                
                print(f"  Processing {filename} (year {year})")
                
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if 'averagemwh' column exists
                if 'averagemwh' not in df.columns:
                    print(f"  'averagemwh' column not found in {filename}")
                    continue
                
                # Extract the averagemwh column and add year
                data = df[['averagemwh']].copy()
                data['year'] = year
                
                all_data.append(data)
                
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")
                continue
        
        if not all_data:
            print(f"  No valid data found for {subdir}")
            continue
        
        # Combine all data from different years
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by year (optional)
        combined_data = combined_data.sort_values('year')
        combined_data["index"] = range(len(combined_data))
        # Save the new CSV file
        output_filename = os.path.join(subdir_path, "averagemwh_by_year.csv")
        combined_data.to_csv(output_filename, index=False)
        print(f"  Saved combined data to {output_filename}")
        
        # Print summary
        print(f"  Processed {len(all_data)} files with {len(combined_data)} total rows")

if __name__ == "__main__":
    process_elec_data()
    print("Processing complete!")