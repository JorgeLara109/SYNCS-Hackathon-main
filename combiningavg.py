import os
import pandas as pd

def process_elec_data(base_dir):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        print(f"Processing folder: {folder_path}")
        
        try:
            # Read all CSV files
            df1 = pd.read_csv(os.path.join(folder_path, "1.csv"))  # Rainfall
            df2 = pd.read_csv(os.path.join(folder_path, "2.csv"))  # Solar exposure
            df3 = pd.read_csv(os.path.join(folder_path, "3.csv"))  # Min temp
            df4 = pd.read_csv(os.path.join(folder_path, "4.csv"))  # Max temp
            dfavg = pd.read_csv(os.path.join(folder_path, "averagemwh_by_year.csv"))

            # Filter data for years >= 2015
            dfs = [df1, df2, df3, df4, dfavg]
            for i, df in enumerate(dfs):
                year_col = 'year' if i == 4 else 'Year'  # dfavg uses 'year' instead of 'Year'
                dfs[i] = df[df[year_col] >= 2015].copy()

            # Rename dfavg's 'year' column to 'Year' for consistency
            dfs[4].rename(columns={'year': 'Year'}, inplace=True)

            # Merge all DataFrames on both 'Year' and 'index'
            merged_df = dfs[0][['Year', 'index', 'Rainfall amount (millimetres)']]
            for i, df in enumerate(dfs[1:], start=1):
                col_name = {
                    1: 'Daily global solar exposure (MJ/m*m)',
                    2: 'Minimum temperature (Degree C)',
                    3: 'Maximum temperature (Degree C)',
                    4: 'averagemwh'
                }[i]
                merged_df = merged_df.merge(
                    df[['Year', 'index', col_name]],
                    on=['Year', 'index'],
                    how='inner'  # Keep only matching rows
                )

            # Rename columns to desired names
            merged_df.columns = [
                'Year', 'index', 'Rainfall', 'Solar_Exposure', 
                'Min_Temp', 'Max_Temp', 'Average_MWH'
            ]
            merged_df['index'] = range(len(merged_df))
            merged_df = merged_df.dropna()
            # Save the combined file
            output_path = os.path.join(folder_path, "combined_avg.csv")
            merged_df.to_csv(output_path, index=False)
            print(f"Saved combined file: {output_path}")
            
        except FileNotFoundError as e:
            print(f"Missing file in folder {folder}: {e}")
        except KeyError as e:
            print(f"Column not found in folder {folder}: {e}")
            if 'dfavg' in locals():
                print(f"Available columns in dfavg: {dfavg.columns.tolist()}")
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

if __name__ == "__main__":
    base_directory = "PreprocessData/elec"
    process_elec_data(base_directory)