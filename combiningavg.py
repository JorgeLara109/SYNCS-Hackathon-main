import os
import pandas as pd
import glob

def process_elec_data(base_dir):
    # Get all folders in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        print(f"Processing folder: {folder_path}")
        
        try:
            # Read the CSV files from the current folder
            df1 = pd.read_csv(os.path.join(folder_path, "1.csv"))  # Rainfall
            df2 = pd.read_csv(os.path.join(folder_path, "2.csv"))  # Solar exposure
            df3 = pd.read_csv(os.path.join(folder_path, "3.csv"))  # Min temp
            df4 = pd.read_csv(os.path.join(folder_path, "4.csv"))  # Max temp
            dfavg = pd.read_csv(os.path.join(folder_path, "averagemwh_by_year.csv"))

            # Filter data after 2015
            df1 = df1[df1['Year'] >= 2015]
            df2 = df2[df2['Year'] >= 2015]
            df3 = df3[df3['Year'] >= 2015]
            df4 = df4[df4['Year'] >= 2015]
            dfavg = dfavg[dfavg['year'] >= 2015]
            
            # Reset indices to ensure proper alignment
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
            df3 = df3.reset_index(drop=True)
            df4 = df4.reset_index(drop=True)
            dfavg = dfavg.reset_index(drop=True)
            
            # Extract only the specific columns we want
            year_col = df1['Year']
            rainfall_col = df1['Rainfall amount (millimetres)']
            solar_col = df2['Daily global solar exposure (MJ/m*m)']
            min_temp_col = df3['Minimum temperature (Degree C)']
            max_temp_col = df4['Maximum temperature (Degree C)']
            avg_mwh_col = dfavg['averagemwh']  # Assuming the column name is 'averagemwh'
            
            # Create the combined dataframe with exactly 6 columns
            combined_df = pd.DataFrame({
                'Year': year_col,
                'Rainfall': rainfall_col,
                'Solar_Exposure': solar_col,
                'Min_Temp': min_temp_col,
                'Max_Temp': max_temp_col,
                'Average_MWH': avg_mwh_col
            })
            
            # Save the combined file in the same folder
            output_path = os.path.join(folder_path, "combined_avg.csv")
            combined_df.to_csv(output_path, index=False)
            print(f"Saved combined file: {output_path}")
            
        except FileNotFoundError as e:
            print(f"Missing file in folder {folder}: {e}")
        except KeyError as e:
            print(f"Column not found in folder {folder}: {e}")
            # Print available columns for debugging
            try:
                print(f"Available columns in dfavg: {dfavg.columns.tolist()}")
            except:
                pass
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

# Usage example
if __name__ == "__main__":
    base_directory = "PreprocessData/elec"
    process_elec_data(base_directory)