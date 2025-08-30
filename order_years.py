import pandas as pd
import os

def order_csv_by_year(input_file="combined_weather_data.csv", output_file="combined_weather_data_ordered.csv"):
    """
    Order the combined weather data CSV file by the year column
    """
    try:
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Check if 'year' column exists
        if 'year' not in df.columns:
            print("Error: 'year' column not found in the CSV file.")
            print("Available columns:", df.columns.tolist())
            return
        
        # Sort the dataframe by year column
        print("Sorting data by year...")
        df_sorted = df.sort_values('year').reset_index(drop=True)
        
        # Save the sorted dataframe
        print(f"Saving sorted data to {output_file}...")
        df_sorted.to_csv(output_file, index=False)
        
        print(f"Successfully ordered data by year!")
        print(f"Original file: {len(df)} rows")
        print(f"Sorted file: {len(df_sorted)} rows")
        print(f"Year range: {df_sorted['year'].min()} to {df_sorted['year'].max()}")
        
        # Display first few rows
        print("\nFirst 5 rows of sorted data:")
        print(df_sorted.head())
        
        return df_sorted
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        print("Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

def order_and_clean_csv(input_file="combined_weather_data.csv", output_file="combined_weather_data_ordered.csv"):
    """
    Order by year and perform additional cleaning
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        if 'year' not in df.columns:
            print("Error: 'year' column not found.")
            return
        
        # Sort by year
        df_sorted = df.sort_values('year').reset_index(drop=True)
        
        # Remove duplicate rows if any
        initial_count = len(df_sorted)
        df_sorted = df_sorted.drop_duplicates()
        final_count = len(df_sorted)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate rows.")
        
        # Save the sorted and cleaned dataframe
        df_sorted.to_csv(output_file, index=False)
        
        print(f"Data successfully ordered and saved to {output_file}")
        print(f"Total records: {len(df_sorted)}")
        
        return df_sorted
        
    except Exception as e:
        print(f"Error: {e}")

# Additional utility function to check the data
def analyze_combined_data(file_path="combined_weather_data.csv"):
    """
    Analyze the combined data file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data analysis for {file_path}:")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    # First, let's check if the file exists and analyze it
    if os.path.exists("combined_weather_data.csv"):
        print("Analyzing the current combined data file...")
        analyze_combined_data()
        print("\n" + "="*50 + "\n")
        
        # Order the CSV by year
        ordered_data = order_csv_by_year()
        
        if ordered_data is not None:
            print("\n" + "="*50)
            print("Analysis of the ordered data:")
            analyze_combined_data("combined_weather_data_ordered.csv")
    else:
        print("combined_weather_data.csv not found in the current directory.")
        print("Please make sure the file exists before running this script.")