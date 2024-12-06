import pandas as pd
import os

def combine_csv_files(start_year=2012, end_year=2024, input_dir=".\\imbalance\\development\\data\\", output_file="combined_test_imb.csv"):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for year in range(start_year, end_year + 1):
        # Construct the filename
        filename = f"test_imb-{year}.csv"
        file_path = os.path.join(input_dir, filename)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping...")
            continue

        print(f"Processing file: {file_path}")
        # Load the current year's CSV
        current_df = pd.read_csv(file_path, parse_dates=["timestamp"])

        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, current_df], ignore_index=True)

        # Ensure the DataFrame is sorted by timestamp
        combined_df.sort_values(by="timestamp", inplace=True)

    # Save the combined DataFrame to a CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to: {output_file}")

# Run the function
combine_csv_files(start_year=2012, end_year=2023, output_file="combined_test_imb-training-data.csv")
combine_csv_files(start_year=2024, end_year=2024, output_file="combined_test_imb-test-data.csv")


def list_csv_files(root_dir="."):
    """
    Lists all .csv files in the root directory and its subdirectories.

    Parameters:
    - root_dir (str): The directory to start searching from. Defaults to the current directory.

    Returns:
    - None
    """
    print(f"Searching for .csv files in: {os.path.abspath(root_dir)}\n")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".csv"):
                full_path = os.path.join(dirpath, file)
                print(full_path)

# Call the function to list all CSV files
#list_csv_files()
