
import pandas as pd
import os
import glob


# Example: Adjust pandas display settings
pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', 1000)         # Set the display width to avoid wrapping
pd.set_option('display.colheader_justify', 'left')  # Align headers to the left
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

def cutoff():
    # Print the current working directory
    print("Current working directory:", os.getcwd())

    # Load the CSV file into a DataFrame
    df = pd.read_csv('.\\imbalance\\development\\data\\test_imb-2012-2013.csv')

    # Parse the 'timestamp' column as datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Define the cutoff time
    cutoff_time = pd.Timestamp('2012-09-20 18:21:00')

    # Step 1: Identify the index of the cutoff period
    cutoff_index = df.index[df['timestamp'] >= cutoff_time][0]

    # Step 2: Copy the 'mid_price' column to a separate variable, dropping empty cells
    mid_price_safe = df['mid_price'].dropna().copy()

    # Step 3: Remove rows earlier than the cutoff timestamp
    df = df[df['timestamp'] >= cutoff_time]

    # Step 4: Assign the preserved 'mid_price' values back to the shortened DataFrame
    df['mid_price'] = mid_price_safe.values[:len(df)]

    # Save the resulting DataFrame to a new CSV
    df.to_csv('.\\imbalance\\development\\data\\test_imb-2012-2013.csv', index=False)

    print("Processed CSV saved as 'filtered_file.csv'")


def deduplicate():

    # Step 1: Locate and sort files
    file_pattern = ".\\imbalance\\development\\data\\test_imb-*.csv"
    files = sorted(glob.glob(file_pattern))  # Sorted chronologically

    # Step 2: Separate files for 2012, 2024, and 2013–2023
    files_2012_2024 = [f for f in files if '2012' in f or '2024' in f]
    files_2013_2023 = [f for f in files if f not in files_2012_2024]

    # Step 3: Read files for 2012 and 2024 directly
    data_2012_2024 = [pd.read_csv(file, parse_dates=['timestamp']) for file in files_2012_2024]

    # Step 4: Process and deduplicate files for 2013–2023
    data_2013_2023 = []
    for file in files_2013_2023:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, parse_dates=['timestamp'])  # Parse timestamp as datetime
        data_2013_2023.append(df)

    print("Concatenating and deduplicating 2013–2023 data...")
    combined_df_2013_2023 = pd.concat(data_2013_2023, ignore_index=True)

    # Deduplicate by timestamp, keeping the first occurrence
    deduplicated_df_2013_2023 = combined_df_2013_2023.drop_duplicates(subset='timestamp', keep='first')

    # Step 5: Check for discrepancies in overlapping timestamps
    print("Checking for discrepancies in overlapping timestamps...")
    overlap_logs = []
    duplicates = combined_df_2013_2023[combined_df_2013_2023.duplicated(subset='timestamp', keep=False)]
    print("Duplication set made...")
    i = 0
    for ts, group in duplicates.groupby('timestamp'):
        one = group.iloc[0:1]
        rest = group.iloc[1:]
        if not rest.equals(one):
            if i % 1 == 0 and i != 0:
                print(".",end=" ")
            if i % 10 == 0 and i != 0:
                print("",end="\n")
            if i % 100 == 0 and i != 0:
                print("-------------------------------------")
            i += 1 
            overlap_logs.append(f"Discrepancy at {ts}: Rows differ. \n1:{one.astype(str)}\n2:{rest.astype(str)}")
    # Log discrepancies if any
    if overlap_logs:
        with open("overlap_discrepancies.log", "w") as log_file:
            log_file.write("\n".join(overlap_logs))
        print(f'Discrepancies logged in overlap_discrepancies.log, in total {i} discrepancies found')
    else:
        print("No discrepancies found.")

    # Step 6: Combine all data
    print("Combining all data...")
    data_2012_2024_combined = pd.concat(data_2012_2024, ignore_index=True)
    final_combined_df = pd.concat([data_2012_2024_combined, deduplicated_df_2013_2023], ignore_index=True)

    # Combine all data
    final_combined_df = pd.concat([data_2012_2024_combined, deduplicated_df_2013_2023], ignore_index=True)

    # Sort by timestamp to ensure chronological order
    final_combined_df = final_combined_df.sort_values(by='timestamp').reset_index(drop=True)

    # Remove duplicates again (for safety) and ensure time range
    final_combined_df = final_combined_df.drop_duplicates(subset='timestamp', keep='first')


    # Step 7: Save to Feather file
    output_file = "test_imb-total.feather"
    print(f"Saving to {output_file}...")
    final_combined_df.to_feather(output_file)

    print("Process completed successfully!")
deduplicate()