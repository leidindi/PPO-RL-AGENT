import numpy as np
from scipy.interpolate import CubicSpline
import os
import pandas as pd
import glob

# Updated function with cleaning for invalid characters in dates
def create_inflation_correction_function_with_cleaning(csv_file_path):
    """
    Creates an inflation correction function with error handling and data cleaning.

    Args:
        csv_file_path (str): Path to the CSV file with inflation data.

    Returns:
        function: A function that adjusts values for inflation given a timestamp.
    """
    try:
        # Load the CSV file
        inflation_data = pd.read_csv(csv_file_path)
        
        # Clean invalid characters in the 'date' column
        inflation_data['date'] = inflation_data['date'].str.replace(r'\*', '', regex=True)
        
        # Convert the cleaned 'date' column to datetime
        inflation_data['date'] = pd.to_datetime(inflation_data['date'], format='%Y %B')
        inflation_data['rate'] = inflation_data['rate'] / 100  # Convert percentage to decimal

        # Ensure the data is sorted by date
        inflation_data = inflation_data.sort_values('date')

        # Compute monthly factors and cumulative factors
        inflation_data['monthly_factor'] = (1 + inflation_data['rate'])**(1/12)
        inflation_data['cumulative_factor'] = inflation_data['monthly_factor'].cumprod()

        # Convert dates to UNIX timestamps for interpolation
        inflation_data['unix_time'] = inflation_data['date'].apply(lambda x: x.timestamp())
        dates_unix = inflation_data['unix_time'].values
        cumulative_factors = inflation_data['cumulative_factor'].values

        # Create the continuous interpolation function
        inflation_function = CubicSpline(dates_unix, cumulative_factors)

        # Define the inflation adjustment function with range validation
        def adjust_for_inflation_safe(value, timestamp):
            """
            Adjusts a value for inflation with range validation.

            Args:
                value (float): Original value (e.g., today's value in euros).
                timestamp (pd.Timestamp): Timestamp to adjust to (accurate to the minute).

            Returns:
                float: Inflation-adjusted value.
            """
            # Convert timestamp to UNIX time in seconds
            
         
            unix_time = timestamp.timestamp()

            # Validate range
            min_date = inflation_data['date'].min().timestamp()
            max_date = inflation_data['date'].max().timestamp()
            if unix_time < min_date or unix_time > max_date:
                raise ValueError(f"Timestamp {timestamp} is out of range. Valid range is {min_date} to {max_date}.")

            # Get the correction factor for the given time
            correction_factor = inflation_function(unix_time)
            return value * correction_factor

        return adjust_for_inflation_safe

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{csv_file_path}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the inflation data: {e}")

def convert_timestamp_to_cyclical_features(timestamp):
    # --- Fraction of Month ---
    # Start of the current month
    start_of_month = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # End of the current month (next month start)
    # Using MonthEnd(1) to jump to the end of the current month
    end_of_month = (start_of_month + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59, microsecond=99)
    
    total_minutes_in_month = (end_of_month - start_of_month).total_seconds() / 60.0
    elapsed_minutes_in_month = (timestamp - start_of_month).total_seconds() / 60.0
    fraction_of_month = elapsed_minutes_in_month / total_minutes_in_month
    
    # --- Fraction of Week ---
    # dayofweek: Monday=0, Sunday=6
    day_of_week = timestamp.dayofweek
    hour = timestamp.hour
    minute = timestamp.minute
    second = timestamp.second
    microsecond = timestamp.microsecond
    
    # Total seconds in a week
    total_seconds_in_week = 7 * 24 * 3600.0
    # Seconds elapsed this week = (day_of_week * one_day) + hours + minutes + seconds
    elapsed_seconds_in_week = day_of_week*24*3600.0 + hour*3600.0 + minute*60.0 + second + microsecond/1_000_000.0
    fraction_of_week = elapsed_seconds_in_week / total_seconds_in_week
    
    # --- Fraction of Day ---
    # Total minutes in a day = 1440
    total_minutes_in_day = 24 * 60.0
    elapsed_minutes_in_day = hour*60.0 + minute + (second/60.0) + (microsecond/60_000_000.0)
    fraction_of_day = elapsed_minutes_in_day / total_minutes_in_day
    
    # Now apply the cyclical transformation
    # For month: full cycle is 12 months, but we now have a continuous fraction_of_month
    # which naturally resets every month. We'll treat fraction_of_month as the fraction
    # through the month's cycle. The sine encoding will ensure a smooth cycle from start to end of month.
    month_enc = np.sin(2*np.pi * fraction_of_month)
    
    # For day_of_week: fraction_of_week is a continuous fraction from 0 to 1 through the week
    day_of_week_enc = np.sin(2*np.pi * fraction_of_week)
    
    # For hour_of_day: fraction_of_day is a continuous fraction from 0 to 1 through the day
    hour_of_day_enc = np.sin(2*np.pi * fraction_of_day)

       
    return month_enc, day_of_week_enc, hour_of_day_enc

def apply_deinfl():
    inflation_csv_path = 'dutch-inflation-2004jan-2024okt.csv'
    # Example: Create the inflation adjustment function
    try:
        if not os.path.exists(inflation_csv_path):
            raise FileNotFoundError(f"The file at path '{inflation_csv_path}' does not exist.")
        print(f"File exists: {inflation_csv_path}")
    except FileNotFoundError as e:
        print(e)

    # Create the inflation adjustment function with cleaning
    deinflation = create_inflation_correction_function_with_cleaning(inflation_csv_path)
    
    # Step 1: Load the CSV file
    csv_file = 'combined_test_imb-training-data.csv'  # Replace with your file name
    df = pd.read_csv(csv_file)

    # Step 2: Define the columns to adjust and your deinflation function
    columns_to_adjust = ['high_feed_price','low_take_price','mid_price','imbalance_take_price','imbalance_feed_price']  # Replace with the columns you want to test
    timestamp_column = 'timestamp'  # Replace with the name of the timestamp column

    # Deinflation function is assumed to be already defined
    # Example: deinflation(value=100, timestamp=pd.Timestamp('2020-01-01 00:00:01'))

    # Step 3: Convert the timestamp column to Pandas datetime for easier processing
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    print("Timestamps successfully converted")
    # Step 4: Apply deinflation to the specific columns
    for column in columns_to_adjust:
        if column in df.columns:
            df[column] = df.apply(lambda row: deinflation(value=row[column], timestamp=row[timestamp_column]),axis=1)

        print("Column " + column + " is done")
    # Step 5: Save the corrected DataFrame to a new CSV file
    output_file = f"deinflated-{csv_file}"
    df.to_csv(output_file, index=False)

    print(f"Corrected file saved to {output_file}")

    
    
    
    #print(deinflation(100,pd.Timestamp(f'2020-01-01 00:00:01')))
    # Example: Adjust today's value (1 euro) to January 1, 2004, at 12:00 PM
    #vals = []
    #try:
    #    for year in range(2004,2024+1,1):
    #        for month in range(1,12+1,1):
    #            if 10 > month:
    #                month = "0" + str(month)
    #            example_timestamp = pd.Timestamp(f'{year}-{month}-01 00:00:00')  # Minute-level accuracy
    #            adjusted_value_cleaned = inflation_correction_function_cleaned(1.0, example_timestamp)
    #            print(adjusted_value_cleaned)
    #            vals.append(adjusted_value_cleaned)
    #except ValueError as e:
    #    print(e)
    #import plotly.graph_objects as go
    # Create a Plotly figure
    #fig = go.Figure()

    # Add a line plot
    #fig.add_trace(go.Scatter(y=vals, mode='lines', name='Line Plot'))

    # Show the figure
    #fig.show()

def apply_date_conv():
    # Step 1: Load the CSV file
    type_of_data = "training"
    csv_file = f"deinflated-combined_test_imb-{type_of_data}-data.csv"  # Replace with your file name
    df = pd.read_csv(csv_file) 
    timestamp_column = 'timestamp'  # Replace with the name of the timestamp column

    # Step 3: Convert the timestamp column to Pandas datetime for easier processing
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    print("Timestamps successfully converted")

    # Instead of storing or expanding the file with more columns, override the timestamp column.
    # Extract cyclical features and overwrite the timestamp column and replace it with our three cyclical columns.
    df['month'], df['day_of_week'], df['hour_of_day'] = zip(*df[timestamp_column].apply(convert_timestamp_to_cyclical_features))

    # Step 5: Save the corrected DataFrame to a new CSV file
    output_file = f"final-imbalance-data-{type_of_data}.csv"
    df.to_csv(output_file, index=False)

    print(f"Corrected file saved to {output_file}")

    import time
    
    # Initialize dual progress bars
    total_scenarios = 5
    steps_per_scenario = 100
    progress = DualProgress(total_scenarios, steps_per_scenario)
    
    # Simulate training process
    for scenario in range(total_scenarios):
        for step in range(steps_per_scenario + 1):
            # Simulate some metrics
            metrics = {
                'reward': (scenario * steps_per_scenario + step) / (total_scenarios * steps_per_scenario) * 2,
                'loss': 1 - (scenario * steps_per_scenario + step) / (total_scenarios * steps_per_scenario)
            }
            
            # Update progress bars
            progress.update(scenario, step, metrics)
            
            # Simulate training time
            time.sleep(0.1)


def apply_reg_state_live_to_csv():
    csv_files = glob.glob('final-imbalance-data-sanity-test.csv')

    def add_reg_state_column(df):
        # Placeholder function - replace with your actual implementation
        return df

    # Process each file
    for file in csv_files:
        # Read the CSV
        df = pd.read_csv(file)
        
        if 'live_state' not in df.columns:
            print(f"Adding live_state column to {file}")
            df = add_reg_state_column(df)
        else:
            print(f"live_state column already exists in {file}")
        # Optionally save the processed file
        # output_file = file.replace('.csv', '_processed.csv')
        # df.to_csv(output_file, index=False)

    print(f"Processed {len(csv_files)} files")

if __name__ == '__main__':
    apply_reg_state_live_to_csv()