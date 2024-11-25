
import numpy as np
from scipy.interpolate import CubicSpline
import os
import pandas as pd


inflation_csv_path = 'dutch-inflation-2004jan-2024okt.csv'
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
        inflation_data['monthly_factor'] = (1 + inflation_data['rate']) ** (1 / 12)
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
            return value / correction_factor

        return adjust_for_inflation_safe

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{csv_file_path}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the inflation data: {e}")

if __name__ == '__main__':

    # Example: Create the inflation adjustment function
    try:
        if not os.path.exists(inflation_csv_path):
            raise FileNotFoundError(f"The file at path '{inflation_csv_path}' does not exist.")
        print(f"File exists: {inflation_csv_path}")
    except FileNotFoundError as e:
        print(e)

    # Create the inflation adjustment function with cleaning
    inflation_correction_function_cleaned = create_inflation_correction_function_with_cleaning(inflation_csv_path)

    # Example: Adjust today's value (1 euro) to January 1, 2004, at 12:00 PM
    vals = []
    try:
        for year in range(2004,2024+1,1):
            for month in range(1,12+1,1):
                if 10 > month:
                    month = "0" + str(month)
                example_timestamp = pd.Timestamp(f'{year}-{month}-01 00:00:00')  # Minute-level accuracy
                adjusted_value_cleaned = inflation_correction_function_cleaned(1.0, example_timestamp)
                print(adjusted_value_cleaned)
                vals.append(adjusted_value_cleaned)
    except ValueError as e:
        print(e)
    import plotly.graph_objects as go
    # Create a Plotly figure
    fig = go.Figure()

    # Add a line plot
    fig.add_trace(go.Scatter(y=vals, mode='lines', name='Line Plot'))

    # Show the figure
    fig.show()

