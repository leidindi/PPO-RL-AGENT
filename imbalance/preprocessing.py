
import os
import datetime
import pytz
import xarray as xr
from pathlib import Path
from tennet import TenneTClient, DataType, OutputType
import zarr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Define the root directory
data_folder = ".\\imbalance\\development\\data\\"

def preprocess_data(minute_prices, settlement_prices, mid_prices,year):
    """
    Preprocess the loaded data and combine minute imbalance with settlement prices.

    Args:
        minute_prices (DataFrame): DataFrame of minute imbalance data.
        settlement_prices (DataFrame): DataFrame of settlement prices.

    Returns:
        DataFrame: Combined and processed DataFrame.
    """
    # Replace commas with dots and convert relevant columns to numeric
    minute_prices = minute_prices.replace({',': '.'}, regex=True)
    settlement_prices = settlement_prices.replace({',': '.'}, regex=True)
    mid_prices = mid_prices.replace({',': '.'}, regex=True)

    # Select and rename relevant columns in minute_prices
    minute_prices = minute_prices.rename(columns={
        'datum': 'date',
        'times': 'time',
        'Highest_price_upward': 'high_feed_price',
        'Mid_prijs_opregelen': 'mid_price',
        'Lowest_price_downward': 'low_take_price',
        'Unnamed: 0' : 'index'
    })

       
    # Ensure numeric conversion
    for col in ['high_feed_price', 'mid_price', 'low_take_price']:
        try:
            minute_prices[col] = pd.to_numeric(minute_prices[col])
        except:
            minute_prices[col] = pd.to_numeric(mid_prices['mid-price'])

    for col in ['Consume', 'Feed', 'Regulation state']:
        settlement_prices[col] = pd.to_numeric(settlement_prices[col])

    # Add additional columns
    minute_prices['timestamp'] = pd.to_datetime(minute_prices['timestamp'])
    minute_prices['month'] = minute_prices['timestamp'].dt.month
    minute_prices['day_of_week'] = minute_prices['timestamp'].dt.dayofweek
    minute_prices['hour_of_day'] = minute_prices['timestamp'].dt.hour

    # Combine the data
    j = -1

    consumeArray = []
    midArray = []
    feedArray = []
    regArray = []

    consumeFloat = 0
    midFloat = 0
    feedFloat = 0
    regFloat = 0

    for i in range(minute_prices.shape[0]):
        if i % 15 == 0:
            j += 1
            consumeFloat = settlement_prices['Consume'][j]
            midFloat = minute_prices['mid_price'][i]
            feedFloat = settlement_prices['Feed'][j]
            regFloat = settlement_prices['Regulation state'][j]

        consumeArray.append(consumeFloat) 
        midArray.append(midFloat) 
        feedArray.append(feedFloat) 
        regArray.append(regFloat) 

        if i % 25000 == 0 and i != 0:
            print(".",end=" ")
        if i % 100000 == 0 and i != 0:
            print("",end="\n")

        
    minute_prices['imbalance_take_price'] = consumeArray
    minute_prices['mid_price'] = midArray
    minute_prices['imbalance_feed_price'] = feedArray 
    minute_prices['imbalance_regulation_state'] = regArray


    return minute_prices

def save_combined(df, year, output_folder):
    """
    Save the processed DataFrame

    Args:
        df (DataFrame): Processed DataFrame.
        year (str): Year range for the output file.
        output_folder (str): Path to the output folder.
    """
    output_file = os.path.join(output_folder, f'test_imb-{year}.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}") 

# Main execution
def main():
    output_folder = ".\\imbalance\\development\\data\\"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for year in range(2012, 2025):
        print(f"Processing year: {year}")
        minute_prices =     pd.read_csv(os.path.join(data_folder, f'minute_imbalance_data-{year}.csv'))
        settlement_prices = pd.read_csv(os.path.join(data_folder, f'settlement_prices-{year}.csv'))
        mid_prices =        pd.read_csv(os.path.join(data_folder, f'imbalance_igcc-{year}.csv'))
        
        # Preprocess and combine data
        combined_data = preprocess_data(minute_prices, settlement_prices, mid_prices , year)
        save_combined(combined_data, year, output_folder)
if __name__ == "__main__":
    main()
