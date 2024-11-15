import pandas as pd

def transform_csv(input_file, output_file):
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Add 1 to each element in "imbalance_regulation_state"
    #df['imbalance_regulation_state'] = df['imbalance_regulation_state'] + 1
    
    # Create one-hot encoded columns
    df['is_state_-1'] = (df['imbalance_regulation_state'] == -1).astype(int)
    df['is_state_0'] = (df['imbalance_regulation_state'] == 0).astype(int)
    df['is_state_1'] = (df['imbalance_regulation_state'] == 1).astype(int)
    df['is_state_2'] = (df['imbalance_regulation_state'] == 2).astype(int)
    
    # Drop the original "imbalance_regulation_state" column
    df.drop(columns=['imbalance_regulation_state'], inplace=True)
    
    # Save the transformed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_csv = 'imb_clean.csv'  # Replace with your input CSV file path
output_csv = 'imb_clean_one_hot.csv'  # Replace with your desired output CSV file path
transform_csv(input_csv, output_csv)
