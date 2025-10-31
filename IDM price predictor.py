import pandas as pd
import os

# IDM prediction equation: IDM = 0.643*DAM + 0.26*IDM_lag + 5.22
dam_coeff = 0.643
idm_lag_coeff = 0.26
intercept = 5.22

# File paths
base_path = '/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration/Cleaned market prices/DAM & IDM'

years = [2021,2022, 2023] 

for year in years:
    print(f"Processing {year}...")
    
    # Load data
    df = pd.read_csv(f'{base_path}/{year} DAM IDM Price.csv')
    
    # Create datetime column for easier manipulation
    df['datetime'] = pd.to_datetime(df['Date (UK)'], format='%d/%m/%Y %H:%M')  # Changed format
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Initialize predicted IDM column
    df['Predicted_IDM'] = 0.0
    
    # Predict IDM prices
    for i in range(len(df)):
        dam_price = df.loc[i, 'DAM price']
        
        # For first 24 hours, use actual IDM as lag (no prediction possible)
        if i < 24:
            idm_lag = df.loc[i, 'IDM price']  # Use actual IDM
        else:
            idm_lag = df.loc[i-24, 'Predicted_IDM']  # Use predicted IDM from 24 hours ago
        
        # Apply prediction equation
        predicted_idm = dam_coeff * dam_price + idm_lag_coeff * idm_lag + intercept
        df.loc[i, 'Predicted_IDM'] = round(predicted_idm, 2)
    
    # Create output with Date and Predicted IDM only
    output_df = df[['Date (UK)', 'Predicted_IDM']].copy()
    output_df.columns = ['Date', 'Predicted_IDM_Price']
    
    # Save to CSV
    output_dir = 'IDM Price Predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{year}_predicted_IDM_prices.csv'
    output_df.to_csv(output_file, index=False)
    
    print(f"Saved {year} predictions: {len(output_df)} hours predicted")
    print(f"File: {output_file}")