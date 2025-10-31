import pandas as pd
import os

# Check if DC file exists
dc_file = 'Updated D* profits calc/Updated DC daily outputs/2024_Updated_DC_daily_returns.csv'
if not os.path.exists(dc_file):
    print(f"DC file not found: {dc_file}")
    print("Please run the Updated DC profits calculator first.")
    exit()

dc_df = pd.read_csv(dc_file)
dc_df.columns = ['Date', 'DC_profit']
print(f"Loaded DC profits: {len(dc_df)} days")

# Process each year (2021, 2022, 2023)
for year in [2021, 2022, 2023]:
    print(f"\nProcessing {year}...")
    
    # Check if IDM file exists
    idm_file = f'Updated IDM profits calc/Updated IDM daily profits/{year}_Updated_IDM_daily_profits.csv'
    if not os.path.exists(idm_file):
        print(f"IDM file not found: {idm_file}")
        print("Please run the Updated IDM optimizer first.")
        continue
    
    idm_df = pd.read_csv(idm_file)
    idm_df.columns = ['Date', 'IDM_profit']
    print(f"Loaded {year} IDM profits: {len(idm_df)} days")
    
    # Create month-day columns for matching (ignore year)
    idm_df['month_day'] = pd.to_datetime(idm_df['Date']).dt.strftime('%m-%d')
    dc_df_temp = dc_df.copy()
    dc_df_temp['month_day'] = pd.to_datetime(dc_df_temp['Date']).dt.strftime('%m-%d')
    
    # Merge on month-day
    merged = idm_df.merge(dc_df_temp[['month_day', 'DC_profit']], on='month_day', how='inner')
    
    # Create simple day numbering (1 to 365)
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['Day'] = range(1, len(merged) + 1)
    merged = merged[['Day', 'IDM_profit', 'DC_profit']]
    
    # Check for missing values
    missing_data = merged[merged.isnull().any(axis=1)]
    
    if not missing_data.empty:
        print(f"Missing data found: {len(missing_data)} rows")
        merged_clean = merged.dropna()
        print(f"Clean data: {len(merged_clean)} days")
    else:
        print(f"All days matched successfully: {len(merged)} days")
        merged_clean = merged
    
    # Save merged results
    output_dir = 'Updated merged daily profits'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{year}_updated_merged_profits.csv')
    
    merged_clean.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

print("\nAll years merged successfully!")