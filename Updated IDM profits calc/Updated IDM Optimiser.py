import pandas as pd
import numpy as np
import cplex
import os
from datetime import datetime

# Function to parse datetime from CSV (already in UK time)
def parse_datetime(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M')

# Function for optimising energy arbitrage
def optimize_daily_arbitrage_cplex(prices, power_capacity_mw=5, energy_capacity_mwh=10, 
                                  max_cycles_per_day=1.0, efficiency=1.0):

    solver = cplex.Cplex()
    solver.objective.set_sense(solver.objective.sense.maximize)
    
    # Suppress output 
    solver.set_log_stream(None)
    solver.set_error_stream(None)
    solver.set_warning_stream(None)
    solver.set_results_stream(None)

    n_hours = len(prices)

    # Creating decision variables 
    charge_names = [f'charge_{t}' for t in range(n_hours)]
    discharge_names = [f'discharge_{t}' for t in range(n_hours)]
    soc_names = [f'soc_{t}' for t in range(n_hours)]
    
    # Add variables to solver
    solver.variables.add(names=charge_names, lb=[0]*n_hours, ub=[power_capacity_mw]*n_hours)
    solver.variables.add(names=discharge_names, lb=[0]*n_hours, ub=[power_capacity_mw]*n_hours)
    solver.variables.add(names=soc_names, lb=[0]*n_hours, ub=[energy_capacity_mwh]*n_hours)

    # set matrix coefficients for the objective function
    obj_coeffs = []
    for t in range(n_hours):
        obj_coeffs.append((charge_names[t], -prices[t]))  # charge costs money
        obj_coeffs.append((discharge_names[t], prices[t]))  # discharge earns money
        obj_coeffs.append((soc_names[t], 0))  # soc has no direct cost

    solver.objective.set_linear(obj_coeffs)

    # 1. SOC constraint (in matrix form for cplex)
    for t in range(n_hours):
        if t == 0:
            # soc[0] = charge[0] * sqrt(eff) - discharge[0] / sqrt(eff)
            solver.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    [charge_names[t], discharge_names[t], soc_names[t]], 
                    [np.sqrt(efficiency), -1/np.sqrt(efficiency), -1]
                )],
                senses=["E"],
                rhs=[0]
            )
        else:
            # soc[t] = soc[t-1] + charge[t] * sqrt(eff) - discharge[t] / sqrt(eff)
            solver.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    [soc_names[t-1], charge_names[t], discharge_names[t], soc_names[t]], 
                    [1, np.sqrt(efficiency), -1/np.sqrt(efficiency), -1]
                )],
                senses=["E"],
                rhs=[0]
            )
    
    # 2. Cannot charge and discharge simultaneously
    # We'll rely on the optimizer to not do both when it's not profitable
    
    # 3. Cycle limit 
    solver.linear_constraints.add(
        lin_expr=[cplex.SparsePair(discharge_names, [1.0]*n_hours)],
        senses=["L"],
        rhs=[max_cycles_per_day * energy_capacity_mwh]
    )

    # 4. End SOC should be 0 or equivalent to initial SOC
    solver.linear_constraints.add(
        lin_expr=[cplex.SparsePair([soc_names[-1]], [1.0])],
        senses=["E"],
        rhs=[0]
    )

    # Solve the problem
    solver.solve()

    # Check if solution is optimal
    if solver.solution.get_status() in [1, 101, 102]:  # Optimal solutions
        schedule = {
            'charge': [solver.solution.get_values(charge_names[t]) for t in range(n_hours)],
            'discharge': [solver.solution.get_values(discharge_names[t]) for t in range(n_hours)],
            'soc': [solver.solution.get_values(soc_names[t]) for t in range(n_hours)],
            'net_position': [
                solver.solution.get_values(discharge_names[t]) - solver.solution.get_values(charge_names[t])
                for t in range(n_hours)
            ]
        }
        return solver.solution.get_objective_value(), schedule
    else:
        print(f"Optimization failed with status: {solver.solution.get_status()}")
        return None, None

def analyze_battery_arbitrage(csv_file, power_capacity_mw=1.0, energy_capacity_mwh=1.0, 
                             max_cycles_per_day=1.0, efficiency=1.0):
    """
    Analyze battery arbitrage opportunities across all days in the dataset.
    
    Parameters:
    - csv_file: path to the CSV file with power prices
    - power_capacity_mw: maximum charge/discharge power in MW
    - energy_capacity_mwh: battery energy capacity in MWh
    - max_cycles_per_day: maximum number of full cycles per day
    - efficiency: round-trip efficiency (1.0 = 100%)
    
    Returns:
    - results: dict with analysis results
    """
    
    # read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")

    # parse the datetime and price columns - data already in UK time
    df['datetime'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
    df['price'] = df['Predicted_IDM_Price']  # Use predicted IDM price column
    #remove rows with NaN prices
    df = df.dropna(subset=['price'])
    print(f"\nAfter removing NaN prices: {len(df)} rows")
    # filter to keep only hourly data (remove 30-minute rows)
    df = df[df['datetime'].dt.minute == 0]
    print(f"\nAfter filtering to hourly data: {len(df)} rows")
    # set datetime as index
    df.set_index('datetime', inplace=True)
    # group by date
    daily_groups = df.groupby(df.index.date)
    print(f"\nNumber of unique dates: {len(daily_groups)}")

    daily_profits = []
    daily_schedules = []
    incomplete_days = []
    
    for date, group in daily_groups:
        prices = group['price'].values
        if len(prices) != 24:
            incomplete_days.append((date, len(prices)))
            continue
            
        profit, schedule = optimize_daily_arbitrage_cplex(
            prices, 
            power_capacity_mw=power_capacity_mw,
            energy_capacity_mwh=energy_capacity_mwh,
            max_cycles_per_day=max_cycles_per_day,
            efficiency=efficiency
        )
        
        if profit is not None:
            daily_profits.append(profit)
            daily_schedules.append({
                'date': date,
                'profit': profit,
                'schedule': schedule,
                'prices': prices
            })

    print(f"\nIncomplete days (not 24 hours): {len(incomplete_days)}")
    if incomplete_days:
        print("First few incomplete days:")
        for date, hours in incomplete_days[:5]:
            print(f"  {date}: {hours} hours")
            
    print(f"\nSuccessfully optimized days: {len(daily_profits)}")
    
    if len(daily_profits) == 0:
        print("\nWARNING: No complete 24-hour days found in the data!")
        return {
            'daily_profits': [],
            'average_daily_profit': 0,
            'std_daily_profit': 0,
            'min_daily_profit': 0,
            'max_daily_profit': 0,
            'total_days_analyzed': 0,
            'annual_profit_estimate': 0,
            'daily_schedules': [],
            'parameters': {
                'power_capacity_mw': power_capacity_mw,
                'energy_capacity_mwh': energy_capacity_mwh,
                'max_cycles_per_day': max_cycles_per_day,
                'efficiency': efficiency
            }
        }
        
    results = {
        'daily_profits': daily_profits,
        'average_daily_profit': np.mean(daily_profits),
        'std_daily_profit': np.std(daily_profits),
        'min_daily_profit': np.min(daily_profits),
        'max_daily_profit': np.max(daily_profits),
        'total_days_analyzed': len(daily_profits),
        'annual_profit_estimate': np.mean(daily_profits) * 365,
        'daily_schedules': daily_schedules,
        'parameters': {
            'power_capacity_mw': power_capacity_mw,
            'energy_capacity_mwh': energy_capacity_mwh,
            'max_cycles_per_day': max_cycles_per_day,
            'efficiency': efficiency
        }
    }
    return results

def print_summary(results):
    print("\n" + "="*50)
    print("Battery Arbitrage Analysis Summary")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  Power Capacity: {results['parameters']['power_capacity_mw']} MW")
    print(f"  Energy Capacity: {results['parameters']['energy_capacity_mwh']} MWh")
    print(f"  Max Cycles/Day: {results['parameters']['max_cycles_per_day']}")
    print(f"  Efficiency: {results['parameters']['efficiency']*100:.1f}%")
    print(f"\nResults:")
    print(f"  Days Analyzed: {results['total_days_analyzed']}")
    print(f"  Average Daily Profit: £{results['average_daily_profit']:.2f}")
    print(f"  Std Dev: £{results['std_daily_profit']:.2f}")
    print(f"  Min Daily Profit: £{results['min_daily_profit']:.2f}")
    print(f"  Max Daily Profit: £{results['max_daily_profit']:.2f}")
    print(f"  Estimated Annual Profit: £{results['annual_profit_estimate']:.2f}")

def export_daily_profits_to_csv(results, output_file='daily_profits.csv'):
    """Export daily profits to CSV file in same directory as script"""
    if len(results['daily_schedules']) == 0:
        print("No data to export.")
        return
    
    # Get script directory and create output folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Updated IDM daily profits')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    daily_data = []
    for schedule in results['daily_schedules']:
        daily_data.append({
            'Date': schedule['date'],
            'IDM_Daily_Profit': round(schedule['profit'], 2)
        })
    df = pd.DataFrame(daily_data)
    df.to_csv(output_path, index=False)
    print(f"\nDaily profits exported to: {output_path}")
    print(f"Total days exported: {len(daily_data)}")
    return df

if __name__ == "__main__":
    # Load DAM weight from Baseline optimisation results
    base_path = '/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration'
    baseline_results = pd.read_csv(f'{base_path}/1st Markowitz/Baseline Markowitz outputs/Baseline optimisation results.csv')
    
    # Get DAM weight from first row
    dam_weight = baseline_results.iloc[0]['DAM_Weight']
    
    # Calculate IDM battery capacity (remainder after DAM allocation)
    total_capacity = 10  # MW and MWh
    idm_power = total_capacity * (1 - dam_weight)
    idm_energy = total_capacity * (1 - dam_weight)
    
    print(f"DAM weight: {dam_weight:.3f}")
    print(f"IDM capacity: {idm_power:.2f} MW / {idm_energy:.2f} MWh")
    
    # Process years 2021-2023
    for year in [2021, 2022, 2023]:
        print(f"\nProcessing {year}...")
        input_file = f'{base_path}/IDM Price Predictions/{year}_predicted_IDM_prices.csv'
        
        results = analyze_battery_arbitrage(
            input_file, idm_power, idm_energy, 1.0, 1.0
        )
        print_summary(results)
        export_daily_profits_to_csv(results, f'{year}_Updated_IDM_daily_profits.csv')