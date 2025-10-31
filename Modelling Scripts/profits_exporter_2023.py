"""
BESS Profits Data Exporter (2023 Base Case)
Exports underlying daily profits data for DAM, IDM, and DC markets
Includes means, variances, and correlations for each BESS configuration
Does NOT include optimization/allocation results
"""

import pandas as pd
import numpy as np
import cplex
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# BESS Configuration Parameters - Selected configs only
BESS_CONFIGS = [
    {"config_id": 1, "power_mw": 10, "energy_mwh": 10},
    {"config_id": 5, "power_mw": 10, "energy_mwh": 15},
    {"config_id": 8, "power_mw": 10, "energy_mwh": 20},
    {"config_id": 16, "power_mw": 10, "energy_mwh": 30},
    {"config_id": 17, "power_mw": 15, "energy_mwh": 60}
]

class BESSProfitsExporter2023:
    def __init__(self, base_path):
        self.base_path = base_path
        
    def parse_datetime(self, date_string):
        """Parse datetime from CSV - data is already in UK time"""
        return datetime.strptime(date_string, '%d/%m/%Y %H:%M')

    def optimize_daily_arbitrage_cplex(self, prices, power_capacity_mw=5, energy_capacity_mwh=10, 
                                      max_cycles_per_day=1.0, efficiency=0.86):
        """Optimize battery operation for a single day using CPLEX"""
        solver = cplex.Cplex()
        solver.objective.set_sense(solver.objective.sense.maximize)
        solver.set_log_stream(None)
        solver.set_error_stream(None)
        solver.set_warning_stream(None)
        solver.set_results_stream(None)

        n_hours = len(prices)
        charge_names = [f'charge_{t}' for t in range(n_hours)]
        discharge_names = [f'discharge_{t}' for t in range(n_hours)]
        soc_names = [f'soc_{t}' for t in range(n_hours)]
        
        solver.variables.add(names=charge_names, lb=[0]*n_hours, ub=[power_capacity_mw]*n_hours)
        solver.variables.add(names=discharge_names, lb=[0]*n_hours, ub=[power_capacity_mw]*n_hours)
        solver.variables.add(names=soc_names, lb=[0]*n_hours, ub=[energy_capacity_mwh]*n_hours)

        obj_coeffs = []
        for t in range(n_hours):
            obj_coeffs.append((charge_names[t], -prices[t]))
            obj_coeffs.append((discharge_names[t], prices[t]))
            obj_coeffs.append((soc_names[t], 0))
        solver.objective.set_linear(obj_coeffs)

        for t in range(n_hours):
            if t == 0:
                solver.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        [charge_names[t], discharge_names[t], soc_names[t]], 
                        [np.sqrt(efficiency), -1/np.sqrt(efficiency), -1]
                    )],
                    senses=["E"], rhs=[0]
                )
            else:
                solver.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        [soc_names[t-1], charge_names[t], discharge_names[t], soc_names[t]], 
                        [1, np.sqrt(efficiency), -1/np.sqrt(efficiency), -1]
                    )],
                    senses=["E"], rhs=[0]
                )
        
        solver.linear_constraints.add(
            lin_expr=[cplex.SparsePair(discharge_names, [1.0]*n_hours)],
            senses=["L"], rhs=[max_cycles_per_day * energy_capacity_mwh]
        )

        solver.linear_constraints.add(
            lin_expr=[cplex.SparsePair([soc_names[-1]], [1.0])],
            senses=["E"], rhs=[0]
        )

        solver.solve()

        if solver.solution.get_status() in [1, 101, 102]:
            return solver.solution.get_objective_value()
        else:
            return None

    def calculate_dam_profits(self, power_mw, energy_mwh):
        """Calculate DAM daily profits for 2023"""
        csv_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/2023 DAM IDM Price.csv'
        df = pd.read_csv(csv_file)
        df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
        df['price'] = df['DAM price']
        df = df.dropna(subset=['price'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        daily_groups = df.groupby(df.index.date)
        daily_profits = []
        
        for date, group in daily_groups:
            prices = group['price'].values
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                daily_profits.append({'Date': date, 'DAM_Daily_Profit': round(profit, 2)})
        
        return pd.DataFrame(daily_profits)

    def calculate_idm_profits(self, power_mw, energy_mwh):
        """Calculate IDM daily profits for 2023 using actual IDM prices"""
        idm_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/2023 DAM IDM Price.csv'
        df = pd.read_csv(idm_file)
        df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
        df['price'] = df['IDM price']
        df = df.dropna(subset=['price'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        daily_groups = df.groupby(df.index.date)
        daily_profits = []
        
        for date, group in daily_groups:
            prices = group['price'].values
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                daily_profits.append({'Date': date, 'IDM_Daily_Profit': round(profit, 2)})
        
        return pd.DataFrame(daily_profits)

    def calculate_dc_profits(self, power_mw):
        """Calculate DC daily profits for 2023 using full capacity"""
        dc_bid_size = 0.8 * power_mw
        
        dc_path = f'{self.base_path}/Cleaned market prices/DC'
        dch_file = os.path.join(dc_path, 'DCH/2023 DCH Price.csv')
        dcl_file = os.path.join(dc_path, 'DCL/2023 DCL Price.csv')
        
        def calculate_dc_returns(csv_file, service_type):
            df = pd.read_csv(csv_file)
            df['EFA_Return'] = df['Clearing Price'] * 4 * dc_bid_size
            daily_returns = df.groupby('Date')['EFA_Return'].sum().reset_index()
            daily_returns.columns = ['Date', f'Daily_Return_{service_type}']
            return daily_returns.sort_values('Date')
        
        dch_returns = calculate_dc_returns(dch_file, 'DCH')
        dcl_returns = calculate_dc_returns(dcl_file, 'DCL')
        
        combined_returns = pd.merge(dch_returns, dcl_returns, on='Date', how='outer')
        combined_returns['DC_Daily_Profit'] = (
            combined_returns['Daily_Return_DCH'].fillna(0) + 
            combined_returns['Daily_Return_DCL'].fillna(0)
        ).round(1)
        
        return combined_returns[['Date', 'DC_Daily_Profit']]

    def export_baseline_profits(self, config):
        """Export baseline profits data for a single BESS configuration"""
        config_id = config['config_id']
        power_mw = config['power_mw']
        energy_mwh = config['energy_mwh']
        
        print(f"Exporting profits for Config {config_id}: {power_mw}MW/{energy_mwh}MWh")
        
        # Calculate profits for each market
        dam_df = self.calculate_dam_profits(power_mw, energy_mwh)
        idm_df = self.calculate_idm_profits(power_mw, energy_mwh)
        dc_df = self.calculate_dc_profits(power_mw)
        
        # Convert dates to datetime for merging
        dam_df['Date'] = pd.to_datetime(dam_df['Date'])
        idm_df['Date'] = pd.to_datetime(idm_df['Date'])
        dc_df['Date'] = pd.to_datetime(dc_df['Date'])
        
        # Merge all three markets
        merged = pd.merge(dam_df, idm_df, on='Date', how='inner')
        merged = pd.merge(merged, dc_df, on='Date', how='inner')
        
        # Sort by date and add Year/Day_of_Year columns
        merged = merged.sort_values('Date').reset_index(drop=True)
        merged['Year'] = merged['Date'].dt.year
        merged['Day_of_Year'] = merged['Date'].dt.dayofyear
        
        # Create export format: Year, Day_of_Year, DAM_Profit, IDM_Profit, DC_Profit
        export_df = merged[['Year', 'Day_of_Year', 'DAM_Daily_Profit', 'IDM_Daily_Profit', 'DC_Daily_Profit']].copy()
        export_df.columns = ['Year', 'Day_of_Year', 'DAM_Profit', 'IDM_Profit', 'DC_Profit']
        
        # Export to CSV
        output_dir = f'{self.base_path}/FINAL MODEL/Results'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'2023_ST profit (config {config_id}).csv'
        filepath = os.path.join(output_dir, filename)
        export_df.to_csv(filepath, index=False)
        
        print(f"  Daily profits saved to: {filename}")
        print(f"  Total days: {len(export_df)}")
        
        # Calculate statistics
        dam_returns = merged['DAM_Daily_Profit'].values
        idm_returns = merged['IDM_Daily_Profit'].values
        dc_returns = merged['DC_Daily_Profit'].values
        
        # Means
        dam_mean = np.mean(dam_returns)
        idm_mean = np.mean(idm_returns)
        dc_mean = np.mean(dc_returns)
        
        # Variances
        dam_var = np.var(dam_returns, ddof=1)  # Sample variance
        idm_var = np.var(idm_returns, ddof=1)
        dc_var = np.var(dc_returns, ddof=1)
        
        # Correlation matrix
        corr_matrix = np.corrcoef([dam_returns, idm_returns, dc_returns])
        
        # Covariance matrix
        cov_matrix = np.cov([dam_returns, idm_returns, dc_returns])
        
        # Create statistics summary
        stats_data = {
            'config_id': config_id,
            'power_mw': power_mw,
            'energy_mwh': energy_mwh,
            'dam_mean': round(dam_mean, 4),
            'idm_mean': round(idm_mean, 4),
            'dc_mean': round(dc_mean, 4),
            'dam_variance': round(dam_var, 4),
            'idm_variance': round(idm_var, 4),
            'dc_variance': round(dc_var, 4),
            'corr_dam_idm': round(corr_matrix[0,1], 4),
            'corr_dam_dc': round(corr_matrix[0,2], 4),
            'corr_idm_dc': round(corr_matrix[1,2], 4),
            'cov_dam_idm': round(cov_matrix[0,1], 4),
            'cov_dam_dc': round(cov_matrix[0,2], 4),
            'cov_idm_dc': round(cov_matrix[1,2], 4),
            'num_days': len(merged)
        }
        
        return export_df

    def export_all_configurations(self):
        """Export profits statistics for all BESS configurations"""
        print("Starting BESS Profits Statistics Export (2023 Data)")
        print("=" * 55)
        
        all_stats = []
        
        for config in BESS_CONFIGS:
            try:
                stats = self.export_baseline_profits(config)
                all_stats.append(stats)
            except Exception as e:
                print(f"  Error processing Config {config['config_id']}: {e}")
                continue
        
        # Export to FINAL MODEL/Results folder
        if all_stats:
            output_dir = f'{self.base_path}/FINAL MODEL/Results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Export statistics summary only
            stats_df = pd.DataFrame(all_stats)
            stats_file = os.path.join(output_dir, 'profits_statistics_summary_2023.csv')
            stats_df.to_csv(stats_file, index=False)
            
            print(f"\nProfits statistics saved to: {stats_file}")
            print(f"Total configurations processed: {len(all_stats)}")
            print("\nExported data includes:")
            print("- Mean profits for DAM, IDM, DC markets")
            print("- Variances for each market")
            print("- Correlation and covariance matrices")
            print("- Number of trading days used")
            print("- Daily profits data for Config 1 (10MW/10MWh)")
        
        return all_stats

def main():
    base_path = '/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration'
    exporter = BESSProfitsExporter2023(base_path)
    results = exporter.export_all_configurations()
    return results

if __name__ == "__main__":
    results = main()