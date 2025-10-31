"""
D* price levele sensitivity with exports of underlying profits
"""

import pandas as pd
import numpy as np
import cplex
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# BESS Configuration Parameters
BESS_CONFIGS = [
    {"config_id": 1, "power_mw": 10, "energy_mwh": 10},
    {"config_id": 2, "power_mw": 30, "energy_mwh": 30},
    {"config_id": 3, "power_mw": 50, "energy_mwh": 50},
    {"config_id": 4, "power_mw": 100, "energy_mwh": 100},
    {"config_id": 5, "power_mw": 10, "energy_mwh": 15},
    {"config_id": 6, "power_mw": 50, "energy_mwh": 75},
    {"config_id": 7, "power_mw": 100, "energy_mwh": 150},
    {"config_id": 8, "power_mw": 10, "energy_mwh": 20},
    {"config_id": 9, "power_mw": 30, "energy_mwh": 60},
    {"config_id": 10, "power_mw": 50, "energy_mwh": 100},
    {"config_id": 11, "power_mw": 100, "energy_mwh": 200},
    {"config_id": 12, "power_mw": 200, "energy_mwh": 400},
    {"config_id": 13, "power_mw": 300, "energy_mwh": 600},
    {"config_id": 14, "power_mw": 400, "energy_mwh": 800},
    {"config_id": 15, "power_mw": 5, "energy_mwh": 15},
    {"config_id": 16, "power_mw": 10, "energy_mwh": 30},
    {"config_id": 17, "power_mw": 15, "energy_mwh": 60}
]

class BESSWorkflowController:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = []
        
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
        """Calculate DAM daily profits for 2024"""
        dam_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/2024 DAM IDM Price.csv'
        df = pd.read_csv(dam_file)
        df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
        df['price'] = df['DAM price']
        df = df.dropna(subset=['price'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        daily_groups = df.groupby(df.index.date)
        daily_profits = []
        
        for date, group in daily_groups:
            prices = group['price'].values
            # Remove 24-hour requirement - accept any number of hours
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                daily_profits.append({'Date': date, 'DAM_Daily_Profit': round(profit, 2)})
        
        return pd.DataFrame(daily_profits)


    def calculate_idm_profits(self, year, power_mw, energy_mwh, dam_weight):
        """Calculate IDM profits for 2024"""
        idm_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/2024 DAM IDM Price.csv'
        df = pd.read_csv(idm_file)
        df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
        df['price'] = df['IDM price']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        daily_groups = df.groupby(df.index.date)
        daily_profits = []
        
        for date, group in daily_groups:
            prices = group['Predicted_IDM'].values
            # Remove 24-hour requirement - accept any number of hours
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                daily_profits.append({'Date': date, f'IDM_Daily_Profit_{year}': round(profit, 2)})
        
        return pd.DataFrame(daily_profits)

    def calculate_dc_profits_with_scaling(self, power_mw, price_scale_factor=1.0):
        """Calculate DC profits using 2024 prices with scaling factor (full capacity for baseline)"""
        dc_bid_size = 0.8 * power_mw
        
        dc_path = f'{self.base_path}/Cleaned market prices/DC'
        dch_file = os.path.join(dc_path, 'DCH/2024 DCH Price.csv')
        dcl_file = os.path.join(dc_path, 'DCL/2024 DCL Price.csv')
        
        def calculate_dc_returns(csv_file, service_type):
            df = pd.read_csv(csv_file)
            # Apply price scaling factor
            df['Scaled_Price'] = df['Clearing Price'] * price_scale_factor
            df['EFA_Return'] = df['Scaled_Price'] * 4 * dc_bid_size
            daily_returns = df.groupby('Date')['EFA_Return'].sum().reset_index()
            daily_returns.columns = ['Date', f'Daily_Return_{service_type}']
            daily_returns[f'Daily_Return_{service_type}'] = daily_returns[f'Daily_Return_{service_type}'].round(1)
            return daily_returns.sort_values('Date')
        
        dch_returns = calculate_dc_returns(dch_file, 'DCH')
        dcl_returns = calculate_dc_returns(dcl_file, 'DCL')
        
        combined_returns = pd.merge(dch_returns, dcl_returns, on='Date', how='outer')
        combined_returns['Daily DC returns'] = (
            combined_returns['Daily_Return_DCH'].fillna(0) + 
            combined_returns['Daily_Return_DCL'].fillna(0)
        ).round(1)
        
        return combined_returns[['Date', 'Daily DC returns']]


    def export_baseline_profits(self, config):
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
        
        return stats_data
    
    def export_all_configurations(self):
        """Export profits statistics for all BESS configurations"""
        print("Starting BESS Profits Statistics Export (2024 Data)")
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
            stats_file = os.path.join(output_dir, 'profits_statistics_summary_2024.csv')
            stats_df.to_csv(stats_file, index=False)
            
            print(f"\nProfits statistics saved to: {stats_file}")
            print(f"Total configurations processed: {len(all_stats)}")
            print("\nExported data includes:")
            print("- Mean profits for DAM, IDM, DC markets")
            print("- Variances for each market")
            print("- Correlation and covariance matrices")
            print("- Number of trading days used")
        
        return all_stats

    def markowitz_optimization(self, mean_returns, cov_matrix, risk_aversion=3):
        """Perform Markowitz portfolio optimization using SciPy (working version)"""
        from scipy.optimize import minimize
        
        # Normalize returns and risks (mean=0, std=1) like the working test version
        returns_original = np.array(mean_returns)
        
        # Calculate standard deviations from covariance matrix diagonal
        risks_original = np.sqrt(np.diag(cov_matrix))
        
        # Calculate correlation matrix from covariance matrix
        correlation_matrix = cov_matrix / np.outer(risks_original, risks_original)
        
        # Normalize data
        returns = (returns_original - returns_original.mean()) / returns_original.std()
        risks = (risks_original - risks_original.mean()) / risks_original.std()
        
        # Rebuild covariance matrix with normalized data
        normalized_cov_matrix = np.outer(risks, risks) * correlation_matrix
        
        def calculate_portfolio_stats(weights):
            """Calculate return and risk for given weights"""
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(normalized_cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            return portfolio_return, portfolio_risk, portfolio_variance
        
        def objective_function(weights):
            """Utility function to maximize"""
            port_return, _, port_variance = calculate_portfolio_stats(weights)
            utility = port_return - 0.5 * risk_aversion * port_variance
            return -utility  # Minimize negative utility = maximize utility
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(len(mean_returns))]
        
        # Initial guess: equal weights
        initial_guess = [1.0/len(mean_returns)] * len(mean_returns)
        
        # Optimize
        result = minimize(objective_function, x0=initial_guess, bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            normalized_return, normalized_risk, _ = calculate_portfolio_stats(weights)
            
            # Convert back to original scale
            original_return = normalized_return * returns_original.std() + returns_original.mean()
            original_risk = normalized_risk * risks_original.std() + risks_original.mean()
            
            return {
                'weights': weights,
                'expected_return': original_return,
                'portfolio_risk': original_risk
            }
        return None

    def run_baseline_stage_with_dc_scaling(self, power_mw, energy_mwh, price_scale_factor=1.0, scenario_name="base"):
        """Run baseline stage (3-asset optimization using 2024 data) with DC price scaling"""
        # Calculate DAM profits
        dam_df = self.calculate_dam_profits(power_mw, energy_mwh)
        
        # Calculate baseline IDM profits (using actual 2024 IDM prices)
        idm_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/2024 DAM IDM Price.csv'
        df = pd.read_csv(idm_file)
        df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
        df['price'] = df['IDM price']
        df = df.dropna(subset=['price'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        daily_groups = df.groupby(df.index.date)
        idm_profits = []
        
        for date, group in daily_groups:
            prices = group['price'].values
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                idm_profits.append({'Date': date, 'IDM_Daily_Profit': round(profit, 2)})
        
        idm_df = pd.DataFrame(idm_profits)
        
        # Calculate DC profits with price scaling
        dc_df = self.calculate_dc_profits_with_scaling(power_mw, price_scale_factor)
        
        # Merge all three assets
        dam_df['Date'] = pd.to_datetime(dam_df['Date'])
        idm_df['Date'] = pd.to_datetime(idm_df['Date'])
        dc_df['Date'] = pd.to_datetime(dc_df['Date'])
        
        merged = pd.merge(dam_df, idm_df, on='Date', how='inner')
        merged = pd.merge(merged, dc_df, on='Date', how='inner')
        
        # Calculate statistics and optimize
        dam_returns = merged['DAM_Daily_Profit'].values
        idm_returns = merged['IDM_Daily_Profit'].values
        dc_returns = merged['Daily DC returns'].values
        
        mean_returns = np.array([np.mean(dam_returns), np.mean(idm_returns), np.mean(dc_returns)])
        cov_matrix = np.cov([dam_returns, idm_returns, dc_returns])
        
        result = self.markowitz_optimization(mean_returns, cov_matrix, risk_aversion=3)
        
        if result:
            return {
                'scenario': scenario_name,
                'price_scale_factor': price_scale_factor,
                'dam_weight': result['weights'][0],
                'idm_weight': result['weights'][1], 
                'dc_weight': result['weights'][2],
                'expected_return': result['expected_return'],
                'portfolio_risk': result['portfolio_risk']
            }
        return None


    def run_single_config_dc_price_levels(self, config):
        """Run baseline optimization across different DC price levels"""
        config_id = config['config_id']
        power_mw = config['power_mw']
        energy_mwh = config['energy_mwh']
        
        print(f"Processing Config {config_id}: {power_mw}MW/{energy_mwh}MWh")
        
        # Define price scaling scenarios
        price_scenarios = {
            'base_case': 1.0,
            'down_35%': 0.65,
            'down_40%': 0.6,
            'down_50%': 0.5,
            'down_45%': 0.55,
            'down 55%': 0.45,
            'down_60%': 0.4,
            'down_65%': 0.35,
            'down_70%': 0.3,
            'down_75%': 0.25,
            'down_80%': 0.2,
            'down_30%': 0.7,
            'down_25%': 0.75,
            'down_20%': 0.8,
            'down_15%': 0.85,
            'down_10%': 0.9,
            'down_5%': 0.95,
            'up_5%': 1.05,
            'up_10%': 1.1,
            'up_15%': 1.15,
            'up_25%': 1.25,
            'up_20%': 1.2,
            'up_30%': 1.3
        }
        
        config_results = []
        
        for scenario_name, scale_factor in price_scenarios.items():
            print(f"  Running {scenario_name} scenario (scale: {scale_factor})")
            
            baseline_result = self.run_baseline_stage_with_dc_scaling(
                power_mw, energy_mwh, scale_factor, scenario_name
            )
            
            if baseline_result:
                result = {
                    'config_id': config_id,
                    'power_mw': power_mw,
                    'energy_mwh': energy_mwh,
                    'scenario': scenario_name,
                    'dc_price_scale_factor': scale_factor,
                    'dam_weight': baseline_result['dam_weight'],
                    'idm_weight': baseline_result['idm_weight'],
                    'dc_weight': baseline_result['dc_weight'],
                    'expected_return': baseline_result['expected_return'],
                    'portfolio_risk': baseline_result['portfolio_risk']
                }
                config_results.append(result)
                print(f"    DAM: {result['dam_weight']:.3f}, IDM: {result['idm_weight']:.3f}, DC: {result['dc_weight']:.3f}")
            else:
                print(f"    {scenario_name} optimization failed")
        
        print(f"  Config {config_id} completed: {len(config_results)}/{len(price_scenarios)} scenarios successful")
        return config_results

    def run_all_configurations_dc_price_levels(self):
        """Run baseline optimization for all configurations across DC price levels"""
        print("Starting BESS DC Price Level Analysis (Baseline Allocations Only)")
        print("=" * 70)
        
        all_results = []
        
        for config in BESS_CONFIGS:
            config_results = self.run_single_config_dc_price_levels(config)
            if config_results:
                all_results.extend(config_results)
        
        # Save consolidated results
        if all_results:
            output_dir = f'{self.base_path}/FINAL MODEL/Results'
            os.makedirs(output_dir, exist_ok=True)
            
            results_df = pd.DataFrame(all_results)
            output_file = os.path.join(output_dir, '2024 D* price level 1h test results NEW.csv')
            results_df.to_csv(output_file, index=False)
            
            print(f"\nDC Price Level Analysis results saved to: {output_file}")
            print(f"Total scenario results: {len(all_results)}")
            print(f"Configurations processed: {len(BESS_CONFIGS)}")
            print(f"Price scenarios per config: 9 (base + ±10%, ±15%, ±20%, ±30%)")
        
        return all_results

def main():
    base_path = '/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration'
    controller = BESSWorkflowController(base_path)
    results = controller.run_all_configurations_dc_price_levels()
    return results

if __name__ == "__main__":
    results = main()