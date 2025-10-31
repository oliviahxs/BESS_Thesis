"""
Master Workflow Controller for BESS Arbitrage Optimization
Integrates all stages of the two-stage portfolio optimization process
"""

import pandas as pd
import numpy as np
import cplex
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Risk aversion values to test
RISK_AVERSION_VALUES = [0.5, 1, 2, 3, 4.5, 6, 7, 8, 9, 10]

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
        csv_file = f'{self.base_path}/cleaned_market_prices/dam_and_idm/2024_dam_idm_price.csv'
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

    def predict_idm_prices(self):
        """Generate IDM price predictions for 2021, 2022, 2023 using their actual DAM prices"""
        
        for year in [2021, 2022, 2023]:
            # Read the actual DAM prices for this scenario year
            dam_idm_file = f'{self.base_path}/cleaned_market_prices/dam_and_idm/{year}_dam_idm_price.csv'
            df = pd.read_csv(dam_idm_file)
            
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], format='%d/%m/%Y %H:%M')
            df = df.dropna(subset=['DAM price'])  # Only need DAM prices for prediction
            df = df[df['datetime'].dt.minute == 0]
            
            actual_dam = df['DAM price'].values
            
            # Generate predicted IDM prices using the autoregressive model
            predicted_idm = np.zeros_like(actual_dam)
            # Initialize first prediction (could use mean or first available IDM price)
            predicted_idm[0] = 0.643 * actual_dam[0] + 5.22  # No lagged term for first prediction
            
            for t in range(1, len(actual_dam)):
                predicted_idm[t] = 0.643 * actual_dam[t] + 0.26 * predicted_idm[t-1] + 5.22
            
            df['Predicted_IDM'] = predicted_idm
            
            # Save the results for this year
            output_file = f'{self.base_path}/part_4_optimisation/predicted_idm_prices_{year}_from_{year}dam.csv'
            df[['datetime', 'DAM price', 'Predicted_IDM']].to_csv(output_file, index=False)

    def calculate_idm_profits(self, year, power_mw, energy_mwh, dam_weight):
        """Calculate IDM profits using predicted prices and remaining capacity"""
        predicted_file = f'{self.base_path}/part_4_optimisation/predicted_idm_prices_{year}_from_{year}dam.csv'
        df = pd.read_csv(predicted_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.minute == 0]
        df.set_index('datetime', inplace=True)
        
        remaining_power = power_mw * (1 - dam_weight)
        remaining_energy = energy_mwh * (1 - dam_weight)
        
        daily_groups = df.groupby(df.index.date)
        daily_profits = []
        
        for date, group in daily_groups:
            prices = group['Predicted_IDM'].values
            # Remove 24-hour requirement - accept any number of hours
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=remaining_power, energy_capacity_mwh=remaining_energy,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                daily_profits.append({'Date': date, f'IDM_Daily_Profit_{year}': round(profit, 2)})
        
        return pd.DataFrame(daily_profits)

    def calculate_dc_profits(self, dam_weight, power_mw):
        """Calculate DC profits using 2024 prices and remaining capacity"""
        dc_power_capacity = power_mw * (1 - dam_weight)
        bid_size = 0.8 * dc_power_capacity
        
        dc_path = f'{self.base_path}/cleaned_market_prices/dc'
        dch_file = os.path.join(dc_path, 'dch/2024_dch_price.csv')
        dcl_file = os.path.join(dc_path, 'dcl/2024_dcl_price.csv')
        
        def calculate_dc_returns(csv_file, service_type):
            df = pd.read_csv(csv_file)
            df['EFA_Return'] = df['Clearing Price'] * 4 * bid_size
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

    def merge_idm_dc_profits(self, year, idm_df, dc_df):
        """Merge IDM and DC profits by matching month-day across years"""
        idm_df['Date'] = pd.to_datetime(idm_df['Date'])
        dc_df['Date'] = pd.to_datetime(dc_df['Date'])
        
        idm_df['month_day'] = idm_df['Date'].dt.strftime('%m-%d')
        dc_df['month_day'] = dc_df['Date'].dt.strftime('%m-%d')
        
        merged = pd.merge(idm_df, dc_df, on='month_day', how='inner', suffixes=('', '_dc'))
        merged['day_number'] = range(1, len(merged) + 1)
        
        result_columns = ['day_number', f'IDM_Daily_Profit_{year}', 'Daily DC returns']
        return merged[result_columns].copy()

    def calculate_statistics(self, df, year):
        """Calculate mean and covariance statistics"""
        idm_col = f'IDM_Daily_Profit_{year}'
        dc_col = 'Daily DC returns'
        
        if idm_col not in df.columns or dc_col not in df.columns:
            return None, None
        
        idm_returns = df[idm_col].values
        dc_returns = df[dc_col].values
        
        mean_returns = np.array([np.mean(idm_returns), np.mean(dc_returns)])
        cov_matrix = np.cov(idm_returns, dc_returns)
        
        return mean_returns, cov_matrix

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
        bounds = [(0, 1) for _ in range(len(mean_returns))]
        
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

    def run_baseline_stage(self, power_mw, energy_mwh, risk_aversion=3):
        """Run baseline stage (3-asset optimization using 2024 data)"""
        # Calculate DAM profits
        dam_df = self.calculate_dam_profits(power_mw, energy_mwh)
        
        # Calculate baseline IDM profits (using actual 2024 IDM prices)
        idm_file = f'{self.base_path}/cleaned_market_prices/dam_and_idm/2024_dam_idm_price.csv'
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
            # Remove 24-hour requirement - accept any number of hours
            if len(prices) == 0:
                continue
                
            profit = self.optimize_daily_arbitrage_cplex(
                prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                max_cycles_per_day=1.0, efficiency=0.86
            )
            
            if profit is not None:
                idm_profits.append({'Date': date, 'IDM_Daily_Profit': round(profit, 2)})
        
        idm_df = pd.DataFrame(idm_profits)
        
        # Calculate DC profits (full capacity for baseline)
        dc_bid_size = 0.8 * power_mw
        dc_path = f'{self.base_path}/cleaned_market_prices/dc'
        dch_file = os.path.join(dc_path, 'dch/2024_dch_price.csv')
        dcl_file = os.path.join(dc_path, 'dcl/2024_dcl_price.csv')
        
        def calculate_baseline_dc_returns(csv_file, service_type):
            df = pd.read_csv(csv_file)
            df['EFA_Return'] = df['Clearing Price'] * 4 * dc_bid_size
            daily_returns = df.groupby('Date')['EFA_Return'].sum().reset_index()
            daily_returns.columns = ['Date', f'Daily_Return_{service_type}']
            return daily_returns.sort_values('Date')
        
        dch_returns = calculate_baseline_dc_returns(dch_file, 'DCH')
        dcl_returns = calculate_baseline_dc_returns(dcl_file, 'DCL')
        
        combined_dc = pd.merge(dch_returns, dcl_returns, on='Date', how='outer')
        combined_dc['Daily DC returns'] = (
            combined_dc['Daily_Return_DCH'].fillna(0) + 
            combined_dc['Daily_Return_DCL'].fillna(0)
        ).round(1)
        dc_df = combined_dc[['Date', 'Daily DC returns']]
        
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
        
        result = self.markowitz_optimization(mean_returns, cov_matrix, risk_aversion=risk_aversion)
        
        if result:
            return {
                'dam_weight': result['weights'][0],
                'idm_weight': result['weights'][1], 
                'dc_weight': result['weights'][2],
                'expected_return': result['expected_return'],
                'portfolio_risk': result['portfolio_risk']
            }
        return None

    def run_updated_stage(self, power_mw, energy_mwh, dam_weight, risk_aversion=3):
        """Run updated stage (2-asset optimization for remaining capacity)"""
        # Generate IDM predictions for 2021-2023
        self.predict_idm_prices()
        
        results = {}
        
        for year in [2021, 2022, 2023]:
            # Calculate IDM profits with remaining capacity
            idm_df = self.calculate_idm_profits(year, power_mw, energy_mwh, dam_weight)
            
            # Calculate DC profits with remaining capacity  
            dc_df = self.calculate_dc_profits(dam_weight, power_mw)
            
            # Merge by date matching
            merged_df = self.merge_idm_dc_profits(year, idm_df, dc_df)
            
            # Calculate statistics and optimize
            mean_returns, cov_matrix = self.calculate_statistics(merged_df, year)
            
            if mean_returns is not None:
                result = self.markowitz_optimization(mean_returns, cov_matrix, risk_aversion=risk_aversion)
                
                if result:
                    results[year] = {
                        'idm_weight': result['weights'][0],
                        'dc_weight': result['weights'][1],
                        'expected_return': result['expected_return'],
                        'portfolio_risk': result['portfolio_risk']
                    }
        
        return results

    def run_single_config(self, config):
        """Run complete workflow for a single BESS configuration across all risk aversion values"""
        config_id = config['config_id']
        power_mw = config['power_mw']
        energy_mwh = config['energy_mwh']
        
        print(f"Processing Config {config_id}: {power_mw}MW/{energy_mwh}MWh")
        
        config_results = []
        
        for risk_aversion in RISK_AVERSION_VALUES:
            print(f"  Risk Aversion: {risk_aversion}")
            
            # Stage 1: Baseline optimization
            baseline_result = self.run_baseline_stage(power_mw, energy_mwh, risk_aversion)
            
            if not baseline_result:
                print(f"    Baseline optimization failed for Config {config_id}, RA {risk_aversion}")
                continue
            
            dam_weight = baseline_result['dam_weight']
            print(f"    DAM weight: {dam_weight:.3f}")
            
            # Stage 2: Updated optimization
            updated_results = self.run_updated_stage(power_mw, energy_mwh, dam_weight, risk_aversion)
            
            # Compile results for this risk aversion
            result = {
                'config_id': config_id,
                'power_mw': power_mw,
                'energy_mwh': energy_mwh,
                'risk_aversion': risk_aversion,
                'baseline_dam_weight': dam_weight,
                'baseline_idm_weight': baseline_result['idm_weight'],
                'baseline_dc_weight': baseline_result['dc_weight'],
                'baseline_expected_return': baseline_result['expected_return'],
                'baseline_portfolio_risk': baseline_result['portfolio_risk']
            }
            
            for year in [2021, 2022, 2023]:
                if year in updated_results:
                    result.update({
                        f'updated_{year}_idm_weight': updated_results[year]['idm_weight'],
                        f'updated_{year}_dc_weight': updated_results[year]['dc_weight'],
                        f'updated_{year}_expected_return': updated_results[year]['expected_return'],
                        f'updated_{year}_portfolio_risk': updated_results[year]['portfolio_risk']
                    })
            
            config_results.append(result)
        
        print(f"  Config {config_id} completed: {len(config_results)}/{len(RISK_AVERSION_VALUES)} risk aversion values")
        return config_results

    def run_all_configurations(self):
        """Run workflow for all 17 BESS configurations across multiple risk aversion values"""
        print("Starting BESS Portfolio Optimization Workflow")
        print("=" * 50)
        print(f"Risk aversion values: {RISK_AVERSION_VALUES}")
        
        all_results = []
        
        for config in BESS_CONFIGS:
            config_results = self.run_single_config(config)
            if config_results:
                all_results.extend(config_results)  # Flatten the list of lists
        
        # Save consolidated results
        if all_results:
            output_dir = f'{self.base_path}/figure_12_model_outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            results_df = pd.DataFrame(all_results)
            output_file = os.path.join(output_dir, 'figure_12_2024.csv')
            results_df.to_csv(output_file, index=False)
            
            print(f"\nConsolidated results saved to: {output_file}")
            print(f"Total rows: {len(all_results)}")
            print(f"Configurations: {len(BESS_CONFIGS)}, Risk aversion values: {len(RISK_AVERSION_VALUES)}")
        
        return all_results

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get the project root directory (script is in subfolder)
    base_path = os.path.dirname(os.path.dirname(script_dir))
    
    print(f"Using base path: {base_path}")
    controller = BESSWorkflowController(base_path)
    results = controller.run_all_configurations()
    return results

if __name__ == "__main__":
    results = main()