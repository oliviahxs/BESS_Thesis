"""
Master Workflow Controller for BESS Arbitrage Optimization - 2024+2025 Combined Version
Baseline stage only (3-asset optimization) using combined 2024 and 2025 market data
Tests multiple risk aversion values across all BESS configurations
Data is combined chronologically: 2024 data first, then 2025 data
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

class BESSCombinedController:
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

    def calculate_combined_dam_profits(self, power_mw, energy_mwh):
        """Calculate DAM daily profits combining 2024 and 2025 data"""
        combined_profits = []
        
        for year in [2024, 2025]:
            print(f"      Processing DAM {year}...")
            csv_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/{year} DAM IDM Price.csv'
            df = pd.read_csv(csv_file)
            df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
            df['price'] = df['DAM price']
            df = df.dropna(subset=['price'])
            df = df[df['datetime'].dt.minute == 0]
            df.set_index('datetime', inplace=True)
            
            daily_groups = df.groupby(df.index.date)
            year_profits = []
            
            for date, group in daily_groups:
                prices = group['price'].values
                if len(prices) == 0:
                    continue
                    
                profit = self.optimize_daily_arbitrage_cplex(
                    prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                    max_cycles_per_day=1.0, efficiency=0.86
                )
                
                if profit is not None:
                    year_profits.append({'Date': date, 'DAM_Daily_Profit': round(profit, 2), 'Year': year})
            
            combined_profits.extend(year_profits)
            print(f"        {year}: {len(year_profits)} days")
        
        # Sort chronologically
        combined_df = pd.DataFrame(combined_profits)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        return combined_df[['Date', 'DAM_Daily_Profit']]

    def calculate_combined_idm_profits(self, power_mw, energy_mwh):
        """Calculate IDM daily profits combining 2024 and 2025 data"""
        combined_profits = []
        
        for year in [2024, 2025]:
            print(f"      Processing IDM {year}...")
            csv_file = f'{self.base_path}/Cleaned market prices/DAM & IDM/{year} DAM IDM Price.csv'
            df = pd.read_csv(csv_file)
            df['datetime'] = df.iloc[:, 0].apply(self.parse_datetime)
            df['price'] = df['IDM price']
            df = df.dropna(subset=['price'])
            df = df[df['datetime'].dt.minute == 0]
            df.set_index('datetime', inplace=True)
            
            daily_groups = df.groupby(df.index.date)
            year_profits = []
            
            for date, group in daily_groups:
                prices = group['price'].values
                if len(prices) == 0:
                    continue
                    
                profit = self.optimize_daily_arbitrage_cplex(
                    prices, power_capacity_mw=power_mw, energy_capacity_mwh=energy_mwh,
                    max_cycles_per_day=1.0, efficiency=0.86
                )
                
                if profit is not None:
                    year_profits.append({'Date': date, 'IDM_Daily_Profit': round(profit, 2), 'Year': year})
            
            combined_profits.extend(year_profits)
            print(f"        {year}: {len(year_profits)} days")
        
        # Sort chronologically
        combined_df = pd.DataFrame(combined_profits)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        return combined_df[['Date', 'IDM_Daily_Profit']]

    def calculate_combined_dc_profits(self, power_mw):
        """Calculate DC profits combining 2024 and 2025 data"""
        dc_bid_size = 0.8 * power_mw
        combined_profits = []
        
        for year in [2024, 2025]:
            print(f"      Processing DC {year}...")
            dc_path = f'{self.base_path}/Cleaned market prices/DC'
            dch_file = os.path.join(dc_path, f'DCH/{year} DCH Price.csv')
            dcl_file = os.path.join(dc_path, f'DCL/{year} DCL Price.csv')
            
            def calculate_dc_returns(csv_file, service_type):
                df = pd.read_csv(csv_file)
                df['EFA_Return'] = df['Clearing Price'] * 4 * dc_bid_size
                daily_returns = df.groupby('Date')['EFA_Return'].sum().reset_index()
                daily_returns.columns = ['Date', f'Daily_Return_{service_type}']
                return daily_returns.sort_values('Date')
            
            dch_returns = calculate_dc_returns(dch_file, 'DCH')
            dcl_returns = calculate_dc_returns(dcl_file, 'DCL')
            
            combined_dc = pd.merge(dch_returns, dcl_returns, on='Date', how='outer')
            combined_dc['Daily DC returns'] = (
                combined_dc['Daily_Return_DCH'].fillna(0) + 
                combined_dc['Daily_Return_DCL'].fillna(0)
            ).round(1)
            combined_dc['Year'] = year
            
            year_profits = combined_dc[['Date', 'Daily DC returns']].to_dict('records')
            combined_profits.extend(year_profits)
            print(f"        {year}: {len(year_profits)} days")
        
        # Sort chronologically
        combined_df = pd.DataFrame(combined_profits)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        return combined_df[['Date', 'Daily DC returns']]

    def markowitz_optimization(self, mean_returns, cov_matrix, risk_aversion=3):
        """Perform Markowitz portfolio optimization using SciPy"""
        from scipy.optimize import minimize
        
        # Normalize returns and risks (mean=0, std=1)
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

    def run_combined_optimization(self, power_mw, energy_mwh, risk_aversion=3):
        """Run combined 3-asset optimization using 2024+2025 data chronologically"""
        print(f"    Calculating combined profits for {power_mw}MW/{energy_mwh}MWh...")
        
        # Calculate profits for all three markets (combined 2024+2025)
        dam_df = self.calculate_combined_dam_profits(power_mw, energy_mwh)
        idm_df = self.calculate_combined_idm_profits(power_mw, energy_mwh)
        dc_df = self.calculate_combined_dc_profits(power_mw)
        
        # Merge all three assets by date
        merged = pd.merge(dam_df, idm_df, on='Date', how='inner')
        merged = pd.merge(merged, dc_df, on='Date', how='inner')
        
        print(f"    Combined data: {len(merged)} trading days total")
        
        # Calculate year breakdown for verification
        merged['Year'] = merged['Date'].dt.year
        year_counts = merged['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"      {year}: {count} days")
        
        # Calculate statistics and optimize
        dam_returns = merged['DAM_Daily_Profit'].values
        idm_returns = merged['IDM_Daily_Profit'].values
        dc_returns = merged['Daily DC returns'].values
        
        mean_returns = np.array([np.mean(dam_returns), np.mean(idm_returns), np.mean(dc_returns)])
        cov_matrix = np.cov([dam_returns, idm_returns, dc_returns])
        
        print(f"    Combined mean returns: DAM={mean_returns[0]:.2f}, IDM={mean_returns[1]:.2f}, DC={mean_returns[2]:.2f}")
        
        result = self.markowitz_optimization(mean_returns, cov_matrix, risk_aversion=risk_aversion)
        
        if result:
            return {
                'dam_weight': result['weights'][0],
                'idm_weight': result['weights'][1], 
                'dc_weight': result['weights'][2],
                'expected_return': result['expected_return'],
                'portfolio_risk': result['portfolio_risk'],
                'num_trading_days': len(merged),
                'num_2024_days': year_counts.get(2024, 0),
                'num_2025_days': year_counts.get(2025, 0),
                'dam_mean': mean_returns[0],
                'idm_mean': mean_returns[1],
                'dc_mean': mean_returns[2],
                'dam_std': np.sqrt(cov_matrix[0,0]),
                'idm_std': np.sqrt(cov_matrix[1,1]),
                'dc_std': np.sqrt(cov_matrix[2,2]),
                'corr_dam_idm': cov_matrix[0,1] / (np.sqrt(cov_matrix[0,0]) * np.sqrt(cov_matrix[1,1])),
                'corr_dam_dc': cov_matrix[0,2] / (np.sqrt(cov_matrix[0,0]) * np.sqrt(cov_matrix[2,2])),
                'corr_idm_dc': cov_matrix[1,2] / (np.sqrt(cov_matrix[1,1]) * np.sqrt(cov_matrix[2,2]))
            }
        return None

    def run_single_config(self, config):
        """Run optimization for a single BESS configuration across all risk aversion values"""
        config_id = config['config_id']
        power_mw = config['power_mw']
        energy_mwh = config['energy_mwh']
        
        print(f"Processing Config {config_id}: {power_mw}MW/{energy_mwh}MWh")
        
        config_results = []
        
        for risk_aversion in RISK_AVERSION_VALUES:
            print(f"  Risk Aversion: {risk_aversion}")
            
            # Run combined optimization with 2024+2025 data
            result = self.run_combined_optimization(power_mw, energy_mwh, risk_aversion)
            
            if not result:
                print(f"    ❌ Optimization failed for Config {config_id}, RA {risk_aversion}")
                continue
            
            print(f"    ✅ Optimal weights: DAM={result['dam_weight']:.3f}, IDM={result['idm_weight']:.3f}, DC={result['dc_weight']:.3f}")
            
            # Compile results for this risk aversion
            config_result = {
                'config_id': config_id,
                'power_mw': power_mw,
                'energy_mwh': energy_mwh,
                'risk_aversion': risk_aversion,
                'baseline_dam_weight': result['dam_weight'],
                'baseline_idm_weight': result['idm_weight'],
                'baseline_dc_weight': result['dc_weight'],
                'baseline_expected_return': result['expected_return'],
                'baseline_portfolio_risk': result['portfolio_risk'],
                'num_trading_days': result['num_trading_days'],
                'num_2024_days': result['num_2024_days'],
                'num_2025_days': result['num_2025_days'],
                'dam_mean': result['dam_mean'],
                'idm_mean': result['idm_mean'],
                'dc_mean': result['dc_mean'],
                'dam_std': result['dam_std'],
                'idm_std': result['idm_std'],
                'dc_std': result['dc_std'],
                'corr_dam_idm': result['corr_dam_idm'],
                'corr_dam_dc': result['corr_dam_dc'],
                'corr_idm_dc': result['corr_idm_dc']
            }
            
            config_results.append(config_result)
        
        print(f"  ✅ Config {config_id} completed: {len(config_results)}/{len(RISK_AVERSION_VALUES)} risk aversion values")
        return config_results

    def run_all_configurations(self):
        """Run optimization for all 17 BESS configurations across multiple risk aversion values"""
        print("🚀 Starting BESS 2024+2025 Combined Portfolio Optimization Workflow")
        print("=" * 70)
        print(f"📊 Risk aversion values: {RISK_AVERSION_VALUES}")
        print(f"🔋 BESS configurations: {len(BESS_CONFIGS)}")
        print(f"📅 Using combined 2024+2025 market data (chronologically ordered)")
        print(f"🎯 Baseline stage only (3-asset optimization)")
        print()
        
        all_results = []
        
        for config in BESS_CONFIGS:
            config_results = self.run_single_config(config)
            if config_results:
                all_results.extend(config_results)  # Flatten the list of lists
        
        # Save consolidated results
        if all_results:
            output_dir = f'{self.base_path}/FINAL MODEL'
            os.makedirs(output_dir, exist_ok=True)
            
            results_df = pd.DataFrame(all_results)
            output_file = os.path.join(output_dir, '2024_2025_combined_multiple_A_results.csv')
            results_df.to_csv(output_file, index=False)
            
            print("=" * 70)
            print("🎯 RESULTS SUMMARY")
            print("=" * 70)
            print(f"✅ Results saved to: {output_file}")
            print(f"📊 Total rows: {len(all_results)}")
            print(f"🔋 Configurations processed: {len(BESS_CONFIGS)}")
            print(f"📈 Risk aversion values: {len(RISK_AVERSION_VALUES)}")
            print(f"💯 Expected total combinations: {len(BESS_CONFIGS) * len(RISK_AVERSION_VALUES)}")
            
            # Quick statistics
            if len(all_results) > 0:
                sample_result = all_results[0]
                total_days = sample_result['num_trading_days']
                days_2024 = sample_result['num_2024_days']
                days_2025 = sample_result['num_2025_days']
                
                print(f"📅 Combined trading days: {total_days}")
                print(f"   📅 2024 days: {days_2024}")
                print(f"   📅 2025 days: {days_2025}")
                
                # Show some sample results
                print("\n📋 Sample Results (A=3):")
                a3_results = [r for r in all_results if r['risk_aversion'] == 3][:3]
                for r in a3_results:
                    print(f"   Config {r['config_id']}: DAM={r['baseline_dam_weight']:.3f}, IDM={r['baseline_idm_weight']:.3f}, DC={r['baseline_dc_weight']:.3f}")
        
        return all_results

def main():
    base_path = '/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration'
    controller = BESSCombinedController(base_path)
    results = controller.run_all_configurations()
    return results

if __name__ == "__main__":
    results = main()