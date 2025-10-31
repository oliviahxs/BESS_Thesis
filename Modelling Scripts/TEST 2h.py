'''
Interactive Efficient Frontier for Config 1 Battery with Utility Functions (A=3)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.widgets as widgets
from matplotlib.patches import Polygon

# Load the data
df = pd.read_csv('/Users/haixiaosun/Library/Mobile Documents/com~apple~CloudDocs/Coding Work/Markowitz exploration/Results EDA/Results for EDA/D*_price_addition_profit_stats.csv')

# Filter for Config 1 only (10MW/10MWh = 1-hour battery)
config_1_data = df[df['config_id'] == 8].copy()

print(f"Config 1 Data: {len(config_1_data)} scenarios")
print(f"Scenarios: {sorted(config_1_data['scenario'].unique())}")

def load_scenario_data(scenario='base_case'):
    """Load market data for specific scenario"""
    data = config_1_data[config_1_data['scenario'] == scenario]
    if data.empty:
        raise ValueError(f"No data for scenario '{scenario}'")
    
    row = data.iloc[0]
    
    # Extract returns and build covariance matrix
    returns = np.array([row['dam_mean'], row['idm_mean'], row['dc_mean']])
    stds = np.array([row['dam_std'], row['idm_std'], row['dc_std']])
    
    # Build correlation matrix
    corr_matrix = np.array([
        [1.0, row['corr_dam_idm'], row['corr_dam_dc']],
        [row['corr_dam_idm'], 1.0, row['corr_idm_dc']],
        [row['corr_dam_dc'], row['corr_idm_dc'], 1.0]
    ])
    
    # Convert to covariance matrix
    cov_matrix = np.outer(stds, stds) * corr_matrix
    
    return {
        'returns': returns,
        'cov_matrix': cov_matrix,
        'stds': stds,
        'scenario': scenario,
        'dc_addition_percent': row['dc_price_addition_percent'],
        'dc_mean': row['dc_mean']
    }

def generate_efficient_frontier(returns, cov_matrix, n_portfolios=100):
    """Generate efficient frontier portfolios"""
    n_assets = len(returns)
    results = []
    
    # Target returns from minimum to maximum possible
    min_ret = min(returns)
    max_ret = max(returns)
    target_returns = np.linspace(min_ret, max_ret, n_portfolios)
    
    for target_ret in target_returns:
        # Minimize portfolio variance for target return
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints: weights sum to 1, target return achieved
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - target_ret}
        ]
        
        # Bounds: long only
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(objective, x0=np.array([1/n_assets]*n_assets),
                        bounds=bounds, constraints=constraints, method='SLSQP')
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            results.append({
                'return': portfolio_return,
                'risk': portfolio_risk,
                'weights': weights
            })
    
    return pd.DataFrame(results)

def find_optimal_portfolio(returns, cov_matrix, risk_aversion=3.0):
    """Find utility-maximizing portfolio with given risk aversion"""
    n_assets = len(returns)
    
    def utility_objective(weights):
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        return -utility  # Minimize negative utility
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    
    result = minimize(utility_objective, x0=np.array([1/n_assets]*n_assets),
                    bounds=bounds, constraints=constraints, method='SLSQP')
    
    if result.success:
        weights = result.x
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'utility': portfolio_return - 0.5 * risk_aversion * np.dot(weights, np.dot(cov_matrix, weights))
        }
    return None

# Get all scenarios and sort by DC price addition
scenarios = sorted(config_1_data['scenario'].unique())
dc_percentages = []
for scenario in scenarios:
    dc_pct = config_1_data[config_1_data['scenario'] == scenario]['dc_price_addition_percent'].iloc[0]
    dc_percentages.append(dc_pct)

# Sort scenarios by DC percentage
sorted_indices = np.argsort(dc_percentages)
scenarios = [scenarios[i] for i in sorted_indices]
dc_percentages = [dc_percentages[i] for i in sorted_indices]

print(f"DC Price Scenarios: {len(scenarios)} total")
print(f"Range: {min(dc_percentages):.0f}% to {max(dc_percentages):.0f}%")

# Create the interactive plot with wider aspect ratio like screenshot
fig, ax = plt.subplots(figsize=(14, 8))  # Changed to wider aspect ratio
plt.subplots_adjust(bottom=0.20, right=0.85)  # Adjusted margins

# Colors for different scenarios - use similar colors to screenshot
colors = plt.cm.plasma(np.linspace(0, 1, len(scenarios)))

# Store data for all scenarios
all_frontiers = {}
all_optimal_portfolios = {}

# Generate all frontiers and store data
print("Generating efficient frontiers...")
for i, (scenario, color) in enumerate(zip(scenarios, colors)):
    try:
        data = load_scenario_data(scenario)
        returns = data['returns']
        cov_matrix = data['cov_matrix']
        
        # Generate frontier with more points like screenshot
        frontier = generate_efficient_frontier(returns, cov_matrix, n_portfolios=200)
        all_frontiers[scenario] = {
            'frontier': frontier,
            'data': data,
            'color': color
        }
        
        print(f"  {scenario} ({data['dc_addition_percent']:+.0f}%): {len(frontier)} points")
        
    except Exception as e:
        print(f"Error processing {scenario}: {e}")

# Debug: Let's check what data we actually have
print("\n" + "="*50)
print("DEBUGGING DATA")
print("="*50)

for scenario in scenarios[:3]:  # Check first 3 scenarios
    try:
        data = load_scenario_data(scenario)
        print(f"\nScenario: {scenario}")
        print(f"Returns: {data['returns']}")
        print(f"Stds: {data['stds']}")
        print(f"DC addition: {data['dc_addition_percent']}%")
        
        # Generate a small frontier to test
        frontier = generate_efficient_frontier(data['returns'], data['cov_matrix'], n_portfolios=10)
        print(f"Frontier points: {len(frontier)}")
        if len(frontier) > 0:
            print(f"Risk range: {frontier['risk'].min():.1f} to {frontier['risk'].max():.1f}")
            print(f"Return range: {frontier['return'].min():.1f} to {frontier['return'].max():.1f}")
    except Exception as e:
        print(f"Error with {scenario}: {e}")

print("="*50)

# Global variables for hover functionality
hover_annotation = None
optimal_data_storage = {}

def on_hover(event):
    """Handle mouse hover events to show optimal portfolio composition"""
    global hover_annotation
    
    if event.inaxes != ax:
        return
        
    # Check if mouse is near any optimal portfolio point
    for scenario, opt_data in optimal_data_storage.items():
        if 'scatter' in opt_data and 'optimal' in opt_data:
            scatter = opt_data['scatter']
            optimal = opt_data['optimal']
            
            # Check if mouse is near this point
            contains, _ = scatter.contains(event)
            if contains:
                # Remove previous annotation
                if hover_annotation:
                    hover_annotation.remove()
                
                # Create hover text with portfolio mix
                weights = optimal['weights']
                hover_text = f"{scenario}\nDAM: {weights[0]:.1%}\nIDM: {weights[1]:.1%}\nDC: {weights[2]:.1%}\nReturn: {optimal['return']:.1f}\nRisk: {optimal['risk']:.1f}"
                
                # Add annotation
                hover_annotation = ax.annotate(hover_text, 
                                             xy=(optimal['risk'], optimal['return']),
                                             xytext=(20, 20), textcoords='offset points',
                                             bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8),
                                             fontsize=9, ha='left')
                fig.canvas.draw_idle()
                return
    
    # Remove annotation if not hovering over any point
    if hover_annotation:
        hover_annotation.remove()
        hover_annotation = None
        fig.canvas.draw_idle()

def plot_frontiers_and_utility(risk_aversion=3.0):
    """Plot efficient frontiers with utility curves for given risk aversion"""
    global optimal_data_storage
    ax.clear()
    optimal_data_storage.clear()
    
    frontier_lines = []
    optimal_points = []
    utility_lines = []
    
    # First, let's check what data we actually have and set appropriate axis limits
    all_risks = []
    all_returns = []
    
    # Plot all efficient frontiers as scatter points like screenshot
    for scenario in scenarios:
        if scenario in all_frontiers:
            frontier_data = all_frontiers[scenario]
            frontier = frontier_data['frontier']
            data = frontier_data['data']
            color = frontier_data['color']
            dc_percent = data['dc_addition_percent']
            
            if len(frontier) > 0:  # Make sure we have data
                # Collect all risk/return values for axis limits
                all_risks.extend(frontier['risk'].tolist())
                all_returns.extend(frontier['return'].tolist())
                
                # Plot frontier as scatter points (like screenshot)
                scatter_points = ax.scatter(frontier['risk'], frontier['return'], 
                                          color=color, s=8, alpha=0.7,
                                          label=f'DC μ scaling {dc_percent:+.0f}% (μ={data["dc_mean"]:.1f})')
                
                # Find optimal portfolio for this risk aversion
                returns = data['returns']
                cov_matrix = data['cov_matrix']
                optimal = find_optimal_portfolio(returns, cov_matrix, risk_aversion)
                
                if optimal:
                    # Add optimal portfolio to our collections
                    all_risks.append(optimal['risk'])
                    all_returns.append(optimal['return'])
                    
                    # Plot optimal point with larger marker
                    optimal_scatter = ax.scatter(optimal['risk'], optimal['return'], 
                                               color=color, s=100, marker='*', 
                                               zorder=10, edgecolors='black', linewidth=1)
                    optimal_points.append(optimal_scatter)
                    
                    # Store data for hover functionality
                    optimal_data_storage[scenario] = {
                        'scatter': optimal_scatter,
                        'optimal': optimal
                    }
    
    # Add individual assets for reference (like screenshot)
    if 'base_case' in all_frontiers:
        base_data = all_frontiers['base_case']['data']
        individual_returns = base_data['returns']
        individual_risks = base_data['stds']
        asset_names = ['DAM', 'IDM', 'DC baseline']
        asset_colors = ['blue', 'cyan', 'pink']
        asset_markers = ['o', 's', '^']
        
        for ret, risk, name, color, marker in zip(individual_returns, individual_risks, 
                                                asset_names, asset_colors, asset_markers):
            ax.scatter(risk, ret, color=color, s=80, marker=marker, 
                      alpha=0.9, edgecolors='black', zorder=8, linewidth=1,
                      label=name)
            # Add individual assets to our collections for axis limits
            all_risks.append(risk)
            all_returns.append(ret)
    
    # Set axis limits based on actual data with some padding
    if all_risks and all_returns:
        risk_padding = (max(all_risks) - min(all_risks)) * 0.05
        return_padding = (max(all_returns) - min(all_returns)) * 0.05
        
        ax.set_xlim(min(all_risks) - risk_padding, max(all_risks) + risk_padding)
        ax.set_ylim(min(all_returns) - return_padding, max(all_returns) + return_padding)
        
        print(f"Risk range: {min(all_risks):.1f} to {max(all_risks):.1f}")
        print(f"Return range: {min(all_returns):.1f} to {max(all_returns):.1f}")
    else:
        # Fallback to original limits if no data
        ax.set_xlim(24, 40)
        ax.set_ylim(45, 105)
        print("No data found, using default axis limits")
    
    # Styling to match screenshot
    ax.set_xlabel('Risk (σ)', fontsize=12)
    ax.set_ylabel('Return (μ)', fontsize=12)
    
    ax.set_title(f'Efficient Frontiers for Upward DC Price Scaling Scenarios\n'
                f'Risk Aversion A = {risk_aversion:.1f} | Hover over ★ for portfolio mix', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Legend with smaller font to match screenshot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.draw()

# Initial plot with A=3
print("Creating initial plot with A=3...")
plot_frontiers_and_utility(3.0)

# Connect hover event
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Add interactive slider for risk aversion
ax_slider = plt.axes([0.15, 0.08, 0.5, 0.03])
slider = widgets.Slider(ax_slider, 'Risk Aversion (A)', 0.5, 8.0, 
                       valinit=3.0, valfmt='%.1f', facecolor='lightblue')

def update_plot(val):
    """Update plot when slider changes"""
    new_risk_aversion = slider.val
    plot_frontiers_and_utility(new_risk_aversion)

slider.on_changed(update_plot)

# Add reset button
ax_reset = plt.axes([0.75, 0.08, 0.08, 0.03])
button = widgets.Button(ax_reset, 'Reset A=3', color='lightcoral', hovercolor='red')

def reset(event):
    slider.reset()
    slider.set_val(3.0)

button.on_clicked(reset)

plt.show()

# Print summary
print(f"\n" + "="*80)
print(f"INTERACTIVE EFFICIENT FRONTIER ANALYSIS - CONFIG 1 BATTERY")
print(f"="*80)
print(f"Configuration: 10MW/10MWh (1-hour duration)")
print(f"Total DC Price Scenarios: {len(scenarios)}")
print(f"DC Price Range: {min(dc_percentages):+.0f}% to {max(dc_percentages):+.0f}%")
print(f"Risk Aversion Range: 0.5 to 8.0 (interactive slider)")
print(f"Initial Risk Aversion: A = 3.0")
print(f"\nHover over ★ symbols to see optimal portfolio composition!")
print(f"Each scenario shows a different efficient frontier based on DC price modifications.")