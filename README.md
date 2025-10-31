# BESS Thesis: Battery Energy Storage System Portfolio Optimization

This repository contains the code and data for a thesis on optimizing Battery Energy Storage System (BESS) portfolios across different electricity markets (DAM, IDM, and DC markets).

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CPLEX optimization solver (academic license available)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/BESS_Thesis.git
   cd BESS_Thesis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the optimization models:**
   ```bash
   # Figure 12 models (in organized subfolder)
   python part_4_optimisation/fig_12/fig_12_model_2024.py
   python part_4_optimisation/fig_12/fig_12_model_2023.py
   python part_4_optimisation/fig_12/fig_12_model_2022.py
   
   # Other optimization models
   python part_4_optimisation/fig_13_15_16_model.py
   python part_4_optimisation/fig_14_appendix_9_model.py
   python part_4_optimisation/fig_17_18_model.py
   python part_4_optimisation/fig_35_no_regression.py
   ```

## 📁 Project Structure

```
BESS_Thesis/
├── cleaned_market_prices/       # Market price data
│   ├── dam_and_idm/            # Day-ahead and intraday market prices
│   └── dc/                     # Dynamic containment prices
├── data_visualisation/         # Jupyter notebooks for analysis
├── part_2_analysis_gb_fleet_size/  # GB fleet size analysis
├── part_3_idm_prediction/      # IDM price prediction models
├── part_4_optimisation/        # Main optimization models
│   ├── fig_12/                 # Figure 12 portfolio optimization models
│   │   ├── fig_12_model_2022.py
│   │   ├── fig_12_model_2023.py
│   │   └── fig_12_model_2024.py
│   ├── fig_13_15_16_model.py   # Figures 13, 15-16 models
│   ├── fig_14_appendix_9_model.py  # Figure 14 & Appendix 9
│   ├── fig_17_18_model.py      # Figures 17-18 models
│   └── fig_35_no_regression.py # Figure 35 analysis
├── idm_price_predictions/      # Generated IDM predictions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

The project uses relative paths that automatically adapt to your local setup. No manual path configuration needed!

- **Data paths**: Automatically detected relative to project root
- **Output paths**: Results saved to dedicated output folders

## 📊 Models and Analysis

### Portfolio Optimization Models
- **Two-stage optimization**: First allocate to DAM, then optimize remaining capacity
- **Markowitz portfolio theory**: Risk-return optimization
- **Multiple BESS configurations**: 17 different power/energy combinations
- **Risk aversion analysis**: 10 different risk levels

### Market Data
- **DAM (Day-Ahead Market)**: Historical price data 2021-2025
- **IDM (Intraday Market)**: Historical and predicted prices
- **DC (Dynamic Containment)**: Frequency response market data

## 🔬 Research Output

Results are automatically saved to:
- `figure_12_model_outputs/`: Portfolio optimization results
- `fig_13_15_16_model_outputs/`: Additional model outputs
- `fig_14_appendix_9_model_outputs/`: Price level analysis
- `fig_35_model_outputs/`: Non-regression analysis

## 🤝 Contributing

This is a thesis project. For questions or collaboration:
1. Open an issue for bugs or questions
2. Fork the repo for contributions
3. Create pull requests for improvements

## 📝 License

This project is for academic research purposes.

## 📞 Contact

[Your Name] - [Your Email]
[University/Institution]