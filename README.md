```markdown
# Stock Analysis and Forecasting with Machine Learning

This repository contains a comprehensive Jupyter Notebook that performs exploratory data analysis (EDA) and various modeling techniques on historical stock data for TCS. The analysis includes visualization, statistical testing, and forecasting using multiple approaches, including:

- Exploratory Data Analysis (EDA): Correlation heatmaps, trend analysis, and impact analysis (e.g., dividends and stock splits).
- Technical Analysis: Moving Average Crossover Strategy.
- Time Series Forecasting with LSTM: Using a Long Short-Term Memory network for stock price prediction.
- Traditional Regression Models: Linear Regression, Random Forest, and XGBoost for predicting stock close prices.
- Hyperparameter Tuning: Optimizing the Random Forest model using Grid Search.
- Statistical Time Series Modeling: ARIMA model for forecasting.

## Table of Contents

- Project Overview
- Data
- Features
- Installation
- Usage
- Project Structure
- Contributing
- License
- Contact

## Project Overview

This project analyzes historical stock data for TCS. It includes:
- Data Loading and Preprocessing: Reading data from a CSV file, converting date fields, and sorting data.
- Exploratory Data Analysis (EDA): Visualizations and statistical tests to explore relationships between variables such as close price, volume, dividends, and stock splits.
- Technical Analysis: Creating signals based on moving average crossovers.
- Predictive Modeling: Implementing and comparing several predictive models:
  - LSTM neural network for time series forecasting.
  - Linear Regression for basic predictive modeling.
  - Random Forest and XGBoost Regressors for ensemble learning.
  - Hyperparameter tuning for the Random Forest model.
  - ARIMA model for time series forecasting.

## Data

The primary data file used is `TCS_stock_history.csv`. Ensure that this CSV file is located in the same directory as the notebook or update the file path accordingly.

## Features

- Data Preprocessing:
  - Date conversion and sorting.
  - Computation of rolling averages.
  - Feature engineering (e.g., previous close, day of week, month, etc.).
- Visualizations:
  - Heatmaps, line plots, box plots, scatter plots, and interactive Plotly visualizations.
- Modeling Approaches:
  - LSTM for deep learning-based forecasting.
  - Traditional regression techniques.
  - Ensemble methods with Random Forest and XGBoost.
  - ARIMA for time series analysis.
- Evaluation Metrics:
  - Mean Absolute Error (MAE) for model performance evaluation.
- Hyperparameter Tuning:
  - Grid Search for optimizing Random Forest parameters.

## Installation

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook or JupyterLab

### Required Python Libraries

The notebook uses several libraries including:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- tqdm
- plotly
- statsmodels
- xgboost

You can install the required libraries using `pip`. It is recommended to create a virtual environment first.

### Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Sanika2511/TCS_stock.git
   cd TCS_stock
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   If you have a `requirements.txt` file, run:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, you can install the libraries manually:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow tqdm plotly statsmodels xgboost
   ```

## Usage

1. Start Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

2. **Open the Notebook:**

   Locate the notebook file (e.g., `TCS_stockpred.ipynb`) in your browser and open it.

3. **Run the Notebook:**

   Execute the cells sequentially to perform data analysis, visualization, and modeling.

4. **Review the Output:**

   - Visualizations (plots and interactive charts) will be displayed inline.
   - Model performance metrics such as Mean Absolute Error (MAE) will be printed in the output.
   - Predictions from the LSTM model are saved to `lstm_predictions.csv`.

## Project Structure

```
your-repository/
├── TCS_stock_history.csv        # Historical stock data for TCS
├── TCS_stock_analysis.ipynb     # Jupyter Notebook containing the code and analysis
├── lstm_predictions.csv         # CSV file with LSTM model predictions (generated after running the notebook)
├── requirements.txt             # (Optional) File listing required Python packages
└── README.md                    # This file
```

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE). You can use, modify, and distribute this project as per the license terms.

## Contact

For questions or further information, please contact:

- Sanika Sharma - sanika261101@gmail.com
- [GitHub Profile](https://github.com/Sanika2511)

---

*Happy Analyzing and Forecasting!*
```
