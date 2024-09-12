# India Economic Indicators Data Analysis

## Project Overview

This project involves analyzing India's economic indicators, including GDP, inflation, and unemployment, to identify patterns and forecast future trends. Using machine learning models such as ARIMA and Prophet, this project focuses on predicting GDP growth and unemployment rates.

## Dataset

The project uses three datasets:

- **India GDP Data**: Contains annual GDP data for India.
- **Unemployment in India**: Contains unemployment rates across different regions and time periods.
- **Inflation Data**: Contains monthly inflation data categorized by sectors.

### File Descriptions

- `India_GDP_Data.csv`: Annual GDP data for India.
- `Unemployment in India.csv`: Unemployment rate data for various states and regions.
- `All_India_Index_july2019_20Aug2020.csv`: Monthly inflation data categorized by sectors.
- `indiagdpproject.py`: Python script for loading, cleaning, analyzing, and modeling the data.

### Tools & Libraries

- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Plotly**: Interactive plots.
- **Scikit-learn**: Machine learning for linear regression.
- **Statsmodels**: Time-series forecasting using ARIMA.
- **Prophet**: Time-series forecasting for unemployment data.

### Project Steps

1. **Data Cleaning and Preparation**:
   - Removed unnecessary columns and fixed data format issues.
   - Merged GDP, unemployment, and inflation datasets on the year.
2. **Exploratory Data Analysis (EDA)**:

   - Generated correlation matrices to analyze the relationship between economic indicators.
   - Plotted distributions for GDP, unemployment rates, and inflation.

3. **GDP Forecasting**:

   - Implemented ARIMA model to predict future GDP.
   - Forecasted GDP for the next 10 periods.

4. **Unemployment Rate Forecasting**:

   - Used Prophet model to forecast unemployment rates over the next 24 months.

5. **Linear Regression**:
   - Explored the relationship between GDP, unemployment, and inflation using linear regression.
   - Evaluated the model with R-squared and Mean Squared Error (MSE).

### Key Findings

A detailed summary of findings can be found in the `findings_summary.md` file.

### Usage

To run the analysis:

1. Clone the repository.
2. Ensure that the following Python packages are installed:
   ```bash
   pip install pandas matplotlib seaborn plotly scikit-learn statsmodels prophet
   ```
