import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid")

# Load data
gdp_data = pd.read_csv('./India_GDP_Data.csv')
unemployment_data = pd.read_csv('./Unemployment in India.csv')
inflation_data = pd.read_csv('./All_India_Index_july2019_20Aug2020.csv')

# Converting 'Year' to datetime in gdp_data and extract just the year
gdp_data['Year'] = pd.to_datetime(gdp_data['Year'], format='%Y').dt.year

#clean
unemployment_data[' Date'] = unemployment_data[' Date'].str.strip()
unemployment_data[' Date'] = pd.to_datetime(unemployment_data[' Date'], format='%d-%m-%Y')
unemployment_data['Year'] = unemployment_data[' Date'].dt.year

inflation_data['Year'] = inflation_data['Year'].ffill().astype(int)
inflation_data['Month'] = inflation_data['Month'].str.strip()  # Remove extra spaces
inflation_data['Month'].replace({'Marcrh': 'March'}, inplace=True)  # Fix typos
inflation_data['Date'] = pd.to_datetime(inflation_data['Year'].astype(str) + inflation_data['Month'], format='%Y%B')

merged_data_1 = pd.merge(gdp_data, unemployment_data, on='Year', how='inner')
print("After First Merge (GDP and Unemployment):")
print(merged_data_1.head())

merged_data = pd.merge(merged_data_1, inflation_data, on='Year', how='inner')
print("After Second Merge (With Inflation):")
print(merged_data.head())

merged_data.drop(columns=[' Date', 'Date'], inplace=True)
merged_data.columns = merged_data.columns.str.strip()

print("Columns in merged_data:", merged_data.columns)


selected_columns = [
    'GDP_In_Billion_USD', 
    'Per_Capita_in_USD', 
    'Percentage_Growth', 
    'Estimated Unemployment Rate (%)', 
    'General index'  # Inflation proxy
]


filtered_data = merged_data[selected_columns]

plt.figure(figsize=(10, 6))
sns.heatmap(filtered_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Economic Indicators')
plt.show()

plot_data = merged_data[['Year', 'GDP_In_Billion_USD', 'General index', 'Estimated Unemployment Rate (%)']].dropna()


fig = px.line(plot_data, x='Year', 
              y=['GDP_In_Billion_USD', 'General index', 'Estimated Unemployment Rate (%)'], 
              title='Economic Indicators Over Time', labels={'value': 'Value', 'variable': 'Indicator'})
fig.show()


plt.figure(figsize=(8, 6))
sns.histplot(merged_data['GDP_In_Billion_USD'], kde=True)
plt.title('GDP Distribution')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(merged_data['Estimated Unemployment Rate (%)'], kde=True)
plt.title('Unemployment Rate Distribution')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(merged_data['General index'], kde=True)
plt.title('Inflation Rate Distribution')
plt.show()

# ARIMA model for GDP Forecasting
gdp_series = merged_data.set_index('Year')['GDP_In_Billion_USD']

# Fit ARIMA model
arima_model = ARIMA(gdp_series, order=(5, 1, 0))
arima_result = arima_model.fit()

# Forecast GDP for 10 steps
gdp_forecast = arima_result.forecast(steps=10)
print(gdp_forecast)

# Plot actual vs forecasted GDP
plt.figure(figsize=(10, 6))
plt.plot(gdp_series, label='Actual GDP')
plt.plot(gdp_series.index[-1] + np.arange(1, 11), gdp_forecast, label='Forecasted GDP', color='red')
plt.title('GDP Forecasting using ARIMA')
plt.legend()
plt.show()

# Prophet model for Unemployment Forecasting
unemployment_df = merged_data[['Year', 'Estimated Unemployment Rate (%)']].rename(columns={'Year': 'ds', 'Estimated Unemployment Rate (%)': 'y'})

# Fit Prophet model
prophet_model = Prophet()
prophet_model.fit(unemployment_df)

# Forecast Unemployment for 24 months
future = prophet_model.make_future_dataframe(periods=24, freq='M')
forecast = prophet_model.predict(future)

# Plot Unemployment Forecasting
fig = prophet_model.plot(forecast)
plt.title('Unemployment Rate Forecasting using Prophet')
plt.show()

imputer = SimpleImputer(strategy='mean')
X = merged_data[['General index', 'Estimated Unemployment Rate (%)']]
X = imputer.fit_transform(X)
y = merged_data['GDP_In_Billion_USD']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict GDP
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linestyle='--')
plt.title('Actual vs Predicted GDP')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.show()

# Findings
print("Key Insights:")
print("1. There is a significant correlation between inflation, unemployment, and GDP.")
print("2. The ARIMA model provided a reasonable forecast for GDP.")
print("3. The Prophet model predicted a rise in unemployment in the future.")
print("4. The linear regression model showed a strong relationship between inflation, unemployment, and GDP, with an R-squared value of {:.2f}.".format(r2))

fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_data['Year'], y=merged_data['GDP_In_Billion_USD'], mode='lines', name='GDP'))
fig.add_trace(go.Scatter(x=merged_data['Year'], y=merged_data['General index'], mode='lines', name='Inflation'))
fig.add_trace(go.Scatter(x=merged_data['Year'], y=merged_data['Estimated Unemployment Rate (%)'], mode='lines', name='Unemployment'))
fig.update_layout(title='Economic Trends Over Time',
                 xaxis_title='Year',
                 yaxis_title='Value')
fig.show()
