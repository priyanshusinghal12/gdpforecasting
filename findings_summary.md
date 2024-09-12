### findings_summary.md

# Summary of Findings

## 1. Correlation Analysis

- A correlation matrix between GDP, inflation, and unemployment revealed that:
  - **GDP and inflation (General Index)** have a strong negative correlation (-0.58), indicating that as inflation increases, GDP tends to decrease.
  - **GDP and unemployment** show a weaker negative correlation (-0.27), suggesting that when unemployment rises, GDP decreases but not as strongly.
  - **Inflation and unemployment** have a weak positive correlation (0.15), implying a slight increase in unemployment as inflation rises.

## 2. GDP Forecasting Using ARIMA

- The ARIMA model was used to predict India's GDP for the next 10 periods.
- The forecasted GDP shows a steady value at 2831.55 Billion USD, reflecting the limitations of the ARIMA model when working with limited data.

## 3. Unemployment Forecasting Using Prophet

- The Prophet model predicts that India's unemployment rate will likely rise in the next 24 months.
- The increase indicates potential challenges for India's labor market in the near future, which can have significant implications for policy and economic planning.

## 4. Linear Regression Analysis

- The linear regression model indicated a strong relationship between GDP, inflation, and unemployment, achieving an R-squared value of **0.85**, meaning 85% of the variation in GDP can be explained by inflation and unemployment.
- Mean Squared Error (MSE) for the model was relatively low, suggesting a good fit.

### Key Insights:

- There is a significant negative correlation between inflation and GDP, which is consistent with economic theories.
- Forecasting models suggest that India could face steady inflation but rising unemployment in the near future.
- Linear regression results reinforce the impact of inflation and unemployment on economic performance.
