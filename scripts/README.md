# Time Series Forecasting and Portfolio Optimization

This repository contains the project for optimizing portfolio management using advanced time series forecasting techniques. The goal is to predict market trends and optimize asset allocation to enhance portfolio performance.

## Project Overview

Guide Me in Finance (GMF) Investments specializes in personalized portfolio management by leveraging data-driven insights. This project involves applying time series forecasting to historical financial data to enhance portfolio management strategies.

## Objectives

- Preprocess financial data for analysis.
- Develop time series forecasting models (ARIMA, SARIMA, LSTM).
- Forecast future market trends for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY).
- Optimize a sample investment portfolio based on forecasted data.

## Tasks

### Task 1: Data Preprocessing and Exploration

- Extract and clean data using YFinance.
- Conduct Exploratory Data Analysis (EDA).
- Decompose time series into trend, seasonal, and residual components.
  
### Task 2: Time Series Forecasting

- Choose and train models (ARIMA, SARIMA, LSTM).
- Optimize model parameters and evaluate performance using metrics like MAE, RMSE, and MAPE.

### Task 3: Forecast Analysis

- Generate future price predictions and analyze trends.
- Visualize forecasts with confidence intervals.

### Task 4: Portfolio Optimization

- Adjust a sample portfolio with TSLA, BND, and SPY.
- Calculate returns, optimize using the Sharpe Ratio, and visualize performance.

**scripts structure**

```
data_preprocessing.py                         # script for data processing
