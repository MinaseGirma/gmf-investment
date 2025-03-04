import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yfinance as yf 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Reshape into sequences
def create_sequences(data, seq_length=60):
    '''
    Creating sequences from time series data.
    
    Parameters:
    data (pd.DataFrame or pd.Series): Time series data.
    seq_length (int): Length of sequence from time series data.
    
    Returns:
    np.ndarray: Array of input sequences (X).
    np.ndarray: Array of target values (y).
    '''
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def predict(data, look_back, num_prediction, model):


    # prediction price
    scaler = MinMaxScaler()
    data_t = scaler.fit_transform(data)
    data_t = data_t.reshape((-1))
    prediction_list = data_t[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x, verbose=0)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
    prediction_list = prediction_list.reshape((num_prediction+1, 1))
    prediction_list = scaler.inverse_transform(prediction_list)

    # prediction date
    last_date = data.index.values[-1]
    prediction_dates = pd.date_range(last_date+1, periods=num_prediction+1).tolist()

    return prediction_list, prediction_dates

def plot_forecast(df,df_name, forecast, forecast_dates):
    # Plot historical df and forecast
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Price"], label="Historical Price")

    # Plot LSTM forecast
    plt.plot(forecast_dates, forecast, label="LSTM Forecast")

    plt.title(f"{df_name} Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def compare_models(ar, sar, lstm):
    
    datac = [
        ["ARIMA", ar[0], ar[1], ar[2]],
        ["SARIMA",sar[0], sar[1], sar[2] ],
        ["LSTM",lstm[0], lstm[1], lstm[2]]
    ]

    # Define the column names
    columns = ["Model", "MAE", "RMSE", "MAPE"]

    # Print the table
    print(tabulate(datac, headers=columns, tablefmt="grid"))