#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ytpatel3

Forex Arbitrage ML Project
ML models (Random Forest Classifier, LSTM, ARIMA) to predict arbitrage opportunities

"""
 
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define your paths
FOLDER_PATH = "/Users/ytpatel3/Downloads/fx_signals_data"
CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD"]
DOWNLOADS_PATH = '/Users/ytpatel3/Downloads'

# Load data
def load_data(pair):
    file_path = f"{FOLDER_PATH}/{pair[:3]}_{pair[3:]}_fx_signals.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Random Forest Model for Buy/Sell Signals
def random_forest(df):
    X = df[["open", "high", "low", "close", "SMA_200", "RSI", "SLOW_K", "SLOW_D"]]
    y = df["Combined_Signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
    
    rfmodel = RandomForestClassifier(n_estimators=100, random_state=31, class_weight='balanced')
    rfmodel.fit(X_train, y_train)
    
    predictions = rfmodel.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    
    return rfmodel, accuracy, report

# LSTM Model for Time-Series Forecasting
def lstm_forecast(df, days=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close"]])

    # 60-day lookback window
    X_lstm = []
    y_lstm = []
    window_length = 60
    
    for i in range(window_length, len(scaled_data)):
        X_lstm.append(scaled_data[i-window_length:i, 0])
        y_lstm.append(scaled_data[i, 0])
    
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")
    lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0)
    
    # Predict future prices
    recent_data = scaled_data[-window_length:]
    predictions = []
    
    for _ in range(days):
        recent_data_reshaped = np.reshape(recent_data, (1, window_length, 1))
        predicted_scaled = lstm_model.predict(recent_data_reshaped)
        predictions.append(predicted_scaled[0, 0])
        recent_data = np.append(recent_data, predicted_scaled)[-window_length:]
    
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': forecast_dates, 'LSTM_Predicted_Close': predicted_prices.flatten()})

# ARIMA Model for Short-Term Forecasting
def arima_forecast(df, days=5):
    model = ARIMA(df['close'], order=(5, 1, 0))
    arima_model = model.fit()
    forecast = arima_model.forecast(steps=days)
    
    forecast_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': forecast_dates, 'ARIMA_Forecast_Close': forecast})


def generate_trading_signals(forecast_df, column_name='LSTM_Predicted_Close'):
    forecast_df['Signal'] = 0  # Default: 0 means hold
    forecast_df['Signal'][1:] = [
        1 if forecast_df[column_name].iloc[i] > forecast_df[column_name].iloc[i-1] else -1 
        for i in range(1, len(forecast_df))
    ]
    forecast_df['Trade'] = forecast_df['Signal'].map({1: 'Long', -1: 'Short', 0: 'Hold'})
    return forecast_df[['Date', column_name, 'Trade']]


def main():
    forecast_data = {}
    
    output_dir = DOWNLOADS_PATH + '/fx_arbitrage'
    os.makedirs(output_dir, exist_ok=True)

    for pair in CURRENCY_PAIRS:
        df = load_data(pair)
        
        print(f"{pair[:3]}/{pair[3:]} Data:")
        
        # Run Random Forest for classification report
        rfmodel, accuracy, report = random_forest(df)
        print(f"Random Forest Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        
        # Forecasting with LSTM and ARIMA
        lstm_forecast_df = lstm_forecast(df, days=10)
        arima_forecast_df = arima_forecast(df, days=5)

        # Save predictions to dictionary for easy access
        forecast_data[pair] = {'LSTM': lstm_forecast_df, 'ARIMA': arima_forecast_df}

        # generate signals
        lstm_trading_signals = generate_trading_signals(lstm_forecast_df, column_name='LSTM_Predicted_Close')
        arima_trading_signals = generate_trading_signals(arima_forecast_df, column_name='ARIMA_Forecast_Close')
        
        combined_signals = pd.merge(
            lstm_trading_signals, arima_trading_signals, on='Date', how='outer', suffixes=('_LSTM', '_ARIMA')
        )

        # Display the final combined DataFrame
        print(f"{pair[:3]}/{pair[3:]} Combined Trading Signals:\n", combined_signals)
        
        clean_symbol = pair.replace('/', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"{clean_symbol}_arbitrage_opportunties.csv")
        combined_signals.to_csv(file_path, index=True)
     

if __name__ == "__main__":
    main()
