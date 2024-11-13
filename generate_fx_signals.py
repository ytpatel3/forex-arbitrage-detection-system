"""
Forex Arbitrage ML Project
Generate Buy/Hold/Sell Signals using forex data
"""

import pandas as pd
import os

FOLDER_PATH = "/Users/ytpatel3/Downloads/forex_pairs_data"
CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD"]
DOWNLOADS_PATH = '/Users/ytpatel3/Downloads'


def create_signals(df, buy_threshold=0.5, sell_threshold=-0.5, weight_increment=0.1):
    
    # use weights to create a signal column 
    
    df['Signal_Score'] = 0
    weights = {'sma_trend': 0.1, 'rsi': 0.1, 'stoch': 0.1, 'price_pattern': 0.1}

    # Condition: SMA-based trend
    df['sma_trend'] = 0
    df.loc[df['close'] > df['SMA_200'], 'sma_trend'] = 1  # Uptrend
    df.loc[df['close'] < df['SMA_200'], 'sma_trend'] = -1 # Downtrend

    # Condition: RSI
    df['rsi_signal'] = 0
    df.loc[df['RSI'] < 30, 'rsi_signal'] = 1  # Strong buy
    df.loc[df['RSI'] > 70, 'rsi_signal'] = -1 # Strong sell

    # Condition: Stochastic Oscillator
    df['stoch_signal'] = 0
    df.loc[(df['SLOW_K'] < 20) & (df['SLOW_K'] > df['SLOW_D']), 'stoch_signal'] = 1  # Buy signal
    df.loc[(df['SLOW_K'] > 80) & (df['SLOW_K'] < df['SLOW_D']), 'stoch_signal'] = -1 # Sell signal

    # Condition: Simple OHLC pattern (e.g., higher highs or lower lows over last 3 days)
    df['price_pattern'] = 0
    df.loc[(df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)), 'price_pattern'] = 1  # Upward momentum
    df.loc[(df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)), 'price_pattern'] = -1 # Downward momentum

    # Iteratively increase weights until we get sufficient non-zero signals
    while (df['Signal_Score'].gt(buy_threshold).sum() == 0) and (df['Signal_Score'].lt(sell_threshold).sum() == 0):
        # Calculate Signal_Score with current weights
        df['Signal_Score'] = (
            weights['sma_trend'] * df['sma_trend'] +
            weights['rsi'] * df['rsi_signal'] +
            weights['stoch'] * df['stoch_signal'] +
            weights['price_pattern'] * df['price_pattern']
        )

        # Check for non-zero signals and adjust weights if too many zeros
        if df['Signal_Score'].gt(buy_threshold).sum() == 0:
            weights['sma_trend'] += weight_increment
            weights['rsi'] += weight_increment
            weights['stoch'] += weight_increment
            weights['price_pattern'] += weight_increment

        # Repeat for sell threshold if no sell signals
        elif df['Signal_Score'].lt(sell_threshold).sum() == 0:
            weights['sma_trend'] -= weight_increment
            weights['rsi'] -= weight_increment
            weights['stoch'] -= weight_increment
            weights['price_pattern'] -= weight_increment

    # Define buy/sell signals based on final weighted scores
    df['Combined_Signal'] = 0
    df.loc[df['Signal_Score'] >= buy_threshold, 'Combined_Signal'] = 1
    df.loc[df['Signal_Score'] <= sell_threshold, 'Combined_Signal'] = -1

    # Drop intermediate columns for cleanliness
    df = df.drop(columns=['sma_trend', 'rsi_signal', 'stoch_signal', 'price_pattern'])

    return df

def main():
    
    # load datasets into a dictionary
    data_dct = {}

    for i in CURRENCY_PAIRS:
        file_path = FOLDER_PATH + "/" + i[:3] + "_" + i[3:] + "_forex_data.csv"
        df = pd.read_csv(file_path)
    
        # rename column 1
        df = df.rename(columns={"Unnamed: 0": "Date"})
        
        symbol = i[:3] + "/" + i[3:]
        data_dct[symbol] = df
        
        
    # create signal column for all datasets (targets)
    for pair, df in data_dct.items():
        data_dct[pair] = create_signals(df)
        
    for pair, df in data_dct.items():
        print("\n")
        print(pair, df[["Combined_Signal"]])


    output_dir = DOWNLOADS_PATH + '/fx_signals_data'
    os.makedirs(output_dir, exist_ok=True)

    for pair, df in data_dct.items():
        clean_symbol = pair.replace('/', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"{clean_symbol}_fx_signals.csv")
        df.to_csv(file_path, index=True)
        print(f"Saved {pair} data to {file_path}")

    
    
if __name__ == "__main__":
    main()