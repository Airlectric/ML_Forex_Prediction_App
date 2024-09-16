import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Get the current file's directory
current_dir = os.path.dirname(__file__)

# Construct the full path to the model files
best_xgb_path_D1 = os.path.join(current_dir, 'models', 'Price_Prediction_Models', 'D1', 'best_xgb_D1.pkl')
best_lgb_path_D1 = os.path.join(current_dir, 'models', 'Price_Prediction_Models', 'D1', 'best_lgb_D1.pkl')

# Load the models
best_xgb = joblib.load(best_xgb_path_D1)
best_lgb = joblib.load(best_lgb_path_D1)

# Function to calculate engineered features
def calculate_engineered_features(df):
    # Moving Averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()

    # Relative Strength Index (RSI)
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    df['RSI'] = calculate_rsi(df['Close'], window=14)

    # Bollinger Bands
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA_20'] + (df['Std_20'] * 2)
    df['Lower_Band'] = df['MA_20'] - (df['Std_20'] * 2)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

    # Average True Range (ATR)
    def compute_atr(high, low, close, window=14):
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    df['ATR_14'] = compute_atr(df['High'], df['Low'], df['Close'], window=14)

    # Stochastic Oscillator
    def compute_stochastic(close, low, high, window=14):
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        stochastic_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        return stochastic_k
    df['Stochastic_Oscillator'] = compute_stochastic(df['Close'], df['Low'], df['High'], window=14)

    # Lagged Returns / Price Differences
    df['1_day_return'] = df['Close'].pct_change(1)
    df['7_day_return'] = df['Close'].pct_change(7)
    df['14_day_return'] = df['Close'].pct_change(14)

    # Volume Change and Moving Averages
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA_7'] = df['Volume'].rolling(window=7).mean()

    # Fill missing values
    df.fillna(method='bfill', inplace=True)
    
    return df

# Prediction function
def predict_price_direction(input_data):
    """
    Predicts the price direction using pre-trained XGBoost and LightGBM models.

    Parameters:
    - input_data: A dictionary containing the basic input features (Open, High, Low, Close, Volume).

    Returns:
    - A dictionary with predictions from both XGBoost and LightGBM models.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Feature Engineering
    df = calculate_engineered_features(df)

    # Drop unnecessary columns before prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'SMA_14', 'EMA_7', 'RSI', 'MA_20', 'Upper_Band', 
                'Lower_Band', 'OBV', 'ATR_14', 'Stochastic_Oscillator', '1_day_return', '7_day_return', 
                '14_day_return', 'Volume_Change', 'Volume_SMA_7']

    X = df[features]

    # Predict using XGBoost
    xgb_pred = best_xgb.predict(X)

    # Predict using LightGBM
    lgb_pred = best_lgb.predict(X)

    # LabelEncoder to map back to original labels ('Up', 'Down', 'Neutral')
    le = LabelEncoder()
    le.classes_ = np.array(['Down', 'Neutral', 'Up'])

    return {
        'XGBoost_Prediction': le.inverse_transform(xgb_pred)[0],
        'LightGBM_Prediction': le.inverse_transform(lgb_pred)[0]
    }

# # Example input from the user
# input_data = {
#     'Open': 1.1837,
#     'High': 1.1851,
#     'Low': 1.1821,
#     'Close': 1.1839,
#     'Volume': 100000
# }

# # Call the prediction function
# predictions = predict_price_direction(input_data)
# print(f"XGBoost Prediction: {predictions['XGBoost_Prediction']}")
# print(f"LightGBM Prediction: {predictions['LightGBM_Prediction']}")
