import joblib
import numpy as np
import pandas as pd
import os

# Get the current file's directory
current_dir = os.path.dirname(__file__)

# Construct the full paths to the model files
best_lr_path = os.path.join(current_dir, 'models', 'Close_Price_Models', 'D1', 'best_lr_D1.pkl')
# best_rf_path = os.path.join(current_dir, 'models', 'Close_Price_Models', 'D1', 'best_rf_D1.pkl')
best_xgbr_path = os.path.join(current_dir, 'models', 'Close_Price_Models', 'D1', 'best_xgbr_D1.pkl')

# Load the models
best_lr = joblib.load(best_lr_path)
# best_rf = joblib.load(best_rf_path)
best_xgbr = joblib.load(best_xgbr_path)


def calculate_engineered_features(df):
    """
    Calculate feature-engineered columns based on initial features.
    
    Parameters:
    df (pd.DataFrame): DataFrame with basic feature columns.
    
    Returns:
    pd.DataFrame: DataFrame with feature-engineered columns added.
    """
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Open_High_Range'] = df['High'] - df['Open']
    df['Open_Low_Range'] = df['Open'] - df['Low']
    df['Rolling_Volatility'] = df[['Open', 'High', 'Low']].std(axis=1).rolling(window=3).mean()
    df['True_Range'] = df.apply(lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Open']), abs(row['Low'] - row['Open'])), axis=1)
    df['SMA_Open_3'] = df['Open'].rolling(window=3).mean()
    df['SMA_High_3'] = df['High'].rolling(window=3).mean()
    df['Momentum_Open_1'] = df['Open'] - df['Open'].shift(1)
    df['Momentum_High_1'] = df['High'] - df['High'].shift(1)
    df['EMA_Open'] = df['Open'].ewm(span=3, adjust=False).mean()
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    df['SMA_Volume_3'] = df['Volume'].rolling(window=3).mean()
    df['Lagged_Open_1'] = df['Open'].shift(1)
    df['Lagged_High_1'] = df['High'].shift(1)
    df['Lagged_Low_1'] = df['Low'].shift(1)
    df['Lagged_Volume_1'] = df['Volume'].shift(1)
    
    return df


def make_predictions(user_inputs):
    """
    Takes user inputs, preprocesses them, and feeds them into trained models to make predictions.
    
    Parameters:
    user_inputs (dict): A dictionary of user inputs corresponding to the features.
    
    Returns:
    dict: A dictionary containing predictions from each model.
    """
    
    # Step 1: Define the feature names
    feature_names = ['Open', 'High', 'Low', 'Volume']
    
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([user_inputs], columns=feature_names)
    
    # Calculate feature-engineered columns
    input_data = calculate_engineered_features(input_data)
    
    # Ensure that all numeric columns are in the correct dtype (float)
    input_data = input_data.astype(float)

    # Define the columns that should be present in the model
    all_columns = ['Open', 'High', 'Low', 'Volume', 
                   'High_Low_Range', 'Open_High_Range', 'Open_Low_Range', 
                   'Rolling_Volatility', 'True_Range', 'SMA_Open_3', 
                   'SMA_High_3', 'Momentum_Open_1', 'Momentum_High_1', 
                   'EMA_Open', 'Volume_Change', 'SMA_Volume_3', 
                   'Lagged_Open_1', 'Lagged_High_1', 'Lagged_Low_1', 
                   'Lagged_Volume_1']

    # Ensure the DataFrame contains all required columns
    for feature in all_columns:
        if feature not in input_data.columns:
            input_data[feature] = np.nan

    # Fill NaN values with 0 or any other appropriate method
    input_data = input_data.fillna(0)
    
    # Step 2: Ensure the input columns match the order of features used during training
    input_data = input_data[all_columns]
    
    # Step 3: Make predictions using each model
    predictions = {
        "Linear Regression Prediction": best_lr.predict(input_data)[0],
        # "Random Forest Prediction": best_rf.predict(input_data)[0],
        "XGBoost Prediction": best_xgbr.predict(input_data)[0]
    }
    
    return predictions
