import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import os
file_path = os.path.join('..','Historical Forex Data' ,'EURUSD_D1.csv')

def preprocess_data(df,freq):
    """
    Preprocess the raw dataset, resample, and create new features.
    """
    df = pd.read_csv(df)
    df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
    df.set_index('Time', inplace=True)
    
    # Resample data with daily frequency
    df = df.resample(freq).mean()
    df.fillna(method='ffill', inplace=True)
    df.index.freq = freq
    
    # Create additional features (e.g., ranges, momentum, volatility)
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Open_High_Range'] = df['High'] - df['Open']
    df['Open_Low_Range'] = df['Open'] - df['Low']
    df['Rolling_Volatility'] = df[['Open', 'High', 'Low']].std(axis=1).rolling(window=3).mean()
    df['True_Range'] = df.apply(lambda row: max(row['High'] - row['Low'], 
                                                abs(row['High'] - row['Open']), 
                                                abs(row['Low'] - row['Open'])), axis=1)
    
    df['SMA_Open_3'] = df['Open'].rolling(window=3).mean()  # 3-period SMA of Open
    df['SMA_High_3'] = df['High'].rolling(window=3).mean()  # 3-period SMA of High

    # 7. Price Momentum (for Open and High, lag of 1 period)
    df['Momentum_Open_1'] = df['Open'] - df['Open'].shift(1)
    df['Momentum_High_1'] = df['High'] - df['High'].shift(1)

    # 8. Exponential Moving Average (EMA) for Open (with a smoothing factor, alpha)
    df['EMA_Open'] = df['Open'].ewm(span=3, adjust=False).mean()  # 3-period EMA of Open

    # 9. Volume Change (percentage change)
    df['Volume_Change'] = df['Volume'].pct_change() * 100

    # 10. Moving Average of Volume (SMA)
    df['SMA_Volume_3'] = df['Volume'].rolling(window=3).mean()

    # 11. Lagged features for Open, High, Low, and Volume (lag of 1 period)
    df['Lagged_Open_1'] = df['Open'].shift(1)
    df['Lagged_High_1'] = df['High'].shift(1)
    df['Lagged_Low_1'] = df['Low'].shift(1)
    df['Lagged_Volume_1'] = df['Volume'].shift(1)

    window = 14  # Common default period for RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']


    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()


    df['Trend_Direction'] = np.where(df['Close'] > df['Open'], 1, -1)


    df['Returns'] = df['Close'].pct_change()
    df['Lagged_Returns_1'] = df['Returns'].shift(1)


    df['EMA_Volume'] = df['Volume'].ewm(span=3, adjust=False).mean()


    df['Range_5'] = df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min()

    df['Day_of_Week'] = df.index.dayofweek 
    df['Month'] = df.index.month
        
    return df

def create_pca_features(df, n_components=6):
    """
    Reduce dimensionality using PCA.
    """
    df.dropna(inplace=True)
    pca_columns = ['High_Low_Range', 'Open_High_Range', 'Open_Low_Range',
               'Rolling_Volatility', 'True_Range', 'SMA_Open_3',
               'SMA_High_3', 'Momentum_Open_1', 'Momentum_High_1',
               'EMA_Open', 'Volume_Change', 'SMA_Volume_3',
               'Lagged_Open_1', 'Lagged_High_1', 'Lagged_Low_1',
               'Lagged_Volume_1', 'RSI', 'EMA_12', 'EMA_26',
               'MACD', 'SMA_20', 'Bollinger_Upper',
               'Bollinger_Lower', 'Trend_Direction',
               'Returns', 'Lagged_Returns_1', 'EMA_Volume',
               'Range_5']

    pca_data = df[pca_columns]
    
    # Standardize the data
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(pca_data_scaled)
    
    # Combine PCA components with original features
    component_names = [f'Principal Component {i+1}' for i in range(n_components)]
    reduced_pca_df = pd.DataFrame(data=pca_components, columns=component_names)
    combined_df = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume','Day_of_Week', 'Month']].reset_index(drop=False),
                             reduced_pca_df.reset_index(drop=True)], axis=1)
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df.set_index('Time', inplace=True)
    
    return combined_df

def differencing(df, columns=None, max_differences=2):
    """
    Apply differencing to correct non-stationary nature of specified columns in a DataFrame.
    """
    def check_stationarity(series):
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 indicates stationarity

    def apply_differencing(series, max_differences):
        differenced_series = series.copy()
        for i in range(max_differences):
            if check_stationarity(differenced_series):
                break
            differenced_series = differenced_series.diff().dropna()
        return differenced_series

    df_differenced = df.copy()
    for col in (columns or df.columns):
        if not check_stationarity(df[col]):
            df_differenced[col] = apply_differencing(df[col], max_differences)
    return df_differenced.dropna()


def find_best_lag(df, max_lags=15):
    """
    Find the best lag order manually by fitting VAR models with different lags.

    Parameters:
    - df: pandas DataFrame with time series data.
    - max_lags: maximum number of lags to consider.

    Returns:
    - best_lag: optimal number of lags based on lowest AIC.
    """
    aic_values = []
    bic_values = []
    for lag in range(1, max_lags + 1):
        model = VAR(df)
        try:
            model_fit = model.fit(lag)
            aic = model_fit.aic
            bic = model_fit.bic
            aic_values.append(aic)
            bic_values.append(bic)
        except Exception as e:
            print(f"Error fitting VAR model with lag {lag}: {e}")
            aic_values.append(float('inf'))
            bic_values.append(float('inf'))

    best_lag = aic_values.index(min(aic_values)) + 1
    print(f"Best lag based on AIC: {best_lag}")
    return best_lag



def train_var_model(df, optimal_lags=10):
    """
    Train the VAR model and select the optimal number of lags.
    """
    df.dropna(inplace=True)
    print(df.dtypes)
    model = VAR(df)
    return model.fit(optimal_lags)

def forecast_var(model_fit,n_steps,original_data ,training_data):
    """
    Forecast future values based on the VAR model.
    """
    forecast = model_fit.forecast(y=training_data.values[-model_fit.k_ar:], steps=n_steps)
    forecast_index = pd.date_range(start=training_data.index[-1] + pd.Timedelta(days=1), periods=n_steps, freq='B')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=training_data.columns)
    
    last_actual_values = original_data.iloc[-1].values

    # Inverse differencing
    # Calculate cumulative sum of forecasted differences
    forecast_cumsum = np.cumsum(forecast_df.values, axis=0)

    # Step 3: Add last actual values to get predictions in original scale
    inverse_forecast = forecast_cumsum + last_actual_values

    # Create a DataFrame for inverse forecast
    inverse_forecast_df = pd.DataFrame(inverse_forecast, index=forecast_index, columns=forecast_df.columns)

    return inverse_forecast_df

def evaluate_forecast(actual, forecast):
    """
    Calculate evaluation metrics such as MAE, MSE, RMSE, and MAPE.
    """
    mae = np.mean(np.abs(actual - forecast))
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return mae, mse, rmse, mape

def plot_forecast(actual_last, forecast_df):
    """
    Plots the actual vs forecasted values for common columns between the actual and forecasted data.

    Parameters:
    - actual_last_20: pd.DataFrame of actual values (last 20 data points)
    - forecast_df: pd.DataFrame of forecasted values
    """
    
    # Find the common columns between actual and forecasted data
    common_columns = set(actual_last.columns).intersection(set(forecast_df.columns))

    # Prepare the data for plotting by extracting actual and forecasted values for common columns
    plot_data = {}
    for column in common_columns:
        if column not in ['Principal Component 0', 'Principal Component 1', 'Principal Component 2', 'Principal Component 3',
                          'Principal Component 4', 'Principal Component 5', 'Principal Component 6', 'Principal Component 7','Month','Day_of_Week']:
            actual_values = actual_last[column]
            forecasted_values = forecast_df[column]
            plot_data[column] = (actual_values, forecasted_values)

    # Create a figure for plotting
    fig, axs = plt.subplots(nrows=len(plot_data), ncols=1, figsize=(14, 7 * len(plot_data)), sharex=True)

    # Loop through each common column and create subplots
    for i, (column, (actual, forecasted)) in enumerate(plot_data.items()):
        if column not in ['Principal Component 0', 'Principal Component 1', 'Principal Component 2', 'Principal Component 3',
                          'Principal Component 4', 'Principal Component 5', 'Principal Component 6', 'Principal Component 7','Month','Day_of_Week']:
            axs[i].plot(actual.index, actual, label='Actual', color='blue', marker='o')  # Actual values
            axs[i].plot(forecasted.index, forecasted, label='Forecasted', color='orange', linestyle='--', marker='x')  # Forecasted values
            axs[i].set_title(f'Actual vs Forecasted: {column}')
            axs[i].set_ylabel(column)
            axs[i].legend()
            axs[i].grid()

    # Set common x-label for all subplots
    axs[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()


# Example usage:
def run_pipeline(df, steps=20, freq='4h', n_components=4):
    # Step 1: Preprocess the data
    df = preprocess_data(df,freq)
    
    # Step 2: Apply PCA for feature reduction
    df_pca = create_pca_features(df,n_components)
    
    # Step 3: Split the data into training and testing sets
    train_size = 0.999
    train_index = int(len(df_pca) * train_size)
    train_df = df_pca.iloc[:train_index]
    test_df = df_pca.iloc[train_index:]
    
    # Step 4: Handle non-stationarity
    train_df_stationary = differencing(train_df)

    optimal_lags = find_best_lag(train_df_stationary, max_lags=20)
    
    # Step 5: Train VAR model
    model_fit = train_var_model(train_df_stationary,optimal_lags)
    
    # Step 6: Forecast
    forecast_df = forecast_var(model_fit, steps,train_df, train_df_stationary)
    print(forecast_df)
    
    # Step 7: Evaluate
    actual_last = test_df.iloc[:steps]
    for column in forecast_df.columns:
        mae, mse, rmse, mape = evaluate_forecast(actual_last[column], forecast_df[column])
        print(f"{column} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Step 8: Plot results
    plot_forecast(actual_last, forecast_df)

# Call the pipeline
run_pipeline(file_path, steps=10, freq='D', n_components=6)
