import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order, select_coint_rank
import streamlit as st
import plotly.graph_objects as go
# import os
# file_path = os.path.join('..','Historical Forex Data' ,'EURUSD_D1.csv')


def load_data(filepath, freq='D'):
    raw = pd.read_csv(filepath, header=None)

    # Set column names if the first row has headers
    if raw.iloc[0].str.contains('Time').any():
        raw.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        raw = raw[1:]
    else:
        raw.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    # Attempt to convert 'Time' column to datetime with multiple formats
    try:
        raw['Time'] = pd.to_datetime(raw['Time'], format='%Y-%d-%m %H:%M', errors='raise')
    except ValueError:
        try:
            # Handle a more common format in case the first fails
            raw['Time'] = pd.to_datetime(raw['Time'], format='%Y-%m-%d %H:%M', errors='raise')
        except ValueError:
            # Handle mixed formats or unknown patterns with 'coerce' to NaT
            raw['Time'] = pd.to_datetime(raw['Time'], errors='coerce')

    # Drop rows where 'Time' could not be parsed (NaT values)
    raw.dropna(subset=['Time'], inplace=True)
    raw.set_index('Time', inplace=True)

    # Resample the data at the specified frequency
    df = raw.resample(freq).mean(numeric_only=True)
    
    # Forward fill any missing values
    df.fillna(method='ffill', inplace=True)
    
    df.index.freq = freq

    return df



def feature_engineering(df):
    # Creating new features
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Open_High_Range'] = df['High'] - df['Open']
    df['Open_Low_Range'] = df['Open'] - df['Low']
    df['Rolling_Volatility'] = df[['Open', 'High', 'Low']].std(axis=1).rolling(window=3).mean()
    df['True_Range'] = df.apply(lambda row: max(row['High'] - row['Low'], 
                                                abs(row['High'] - row['Open']), 
                                                abs(row['Low'] - row['Open'])), axis=1)
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

    # RSI Calculation
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD and Bollinger Bands
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()

    # Trend Direction
    df['Trend_Direction'] = np.where(df['Close'] > df['Open'], 1, -1)

    # Return and Range features
    df['Returns'] = df['Close'].pct_change()
    df['Lagged_Returns_1'] = df['Returns'].shift(1)
    df['EMA_Volume'] = df['Volume'].ewm(span=3, adjust=False).mean()
    df['Range_5'] = df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min()

       
    df['Day_of_Week'] = df.index.dayofweek 
    df['Month'] = df.index.month

    df.dropna(inplace=True)
    return df


def apply_pca(df, n_components=6):
    pca_columns = ['High_Low_Range', 'Open_High_Range', 'Open_Low_Range', 'Rolling_Volatility',
                   'True_Range', 'SMA_Open_3', 'SMA_High_3', 'Momentum_Open_1', 'Momentum_High_1',
                   'EMA_Open', 'Volume_Change', 'SMA_Volume_3', 'Lagged_Open_1', 'Lagged_High_1',
                   'Lagged_Low_1', 'Lagged_Volume_1', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 
                   'SMA_20', 'Bollinger_Upper', 'Bollinger_Lower', 'Trend_Direction', 
                   'Returns', 'Lagged_Returns_1', 'EMA_Volume', 'Range_5']

    pca_data = df[pca_columns]
    
    # Standardizing the data
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    # Applying PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(pca_data_scaled)

    # Create a DataFrame with PCA components
    pca_df = pd.DataFrame(data=pca_components, 
                          columns=[f'Principal Component {i+1}' for i in range(n_components)])
    
    # Combine PCA features with original data
    combined_df = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume','Day_of_Week', 'Month']].reset_index(drop=False),
                             pca_df.reset_index(drop=True)], axis=1)
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df.set_index('Time', inplace=True)
    
    return combined_df



def train_vecm(train_data,freq='D'):
    train_data.dropna(inplace=True)
    train_data.index.freq = freq
    lag_order = select_order(data=train_data, maxlags=5, deterministic="ci", seasons=5)
    rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=lag_order.aic, method="trace", signif=0.1)
    
    vecm_model = VECM(train_data, deterministic="ci", k_ar_diff=lag_order.aic, 
                      coint_rank=rank_test.rank)
    vecm_res = vecm_model.fit()            
    
    return vecm_res


def prepare_and_forecast_sarimax(data, n_periods=10):

    n_periods = n_periods * 2
    
    data.dropna(inplace=True)
    
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        raise ValueError("Data index must be a DateTime index.")
    
    data = data.asfreq('D')
    
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)

    model = SARIMAX(data['Volume'], exog=data[['Close', 'Open', 'High', 'Low']], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    last_exog = data[['Close', 'Open', 'High', 'Low']].iloc[-n_periods:]
    
    if last_exog.isnull().any().any():
        print("NaN values found in exogenous variables:")
        print(last_exog[last_exog.isnull().any(axis=1)])
        return None

    forecast_values = model_fit.forecast(steps=n_periods, exog=last_exog)

    if isinstance(forecast_values, np.ndarray) and forecast_values.ndim > 1:
        forecast_values = forecast_values.flatten()
    forecast_values = forecast_values.to_frame(name='Forecasted_Volume')
    last_date = data.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='B')
    
    # Create DataFrame with forecasted values
    forecast_df = pd.DataFrame(forecast_values, index=forecast_index, columns=['Forecasted_Volume'])
    
    return forecast_df


def forecast(vecm_model,data, steps=5, freq='B',vol_df=None):
    forecast = vecm_model.predict(steps=steps)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                                   periods=steps, freq=freq)
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

    if 'Forecasted_Volume' in vol_df.columns:
        forecast_df.drop(columns=['Volume'], inplace=True)
        forecast_df['Volume'] = vol_df['Forecasted_Volume']
    
    return forecast_df

def evaluate_forecast(actual, forecast):
    # Get the columns of interest based on the DataFrame passed
    columns_of_interest = actual.columns.intersection(['Open', 'High', 'Low', 'Close', 'Volume']).tolist()
    
    actual_filtered = actual[columns_of_interest]
    forecast_filtered = forecast[columns_of_interest]

    actual_aligned, forecast_aligned = actual_filtered.align(forecast_filtered, join='inner')

    mae = mean_absolute_error(actual_aligned, forecast_aligned)
    mse = mean_squared_error(actual_aligned, forecast_aligned)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100

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




def pipeline(filepath,train_test_ratio, steps=5, freq='D', prediction_freq='B', n_components=6):
    # Step 1: Load and preprocess data
    df = load_data(filepath, freq=freq)
    
    # Step 2: Feature engineering
    df = feature_engineering(df)
    
    # Step 3: Apply PCA
    df_pca = apply_pca(df, n_components=n_components)
    
    # Step 4: Split into train and test sets
    train_size = int(train_test_ratio * len(df_pca))
    train_data = df_pca[:train_size]
    train_data = train_data.asfreq(freq) 
    test_data = df_pca[train_size:]

    
    # Step 6: Train the VECM model
    vecm_model = train_vecm(train_data,freq=freq)

    volume_df=prepare_and_forecast_sarimax(train_data, n_periods=steps)
    print(volume_df)
    print(type(volume_df))
    
    # Step 7: Forecasting
    forecast_df = forecast(vecm_model,train_data, steps=steps, freq=prediction_freq,vol_df=volume_df)
    print(forecast_df['Volume'])
    # Step 8: Evaluate forecast performance
    actual = test_data.iloc[:steps]
    # for col in actual.columns:
    #     mae, mse, rmse, mape = evaluate_forecast(actual[col], forecast_df[col])
    #     print(f"\nMetrics for '{col}':")
    #     print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

    plot_forecast(test_data, forecast_df)
    
    return forecast_df,test_data



def plot_forecast_streamlit(actual_last, forecast_df):
    """
    Plots the actual vs forecasted values for common columns between the actual and forecasted data in an interactive way using Plotly in Streamlit.

    Parameters:
    - actual_last: pd.DataFrame of actual values (last N data points)
    - forecast_df: pd.DataFrame of forecasted values
    """

    # Find the common columns between actual and forecasted data
    common_columns = set(actual_last.columns).intersection(set(forecast_df.columns))

    # Prepare the data for plotting by filtering out unwanted columns
    plot_data = {}
    exclude_columns = [
        'Principal Component 0', 'Principal Component 1', 'Principal Component 2', 
        'Principal Component 3', 'Principal Component 4', 'Principal Component 5', 
        'Principal Component 6', 'Principal Component 7', 'Month', 'Day_of_Week'
    ]
    
    for column in common_columns:
        if column not in exclude_columns:
            actual_values = actual_last[column]
            forecasted_values = forecast_df[column]
            plot_data[column] = (actual_values, forecasted_values)

    # Create a subplot for each common column
    for column, (actual, forecasted) in plot_data.items():
        st.write(f"### Actual vs Forecasted: {column}")

        # Create a Plotly figure
        fig = go.Figure()

        # Add actual values line
        fig.add_trace(go.Scatter(
            x=actual.index, 
            y=actual, 
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(color='blue', size=6, symbol='circle', line=dict(color='black', width=1)),
            hovertemplate='Date: %{x}<br>Actual Value: %{y}<extra></extra>'
        ))

        # Add forecasted values line
        fig.add_trace(go.Scatter(
            x=forecasted.index, 
            y=forecasted, 
            mode='lines+markers',
            name='Forecasted',
            line=dict(color='orange', dash='dash', width=2),
            marker=dict(color='orange', size=6, symbol='x', line=dict(color='black', width=1)),
            hovertemplate='Date: %{x}<br>Forecasted Value: %{y}<extra></extra>'
        ))

        # Highlight where forecast diverges from actual values
        divergences = forecasted - actual
        divergence_points = divergences[abs(divergences) > divergences.mean()].index
        fig.add_trace(go.Scatter(
            x=divergence_points, 
            y=forecasted.loc[divergence_points], 
            mode='markers', 
            name='Divergence Points',
            marker=dict(color='red', size=10, symbol='triangle-down', line=dict(color='black', width=1)),
            hovertemplate='Date: %{x}<br>Divergence Value: %{y}<extra></extra>'
        ))

        # Update layout for interactivity and clarity
        fig.update_layout(
            title=f'Actual vs Forecasted: {column}',
            xaxis_title='Date',
            yaxis_title=column,
            template='plotly_white',
            hovermode='x',
            legend_title="Legend",
            xaxis_rangeslider_visible=True  # Adds the range slider for interactive zoom
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


