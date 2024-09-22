import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from datetime import datetime
import os

# Import necessary functions
from predictionModels.vecmapi import pipeline,plot_forecast_streamlit,evaluate_forecast

# Function to fetch historical data from the datasets folder
def fetch_historical_data(currency_pair, time_frame):
    file_path = os.path.join('datasets', f'{currency_pair}_{time_frame}.csv')
    if os.path.exists(file_path):
        st.success('Historical data found for the selected currency pair and time frame.')
        return pd.read_csv(file_path, header=None)  # No header to be treated as raw data
    else:
        st.error("Historical data not found for the selected currency pair and time frame.")
        return None
def display_evaluation_metrics(mae, mse, rmse, mape, column_name):
    """
    Displays the evaluation metrics (MAE, MSE, RMSE, MAPE) for a given feature.
    """
    st.subheader(f"📊 {column_name} Forecast Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    col4.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
    
    st.divider()


# Streamlit interface
st.title('General Forex Prediction')

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Option to upload file or select from existing datasets
upload_option = st.sidebar.radio("Choose Data Source", ("Upload CSV", "Select Existing"))

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        historical_data = pd.read_csv(uploaded_file, header=None)
    else:
        historical_data = None
else:
    currency_pair = st.sidebar.selectbox("Select Currency Pair", ["EURUSD", "GBPUSD", "USDJPY"])
    time_frame = st.sidebar.selectbox("Select Time Frame", ["M30", "H1", "H4", "D1"])
    historical_data = fetch_historical_data(currency_pair, time_frame)

# Custom styling for the Streamlit app
st.markdown("""
<style>
.reportview-container {
   background: #f5f5f5;
}
.block-container {
   padding: 2rem;
}
.stButton>button {
   background-color: #4CAF50;
   color: white;
   border: none;
   padding: 15px 32px;
   text-align: center;
   text-decoration: none;
   display: inline-block;
   font-size: 16px;
   margin: 4px 2px;
   cursor: pointer;
   border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# If historical data is available, display it and allow user edits
if historical_data is not None:
    st.header("Edit Historical Data Before Prediction")
    
    # Assign temporary headers for user clarity
    historical_data.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Show editable data table with the full dataset
    edited_df = st.data_editor(historical_data.reset_index(drop=True))

    st.header("Customize Prediction Parameters")

    # Explanation for 'steps'
    st.info("""
    **Steps**: Number of steps ahead to predict. 
    A higher number means predictions further into the future, but with potentially more uncertainty.
    """)
    steps = st.slider("Select the number of steps ahead for prediction", 1, 20, 5)


    # Explanation for 'prediction_freq'
    st.info("""
    **Prediction Frequency**: This specifies the frequency of the forecasted data. 
    For example, 'B' for business days, 'D' for daily.
    """)
    prediction_freq = st.selectbox("Select prediction frequency", options=['D', 'H', 'B'], index=2)

    # Explanation for 'n_components'
    st.info("""
    **Number of Components**: This is used for dimensionality reduction (such as PCA) in the feature space.
    A higher number of components can capture more information from the original data but may add complexity.
    """)
    n_components = st.slider("Select the number of components", 1, 10, 3)

        # New slider for train-test split control
    st.info("""
    **Train-Test Split Ratio**: This controls where the forecasting begins, determining the proportion of data used for training.
    Adjusting this value changes how much data is used to train the model and the pointat which forecasting begins.
    """)
    train_test_ratio = st.slider("Select the train-test split ratio", 0.8, 1.0, 0.9996, step=0.0001)
        

    if st.button("Run Prediction"):
        # Save the edited dataset temporarily without headers
        file_path = 'datasets/edited_data.csv'
        edited_df.to_csv(file_path, index=False, header=False)  # Save without headers
        
        # Run the prediction pipeline with the CSV file
        forecast_result, actual = pipeline(file_path,train_test_ratio, steps=steps, freq='D', prediction_freq=prediction_freq, n_components=n_components)

        if forecast_result is not None and not forecast_result.empty:
            # Update the session state with the new data
            st.session_state['forecast'] = forecast_result
            st.session_state['actual'] = actual 
            st.session_state['historical_data'] = edited_df

        
            st.rerun()
    forecast=None
    actual=None
    # Step 3: If forecast results are available, show them with visualizations
    if 'forecast' in st.session_state:
        forecast_result = st.session_state['forecast']
        actual = st.session_state['actual']
        
        # Calculate Price Direction
        forecast_result['Price Direction'] = forecast_result['Close'].diff().apply(
            lambda x: 'Uptrend' if x > 0 else ('Downtrend' if x < 0 else 'No Change')
        )
        forecast=forecast_result


    if forecast is not None and not forecast_result.empty:
        st.write("### Forecast Results")
        st.dataframe(forecast_result[['Open', 'High', 'Low', 'Close', 'Volume', 'Price Direction']])

        # Plotting visualizations of the forecast
        st.write("### Forecast Visualizations")

        st.write("### Interactive Forex Forecast: Unveiling Trends and Divergences")
        if forecast is not None:
            plot_forecast_streamlit(actual,forecast)
        

        
        # Example: Run and display metrics for Open, Low, High, Close, Volume
        forecast_metrics = {
            'Open': evaluate_forecast(actual[['Open']], forecast_result[['Open']]),
            'Low': evaluate_forecast(actual[['Low']], forecast_result[['Low']]),
            'High': evaluate_forecast(actual[['High']], forecast_result[['High']]),
            'Close': evaluate_forecast(actual[['Close']], forecast_result[['Close']]),
            'Volume': evaluate_forecast(actual[['Volume']], forecast_result[['Volume']])
        }

        # Output each set of metrics in a visually appealing format
        st.write("## 📈 Forex Price and Volume Forecast Evaluation")
        st.write(f"**Mean Absolute Error (MAE)**: Measures the average difference between the actual and predicted value. The lower the value, the better.")
        st.write(f"**Mean Squared Error (MSE)**: Measures how spread out the prediction errors are. A smaller number is better because it means fewer large errors.")
        st.write(f"**Root Mean Squared Error (RMSE)**: Similar to MSE but easier to interpret because it's in the same units as the original data (e.g., price). Again, smaller is better.")
        st.write(f"**Mean Absolute Percentage Error (MAPE)**: Shows the error as a percentage of the actual value. For example, a MAPE of 1% means the prediction is off by about 1% on average.")
        for feature, (mae, mse, rmse, mape) in forecast_metrics.items():

            display_evaluation_metrics(mae, mse, rmse, mape, feature)

     

        # Candlestick Chart using Plotly for better interactivity
        st.write("### Candlestick Chart")
        candlestick_fig = go.Figure(data=[go.Candlestick(x=forecast_result.index,
                                                        open=forecast_result['Open'],
                                                        high=forecast_result['High'],
                                                        low=forecast_result['Low'],
                                                        close=forecast_result['Close'])])
        candlestick_fig.update_layout(title='Candlestick Chart of Forecasted Prices',
                                    xaxis_title='Date',
                                    yaxis_title='Price',
                                    xaxis_rangeslider_visible=False)
        st.plotly_chart(candlestick_fig)

        # Volume Chart
        st.write("### Interactive Volume Chart")
        volume_fig = go.Figure(data=[go.Bar(x=forecast_result.index,
                                            y=forecast_result['Volume'],
                                            marker_color='purple')])
        volume_fig.update_layout(title='Interactive Trading Volume Over Time',
                                xaxis_title='Time',
                                yaxis_title='Volume',
                                xaxis_rangeslider_visible=True,
                                template='plotly_white')
        st.plotly_chart(volume_fig)

        # Moving Average Visualization
        st.write("### Moving Average Visualization")
        moving_average_window = st.slider("Select Moving Average Window Size", min_value=1, max_value=30, value=5)

        # Calculate Moving Average
        forecast_result['Moving Average'] = forecast_result['Close'].rolling(window=moving_average_window, min_periods=1).mean()

        # Create a Plotly figure
        fig_moving_avg = go.Figure()

        # Add forecasted Close Price line
        fig_moving_avg.add_trace(go.Scatter(
            x=forecast_result.index, 
            y=forecast_result['Close'], 
            mode='lines+markers',
            name='Forecasted Close Price',
            line=dict(color='royalblue', width=2),
            marker=dict(color='royalblue', size=6, symbol='circle', line=dict(color='black', width=1)),
            hovertemplate='Date: %{x}<br>Close Price: %{y}<extra></extra>'
        ))

        # Add Moving Average line
        fig_moving_avg.add_trace(go.Scatter(
            x=forecast_result.index, 
            y=forecast_result['Moving Average'], 
            mode='lines',
            name=f'{moving_average_window}-Period Moving Average',
            line=dict(color='orange', dash='dash', width=2),
            hovertemplate='Date: %{x}<br>Moving Average: %{y}<extra></extra>'
        ))

        # Highlight where price crosses above moving average (uptrend)
        crossover_points = forecast_result[forecast_result['Close'] > forecast_result['Moving Average']].index
        fig_moving_avg.add_trace(go.Scatter(
            x=crossover_points, 
            y=forecast_result.loc[crossover_points, 'Close'], 
            mode='markers', 
            name='Uptrend Cross',
            marker=dict(color='green', size=10, symbol='triangle-up', line=dict(color='black', width=1)),
            hovertemplate='Date: %{x}<br>Uptrend Close Price: %{y}<extra></extra>'
        ))

        # Highlight the mean Close price as a reference line
        mean_close = forecast_result['Close'].mean()
        fig_moving_avg.add_trace(go.Scatter(
            x=forecast_result.index, 
            y=[mean_close]*len(forecast_result.index), 
            mode='lines', 
            name='Mean Close Price',
            line=dict(color='gray', width=1, dash='dot'),
            hovertemplate='Date: %{x}<br>Mean Close Price: %{y}<extra></extra>'
        ))

        # Update layout for interactivity and titles
        fig_moving_avg.update_layout(
            title=f'{moving_average_window}-Period Moving Average and Forecasted Close Price',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )

        # Display the Plotly chart
        st.plotly_chart(fig_moving_avg)

else:
    st.error("No historical data available.")
    
    # Include an image or animation that draws attention to the sidebar
    st.info("Please toggle the sidebar to upload a CSV file or select an existing dataset.")

    # Add a spinner animation as an attention grabber
    with st.spinner('Waiting for user to select or upload data...'):
        time.sleep(2)  # Simulates some waiting time
    
    # Optionally include an emoji or an image
    st.markdown("""
        <style>
        .flashing {
            animation: flash 1.5s infinite;
        }
        @keyframes flash {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        <div class="flashing" style="text-align: center;">
            📂 Click the arrow on the top-left to upload or load data 📂
        </div>
    """, unsafe_allow_html=True)
