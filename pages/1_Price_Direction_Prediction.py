import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Placeholder: Replace with your prediction function import
from predictionModels.pricedirectionapi import predict_price_direction  # Placeholder for prediction function

# Helper function for price statistics
def display_statistics(predictions, open_price, high_price, low_price, close_price, volume):
    st.subheader("Price Statistics")

    stats = {
        'Open Price': open_price,
        'High Price': high_price,
        'Low Price': low_price,
        'Close Price': close_price,
        'Volume': volume,
        'Predicted Direction (XGBoost)': predictions['XGBoost_Prediction'],
        'Predicted Direction (LightGBM)': predictions['LightGBM_Prediction']
    }

    df_stats = pd.DataFrame([stats])
    st.table(df_stats)

# Helper function for visualizations
def display_visualizations(time_interval, open_price, high_price, low_price, close_price, volume):
    st.subheader("Stock Price Overview")

    # Prepare data for visualization
    df = pd.DataFrame({
        'Time': [f"{time_interval} Interval"],
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Close': [close_price],
        'Volume': [volume]
    })

    # Line Chart for OHLC
    st.write("### Price Movement (OHLC)")
    fig = px.line(df, x='Time', y=['Open', 'High', 'Low', 'Close'], title="Price Movement")
    st.plotly_chart(fig)

    # Bar chart for Volume
    st.write("### Volume Overview")
    fig = px.bar(df, x='Time', y='Volume', title="Volume")
    st.plotly_chart(fig)

# Main app layout
def app():
    # Set up page title and layout
    st.set_page_config(page_title="Stock Price Prediction", layout="centered")
    
    # Main Title
    st.title("Professional Stock Price Prediction Interface")

    # Date and Time
    st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Instructions
    st.write("""
        This application predicts the direction of stock prices using models such as XGBoost and LightGBM. 
        Enter the required stock data, select the prediction time interval, and view the predictions along with useful statistics.
    """)

    # Time interval selection
    st.subheader("Select Time Interval for Prediction:")
    time_interval = st.selectbox(
        "Choose Time Interval",
        ("30 mins", "1 hr", "4 hrs", "1 day"),
        index=0,
        help="Select the time interval for which you want to predict the stock price direction."
    )

    # Stock Data Input Form
    st.subheader("Enter Stock Data:")
    with st.form(key="stock_data_form"):
        open_price = st.number_input("Open Price", value=1.1837, help="Enter the opening price of the stock.")
        high_price = st.number_input("High Price", value=1.1851, help="Enter the highest price of the stock.")
        low_price = st.number_input("Low Price", value=1.1821, help="Enter the lowest price of the stock.")
        close_price = st.number_input("Close Price", value=1.1839, help="Enter the closing price of the stock.")
        volume = st.number_input("Volume", value=100000, help="Enter the stock volume.")
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict Price Direction")

    # Prediction Section
    if submit_button:
        # Prepare the input data
        input_data = {
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }

        # Call the prediction function
        predictions = predict_price_direction(input_data)
        
        # Display Predictions
        st.subheader(f"Predictions for {time_interval}:")
        st.write(f"**XGBoost Prediction:** {predictions['XGBoost_Prediction']}")
        st.write(f"**LightGBM Prediction:** {predictions['LightGBM_Prediction']}")

        # Display statistics
        display_statistics(predictions, open_price, high_price, low_price, close_price, volume)

        # Display visualizations
        display_visualizations(time_interval, open_price, high_price, low_price, close_price, volume)

# Run the Streamlit app
if __name__ == "__main__":
    app()
