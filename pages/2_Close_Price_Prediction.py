import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Import functions from the other script
from predictionModels.closepriceapi import make_predictions, calculate_engineered_features

# Function to check input ranges
def validate_inputs(open_price, high_price, low_price, volume):
    errors = []
    if open_price <= 0:
        errors.append("Open Price must be positive.")
    if high_price <= 0 or high_price < open_price:
        errors.append("High Price must be positive and greater than or equal to Open Price.")
    if low_price <= 0 or low_price > open_price:
        errors.append("Low Price must be positive and less than or equal to Open Price.")
    if volume < 0:
        errors.append("Volume must be non-negative.")
    return errors

# Streamlit interface
st.title('Forex Prediction Application')

st.write("Enter the feature values below. Ensure that values are within reasonable ranges.")

# User inputs
open_price = st.number_input('Open Price', min_value=0.0, format="%.4f")
high_price = st.number_input('High Price', min_value=0.0, format="%.4f")
low_price = st.number_input('Low Price', min_value=0.0, format="%.4f")
volume = st.number_input('Volume', min_value=0)

# Time interval selection
st.subheader("Select Time Interval for Prediction:")
time_interval = st.selectbox(
    "Choose Time Interval",
    ("30 mins", "1 hr", "4 hrs", "1 day"),
    index=0,
    help="Select the time interval for which you want to predict the stock price direction."
)

# Validate inputs
errors = validate_inputs(open_price, high_price, low_price, volume)
if errors:
    st.write("### Validation Errors")
    for error in errors:
        st.write(f"- {error}")
    st.stop()

user_inputs = {
    'Open': open_price,
    'High': high_price,
    'Low': low_price,
    'Volume': volume
}

# Display user inputs
st.write('### Input Values')
st.write(user_inputs)

# Make predictions
st.write('## Predictions')
predictions = make_predictions(user_inputs)
for model, prediction in predictions.items():
    st.write(f"{model}: {prediction:.5f}")

# Plotting historical data
st.write('## Historical Data Visualization')

# Simulate historical data
date_range = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
data = {
    'Time': date_range,
    'Open': np.random.uniform(1.1000, 1.1100, size=30),
    'High': np.random.uniform(1.1000, 1.1200, size=30),
    'Low': np.random.uniform(1.0900, 1.1100, size=30),
    'Close': np.random.uniform(1.0950, 1.1150, size=30),
    'Volume': np.random.randint(1000, 2000, size=30)
}
df = pd.DataFrame(data)
df.set_index('Time', inplace=True)

# Candlestick chart
st.write('### Candlestick Chart')
fig, ax = plt.subplots(figsize=(12, 6))
mpf.plot(df, type='candle', volume=True, style='charles', title='Candlestick Chart', ax=ax)
st.pyplot(fig)

# Line chart for closing prices
st.write('### Closing Prices Over Time')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=2)
ax.set_title('Closing Prices Over Time', fontsize=16)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Close Price', fontsize=14)
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
st.pyplot(fig)

# Feature engineering
st.write('### Feature Engineering Insights')
df_features = calculate_engineered_features(df.copy())
st.write(df_features.describe())

# Custom styling
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
