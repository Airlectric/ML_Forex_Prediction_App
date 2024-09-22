import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
import matplotlib.colors as mcolors
import time
import os

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


# Function to fetch historical data from the datasets folder
def fetch_historical_data(currency_pair, time_frame):
    file_path = os.path.join('datasets', f'{currency_pair}_{time_frame}.csv')
    if os.path.exists(file_path):
        st.sidebar.success('Historical data found for the selected currency pair and time frame.')
        df = pd.read_csv(file_path, header=None)
        if not all(col in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'] for col in df.columns[:6]):
            df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df
    else:
        st.sidebar.error("Historical data not found for the selected currency pair and time frame.")
        return None


def download_template():
    template_data = {
        "Time": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "Open": [1.1837],
        "High": [1.1851],
        "Low": [1.1821],
        "Close": [1.1839],
        "Volume": [100000]
    }
    df_template = pd.DataFrame(template_data)
    template_path = 'datasets/template.csv'
    df_template.to_csv(template_path, index=False)
    return template_path


def save_uploaded_file(uploaded_file, currency_pair, time_frame):
    directory = os.path.join('datasets', f'{currency_pair}_MetaTrader_CSV')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{currency_pair}_{time_frame}.csv')
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit interface
st.title('Forex Close Price Prediction')

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Option to upload file or select from existing datasets
upload_option = st.sidebar.radio("Choose Data Source", ("Upload CSV", "Select Existing"))

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] < 6:
            st.error("Uploaded file does not have enough columns. Required columns: Time, Open, High, Low, Close, Volume.")
            st.stop()
        first_row = df.iloc[0]
        if all(item in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'] for item in first_row[:6]):
            df.columns = first_row
            df = df[1:]
        else:
            df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        historical_data = df
    else:
        historical_data = None
else:
    currency_pair = st.sidebar.selectbox("Select Currency Pair", ["EURUSD", "GBPUSD", "USDJPY"])
    time_frame = st.sidebar.selectbox("Select Time Frame", ["D1"])
    historical_data = fetch_historical_data(currency_pair, time_frame)

# Limit the number of rows for prediction (max 90 rows)
num_rows = st.sidebar.slider("Number of Rows for Prediction", min_value=1, max_value=90, value=30)

# If data is uploaded via file upload or selected from datasets
if historical_data is not None:
    st.header("Historical Data")
    st.write("You can edit the data below to customize your predictions.")
    
    # Show only the specified number of rows for editing, ordered by most recent date
    editable_data = historical_data.tail(num_rows).copy()
    editable_data = editable_data.iloc[::-1].reset_index(drop=True)
    
    # Allow users to edit the data directly
    editable_data = st.data_editor(editable_data, num_rows="dynamic")
    
    st.subheader("Customize Input for Prediction")
    first_row_data = editable_data.iloc[0]

    open_price = st.number_input('Open Price', value=float(first_row_data['Open']), format="%.5f")
    high_price = st.number_input('High Price', value=float(first_row_data['High']), format="%.5f")
    low_price = st.number_input('Low Price', value=float(first_row_data['Low']), format="%.5f")
    volume = st.number_input('Volume', value=int(first_row_data['Volume']), min_value=0)

    # Validate inputs
    errors = validate_inputs(open_price, high_price, low_price, volume)
    if errors:
        st.write("### Validation Errors")
        for error in errors:
            st.write(f"- {error}")
        st.stop()

    # Update the first row with new user inputs
    user_input_row = [first_row_data['Time'], open_price, high_price, low_price, first_row_data['Close'], volume]
    new_data = pd.DataFrame([user_input_row], columns=editable_data.columns)

    # Append the user input to the data and reorder by the most recent dates
    updated_data = pd.concat([new_data, editable_data.iloc[1:]], ignore_index=True)

    st.success("Your input has been appended to the latest dataset.")
    st.write(updated_data)

    # Button to trigger prediction
    if st.button("Predict Close Price"):
        with st.spinner('💹 Analyzing market data and generating your forecast...'):
            time.sleep(3) 
            st.write('## Predictions')
            user_inputs = {
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Volume': volume
            }
            predictions = make_predictions(user_inputs)
        
        # Define a custom style for model predictions
        st.markdown("""
            <style>
            .prediction-box {
                background-color: #f0f4f8;
                border-left: 6px solid #4CAF50;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 8px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            .model-name {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
            .prediction-value {
                font-size: 16px;
                color: #0275d8;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display predictions in a more organized and professional way
        for model, prediction in predictions.items():
            st.markdown(f"""
                <div class="prediction-box">
                    <span class="model-name">{model}:</span>
                    <span class="prediction-value">{prediction:.5f}</span>
                </div>
            """, unsafe_allow_html=True)


       

    # Custom color map using 'TABLEAU_COLORS' for vibrant graphs
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Plotting recent historical data
    st.write('## Historical Data Visualization')

    recent_data = updated_data.head(30)  # Show the most recent 30 rows
    recent_data['Time'] = pd.to_datetime(recent_data['Time'], dayfirst=True)
    recent_data.set_index('Time', inplace=True)
    plotting_data = recent_data.sort_values(by='Time').head(30)

    # 1. **Closing Prices Line Chart**
    if 'Close' in recent_data.columns:
        st.write('### Closing Prices Over Time')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(recent_data.index, recent_data['Close'], label='Close Price', color=colors[0], linewidth=2)
        ax.set_title('Closing Prices Over Time', fontsize=16)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Close Price', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.info("This line chart shows the closing price of the asset over the last 30 periods. It helps visualize the general trend.")
    else:
        st.info("Closing price data is not available for the line chart.")

    # 2. **Candlestick Chart with Volume**
    st.write('### Candlestick Chart')
    try:
        fig, axlist = mpf.plot(plotting_data, type='candle', volume=True, style='charles', title='Candlestick Chart',
                            mav=(7, 14), # Add 7 and 14-period moving averages
                            returnfig=True)
        st.pyplot(fig)
        st.info("Candlestick charts represent the open, high, low, and close prices. The green and red bars reflect the price movement within a time period, and the moving averages (7, 14 periods) help identify trends.")
    except Exception as e:
        st.error(f"Error plotting candlestick chart: {e}")

    # Feature engineering insights
    st.write('### Feature Engineering Insights')
    if all(col in editable_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        df_features = calculate_engineered_features(editable_data.copy())
        st.write(df_features.select_dtypes(include='number').describe())
    else:
        st.warning("Feature engineering requires 'Open', 'High', 'Low', 'Close', and 'Volume' columns.")
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

# Download template button
st.sidebar.subheader("Download Template")
template_file_path = None  

if st.sidebar.button("Generate Template CSV"):
    template_file_path = download_template()  
    st.sidebar.success("Template ready for download!")  

if template_file_path: 
    with open(template_file_path, "rb") as f:
        st.sidebar.download_button(
            label="Download Template CSV",
            data=f,
            file_name="template.csv",
            mime="text/csv"
        )


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
    background: linear-gradient(90deg, #00C6FF, #0072FF); /* Gradient from light to dark blue */
    color: white;
    border: none;
    padding: 15px 40px; /* Increased padding for a more luxurious feel */
    text-align: center;
    text-decoration: none;
    font-size: 18px; /* Larger font size for better readability */
    margin: 10px 2px; /* Slight margin increase for better spacing */
    cursor: pointer;
    border-radius: 50px; /* Rounded pill-shaped button */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); /* Soft shadow for a 3D effect */
    transition: all 0.3s ease-in-out; /* Smooth transition for hover effects */
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0072FF, #00C6FF); /* Reverse gradient on hover */
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); /* Slightly larger shadow on hover */
    transform: scale(1.05); /* Slightly enlarges the button when hovered */
}

.stButton>button:active {
    transform: scale(0.98); /* Button slightly shrinks when clicked */
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1); /* Lighter shadow on click */
}

</style>
""", unsafe_allow_html=True)
