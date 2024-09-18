import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import matplotlib.colors as mcolors
import mplfinance as mpf
import os



# Import prediction function for price direction
from predictionModels.pricedirectionapi import predict_price_direction

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
        df = pd.read_csv(file_path, header=None)
        if not all(col in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'] for col in df.columns[:6]):
            df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df
    else:
        st.error("Historical data not found for the selected currency pair and time frame.")
        return None

# Function to download a template CSV file
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

# Function to save uploaded file to datasets folder
def save_uploaded_file(uploaded_file, currency_pair, time_frame):
    directory = os.path.join('datasets', f'{currency_pair}_MetaTrader_CSV')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{currency_pair}_{time_frame}.csv')
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit interface
st.title('Forex Price Direction Prediction')

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
    time_frame = st.sidebar.selectbox("Select Time Frame", ["M1", "M15", "M30", "H1", "H4", "D1"])
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

    # Customize input for prediction (editing first row of the data)
    st.subheader("Customize Input for Prediction")
    first_row_data = editable_data.iloc[0]

    open_price = st.number_input('Open Price', value=float(first_row_data['Open']), format="%.5f")
    high_price = st.number_input('High Price', value=float(first_row_data['High']), format="%.5f")
    low_price = st.number_input('Low Price', value=float(first_row_data['Low']), format="%.5f")
    close_price = st.number_input('Close Price', value=float(first_row_data['Close']), format="%.5f")
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
    if st.button("Predict Price Direction"):
        st.write('## Predictions')
        user_inputs = {
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }
        predictions = predict_price_direction(user_inputs)
        
                
        # Function to format and color-code predictions with more engaging design
        def format_prediction(model_name, prediction):
            # Determine color and emoji based on prediction
            if prediction == 'High':
                color = '#4CAF50'  # Green for high prediction
                emoji = '📈'  # Up arrow emoji
            elif prediction == 'Low':
                color = '#f44336'  # Red for low prediction
                emoji = '📉'  # Down arrow emoji
            else:
                color = '#9e9e9e'  # Gray for neutral prediction
                emoji = '⚪'  # Neutral emoji

            # Style the output as a badge-like appearance with bold prediction and icons
            return f"""
            <div style='background-color: {color}; border-radius: 8px; padding: 10px; margin-bottom: 10px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
                <span style='color: white; font-weight: bold; font-size: 18px;'>{model_name}: {prediction} {emoji}</span>
            </div>
            """

        # Display the predictions with creative, well-styled badges
        st.markdown(format_prediction("Predicted Price Direction (XGBoost)", predictions['XGBoost_Prediction']), unsafe_allow_html=True)
        st.markdown(format_prediction("Predicted Price Direction (LightGBM)", predictions['LightGBM_Prediction']), unsafe_allow_html=True)

    st.write('## Visualization of Trends based on current and historical data')
    recent_data = updated_data.head(30)  # Show the most recent 30 rows
    recent_data['Time'] = pd.to_datetime(recent_data['Time'], dayfirst=True)
    recent_data.set_index('Time', inplace=True)

    # Custom color map for creative and vibrant design
    colors = list(mcolors.TABLEAU_COLORS.values())

    # 1. **Candlestick Chart**
    st.write('### Candlestick Chart')
    try:
        fig, axlist = mpf.plot(recent_data, type='candle', volume=True, style='charles', title='Candlestick Chart', returnfig=True, 
                            mav=(7,14), # Adding 7 & 14-day moving averages
                            ylabel='Price', ylabel_lower='Volume')
        st.pyplot(fig)
        st.info("Candlestick charts visually represent the high, low, open, and close prices over a time period. It helps to easily spot trends and price movements.")
    except Exception as e:
        st.error(f"Error plotting candlestick chart: {e}")


    # 2. **Moving Averages (MA) Chart**
    st.write('### Moving Averages (MA) Over Time')
    recent_data['MA_7'] = recent_data['Close'].rolling(window=7).mean()
    recent_data['MA_14'] = recent_data['Close'].rolling(window=14).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent_data.index, recent_data['Close'], label='Close Price', color=colors[0], linewidth=2)
    ax.plot(recent_data.index, recent_data['MA_7'], label='7-Day MA', color=colors[1], linestyle='--')
    ax.plot(recent_data.index, recent_data['MA_14'], label='14-Day MA', color=colors[2], linestyle='--')
    ax.set_title('Close Price with Moving Averages', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("Moving Averages (MAs) smooth out price data to help spot trends more easily. The 7-day MA shows shorter-term trends, while the 14-day MA shows longer-term trends.")


    # 3. **Bollinger Bands**
    st.write('### Bollinger Bands')
    rolling_mean = recent_data['Close'].rolling(window=20).mean()
    rolling_std = recent_data['Close'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent_data.index, recent_data['Close'], label='Close Price', color=colors[3])
    ax.plot(recent_data.index, upper_band, label='Upper Band', color=colors[1], linestyle='--')
    ax.plot(recent_data.index, lower_band, label='Lower Band', color=colors[1], linestyle='--')
    ax.fill_between(recent_data.index, lower_band, upper_band, color=colors[4], alpha=0.2)
    ax.set_title('Bollinger Bands', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("Bollinger Bands measure volatility. When prices move towards the upper or lower bands, it might indicate that the market is overbought or oversold.")


    # 4. **RSI (Relative Strength Index)**
    st.write('### Relative Strength Index (RSI)')
    delta = recent_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent_data.index, rsi, label='RSI', color=colors[5], linewidth=2)
    ax.axhline(70, linestyle='--', alpha=0.7, color=colors[2])
    ax.axhline(30, linestyle='--', alpha=0.7, color=colors[1])
    ax.fill_between(recent_data.index, 30, 70, color=colors[4], alpha=0.1)
    ax.set_title('Relative Strength Index (RSI)', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('RSI', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("The RSI measures momentum. Values above 70 indicate that the asset might be overbought, and values below 30 suggest it could be oversold.")


    # 5. **Cumulative Return**
    st.write('### Cumulative Return Over Time')
    initial_price = recent_data['Close'].iloc[0]
    cumulative_return = (recent_data['Close'] / initial_price - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent_data.index, cumulative_return, label='Cumulative Return (%)', color=colors[0], linewidth=2)
    ax.fill_between(recent_data.index, cumulative_return, color=colors[3], alpha=0.3)
    ax.set_title('Cumulative Return Over Time', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Cumulative Return (%)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("Cumulative return shows the growth or decline of an investment over time. It reflects how much profit or loss would have been made if an initial investment had been made at the start.")


    # 6. **Price Direction Over Time**
    st.write('### Price Direction Over Time')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate price direction (-1 for down, +1 for up)
    price_direction = np.sign(recent_data['Close'].diff())

    # Add a 5-period moving average to smooth the direction line
    moving_avg = price_direction.rolling(window=5).mean()

    # Plot shaded area and line chart
    ax.fill_between(recent_data.index, price_direction, where=(price_direction > 0), color='green', alpha=0.3, label='Up')
    ax.fill_between(recent_data.index, price_direction, where=(price_direction < 0), color='red', alpha=0.3, label='Down')
    ax.plot(recent_data.index, moving_avg, color='blue', linewidth=2, label='5-period Moving Avg')

    # Title and labels
    ax.set_title('Price Direction Over Time', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Direction (Up=1, Down=-1)', fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.info("Price direction helps users quickly identify if the market is generally moving up or down. A value of 1 indicates upward movement, while -1 indicates downward movement.")


    # 7. **Volume Over Time**
    st.write('### Volume Over Time')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a color gradient based on volume size
    cmap = plt.get_cmap('YlGnBu')
    norm = mcolors.Normalize(vmin=recent_data['Volume'].min(), vmax=recent_data['Volume'].max())
    colors = [cmap(norm(v)) for v in recent_data['Volume']]

    # Bar chart with color gradient
    ax.bar(recent_data.index, recent_data['Volume'], color=colors)

    # Highlight the highest volume point
    max_volume_idx = recent_data['Volume'].idxmax()
    ax.bar(max_volume_idx, recent_data['Volume'].max(), color='red', label='Highest Volume')

    # Title and labels
    ax.set_title('Volume Over Time', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Volume', fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.info("Volume shows the number of trades made during a period. Higher volumes often indicate strong trends or price movements.")


    # 8. **Correlation Matrix**
    st.write('### Correlation Between Features')
    correlation = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)

    ax.set_yticks(range(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns)

    ax.set_title('Correlation Heatmap', fontsize=16)
    st.pyplot(fig)
    st.info("A correlation matrix helps users see how different price factors (e.g., Open, Close, Volume) are related. A strong positive correlation (closer to 1) means the two features move together, while a strong negative correlation (closer to -1) means they move in opposite directions.")

else:
    st.error("No historical data available. Please upload a CSV file or select an existing dataset.")

# Download template button
st.sidebar.subheader("Download Template")
if st.sidebar.button("Download Template CSV"):
    template_file_path = download_template()
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