import streamlit as st
from datetime import datetime  # Importing datetime

# Custom CSS for styling
st.markdown("""
<style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    .welcome-container {
        background-color: transparent;  /* No background color */
        padding: 0;  /* No padding */
        text-align: center;
        color: white;
        margin-top: 20px;
    }
    .welcome-title {
        font-size: 2.5em;
        color: #9f7aea;
        margin: 0;  /* No margin */
    }
    .welcome-subtitle {
        font-size: 1.8em;
        color: #fff;
        margin: 0;  /* No margin */
    }
    .welcome-text {
        font-size: 1.2em;
        color: #a0aec0;
        margin: 10px 0;  /* Small margin for spacing */
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.title("Welcome to the Forex & Stock Price Prediction Application")
st.markdown("""
<div class="welcome-container">
    <h1 class="welcome-title">Unlock the Future of Trading</h1>
    <h2 class="welcome-subtitle">Predict Forex and Stock Prices with Confidence</h2>
    <p class="welcome-text">
        This application leverages advanced machine learning models such as XGBoost and LightGBM to provide you with accurate predictions on stock and forex prices.
        Enter your data, explore historical trends, and visualize your predictions in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

# Image section for stock trading workspace
st.markdown('<div class="image-container"><img src="https://img.freepik.com/free-photo/stock-trading-workplace-background_1409-5545.jpg" width="600" alt="Stock trading workspace"/></div>', unsafe_allow_html=True)

# Continue welcome text
st.markdown("""
<div class="welcome-container">
    <p class="welcome-text">
        Whether you're a seasoned trader or just starting out, our intuitive interface and powerful analytics tools will help you make informed decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Current Date and Time
st.write(f"**Current Date and Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Instructions
st.write("""
### How to Use This Application:
1. **Input Parameters**: Navigate to the sidebar to enter the required parameters for your predictions.
2. **Make Predictions**: Click on the 'Predict Price Direction' button to generate predictions.
3. **Visualize Data**: Explore historical data visualizations and insights based on your inputs.
4. **Feature Engineering**: Gain deeper insights into your data with our feature engineering tools.
""")

# Call to action
st.markdown("### Ready to make your predictions? Scroll down to the sidebar to get started!")