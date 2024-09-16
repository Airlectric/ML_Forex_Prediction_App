import streamlit as st
from datetime import datetime

# Set up the page configuration
st.set_page_config(page_title="Forex & Stock Price Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .welcome-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .welcome-title {
        font-size: 2.5em;
        color: #4CAF50;
    }
    .welcome-subtitle {
        font-size: 1.5em;
        color: #555;
    }
    .welcome-text {
        font-size: 1.2em;
        color: #333;
        margin-top: 10px;
    }
    .cta-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin-top: 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Welcome Page Layout
st.title("Welcome to the Forex & Stock Price Prediction Application")

# Welcome message
st.markdown("""
<div class="welcome-container">
    <h1 class="welcome-title">Unlock the Future of Trading</h1>
    <h2 class="welcome-subtitle">Predict Forex and Stock Prices with Confidence</h2>
    <p class="welcome-text">
        This application leverages advanced machine learning models such as XGBoost and LightGBM to provide you with accurate predictions on stock and forex prices.
        Enter your data, explore historical trends, and visualize your predictions in real-time.
    </p>
    <p class="welcome-text">
        Whether you're a seasoned trader or just starting out, our intuitive interface and powerful analytics tools will help you make informed decisions.
    </p>
    <a class="cta-button" href="#prediction-section">Get Started!</a>
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

# A section to link to the prediction functionality
st.markdown("### Ready to make your predictions? Scroll down to the sidebar to get started!")