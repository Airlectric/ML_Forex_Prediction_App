import streamlit as st
from datetime import datetime  # Importing datetime

# Custom CSS for styling and responsiveness
st.markdown("""
<style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f4f9;  /* Light background for contrast */
        color: #333;  /* Dark text for readability */
    }
    .welcome-container {
        background-color: rgba(255, 255, 255, 0.9);  /* White background */
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        text-align: center;
        margin-top: 20px;
        width: 100%;
    }
    .welcome-title {
        font-size: 2.5em;
        color: #6c63ff; /* Primary color */
        margin: 0;
    }
    .welcome-subtitle {
        font-size: 1.8em;
        color: #333;
        margin: 10px 0;
    }
    .welcome-text {
        font-size: 1.2em;
        color: #666;
        margin: 10px 0;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        width: 100%;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s; /* Smooth scaling effect */
        max-width: 100%;
        height: auto;
    }
    img:hover {
        transform: scale(1.05); /* Scale on hover */
    }
    .notification {
        text-align: center;
        font-size: 1.2em;
        color: #6c63ff; /* Primary color */
        animation: flash 1.5s infinite; /* Flashing effect */
    }
    @keyframes flash {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2em;
        }
        .welcome-subtitle {
            font-size: 1.4em;
        }
        .welcome-text {
            font-size: 1em;
        }
        .image-container {
            flex-direction: column;
            width: 100%;
        }
        img {
            width: 90%;
        }
    }

    @media (max-width: 480px) {
        .welcome-title {
            font-size: 1.5em;
        }
        .welcome-subtitle {
            font-size: 1.2em;
        }
        .welcome-text {
            font-size: 0.9em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar functionality
if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = False

def toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open

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

# Section for Close Price Image and Description
st.markdown("""
<div class="image-container">
    <img src="https://img.freepik.com/premium-photo/pictures-coins-money-financial-pictures_1022029-109258.jpg?w=740" width="600" alt="Close Price Chart"/>
</div>
<div class="welcome-container">
    <h3 class="welcome-subtitle">Close Price Chart</h3>
    <p class="welcome-text">
        The Close Price Chart provides a visual representation of stock prices at market close over a specified period. 
        Analyze trends and patterns to make informed trading decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Section for Price Direction Prediction Image and Description
st.markdown("""
<div class="image-container">
    <img src="https://img.freepik.com/premium-vector/red-green-candles-stick-price-action-stock-chart-forex-candles-pattern-vector-currencies_293525-2210.jpg?w=740" width="600" alt="Price direction Chart"/>
</div>
<div class="welcome-container">
    <h3 class="welcome-subtitle">Price Direction Prediction</h3>
    <p class="welcome-text">
        Our Price Direction Prediction model uses advanced algorithms to forecast whether prices will rise or fall in the coming days.
        Stay ahead of the market with accurate predictions tailored to your trading strategy.
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions
st.write("""
### How to Use This Application:
1. **Input Parameters**: Navigate to the sidebar to enter the required parameters for your predictions.
2. **Make Predictions**: Click on the 'Predict Price Direction' button to generate predictions.
3. **Visualize Data**: Explore historical data visualizations and insights based on your inputs.
4. **Feature Engineering**: Gain deeper insights into your data with our feature engineering tools.
""")

# Call to action with flashing notification
st.markdown("""
<div class="notification">
     Click the arrow ▶ on the top-left to explore 🔍 
</div>
""", unsafe_allow_html=True)
