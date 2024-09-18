import streamlit as st
from datetime import datetime  # Importing datetime

# Custom CSS for styling
st.markdown("""
<style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f4f9;  /* Light background for contrast */
        color: #333;  /* Dark text for readability */
    }
    .welcome-container {
        background-color: rgba(255, 255, 255, 0.9);  /* White background */
        padding: 30px;  /* Padding for spacing */
        border-radius: 10px;  /* Rounded corners */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        text-align: center;
        margin-top: 20px;
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
    }
    img {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s; /* Smooth scaling effect */
    }
    img:hover {
        transform: scale(1.05); /* Scale on hover */
    }
    .button {
        background-color: #6c63ff; /* Button color */
        color: white; /* Button text color */
        border-radius: 5px; /* Rounded corners */
        padding: 10px 20px; /* Padding inside button */
        border: none; /* No border */
        cursor: pointer; /* Pointer cursor on hover */
        transition: background-color 0.3s; /* Smooth transition */
    }
    .button:hover {
        background-color: #5753d6; /* Darker button on hover */
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

# Call to action with a styled button
if st.button("Open Sidebar"):
    toggle_sidebar()

if st.session_state.sidebar_open:
    st.sidebar.title("Information")
    
    # Add an icon (using emoji) and a brief informational message
    st.sidebar.markdown("🔍 **Get Insights**")
    st.sidebar.write("Explore various Forex pairs and gain insights into price predictions.")
        
# Additional call to action
st.markdown("<button class='button' onclick='toggle_sidebar()'>Ready to make your predictions? Toggle the sidebar to get started!</button>", unsafe_allow_html=True)