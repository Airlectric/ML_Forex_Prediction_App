import streamlit as st

# Custom CSS for styling and responsiveness
st.markdown("""
<style>
    /* Body and Main Styling */
    body {
        background-color: #f0f4f8;  /* Light background for a clean, trading-inspired look */
        color: #1c1e21;  /* Darker text for readability */
    }
    .welcome-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1); /* Sharp shadow for professional feel */
        text-align: center;
        margin-top: 30px;
        width: 100%;
    }
    .welcome-title {
        font-size: 3em;
        color: #004080;  /* A rich navy blue for professionalism */
        margin: 0;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .welcome-subtitle {
        font-size: 2em;
        color: #ffa500; /* Gold to highlight importance */
        margin: 15px 0;
    }
    .welcome-text {
        font-size: 1.2em;
        color: #333333;
        margin: 15px 0;
        line-height: 1.6em;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 25px;
        width: 100%;
    }
    img {
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); /* Softer shadow for images */
        transition: transform 0.3s ease-out; /* Smooth scaling effect */
        max-width: 90%;
        height: auto;
    }
    img:hover {
        transform: scale(1.07);  /* Slight hover scale for interaction */
    }
    .notification {
        text-align: center;
        font-size: 1.2em;
        color: #004080;
        animation: flash 1.5s infinite; /* Flashing effect for notification */
    }
    @keyframes flash {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2.5em;
        }
        .welcome-subtitle {
            font-size: 1.5em;
        }
        .welcome-text {
            font-size: 1.1em;
        }
        .image-container {
            flex-direction: column;
            width: 100%;
        }
        img {
            width: 100%;
        }
    }
    @media (max-width: 480px) {
        .welcome-title {
            font-size: 2em;
        }
        .welcome-subtitle {
            font-size: 1.2em;
        }
        .welcome-text {
            font-size: 1em;
        }
    }

    /* Button Styling */
    .stButton > button {
        background-color: #004080;  /* Button matches title color */
        color: white;
        font-size: 16px;
        padding: 12px 30px;
        border-radius: 10px;
        border: none;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease 0s;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #ffa500; /* Button changes to gold on hover */
        color: #ffffff;
        box-shadow: 0px 15px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar functionality
if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = False

def toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open

# Main content
st.title("Welcome to the Forex Price Prediction Application")
st.markdown("""
<div class="welcome-container">
    <h1 class="welcome-title">Unlock the Future of Trading</h1>
    <h2 class="welcome-subtitle">Predict Forex Prices with Confidence</h2>
    <p class="welcome-text">
       This application uses a variety of advanced machine learning models tailored to 
       support different forecasting needs. On the General Page, multivariate models like
       VECM and univariate models like SARIMAX are utilized to provide comprehensive predictions.
       For Close Price Prediction, powerful regression models offer precise forecasts. Additionally,
       on the Classification Page, state-of-the-art classification algorithms such as XGBoost and LightGBM
       are applied for prediction tasks. Explore historical trends, input your data, and receive real-time 
       predictions with tailored visualizations.
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

# Instructions for traders
st.write("""
### How to Use This Application:
1. **Input Parameters**: Navigate to the sidebar to enter the required parameters for your predictions.
2. **Make Predictions**: Click on the 'Predict Price Direction'or any other button to generate predictions.
3. **Visualize Data**: Explore historical and predicteed data visualizations and insights based on your inputs.
""")

# Call to action with flashing notification
st.markdown("""
<div class="notification">
     Click the arrow ▶ on the top-left to explore 🔍 
</div>
""", unsafe_allow_html=True)
