import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch

# Ensure the model runs on CPU if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Hugging Face zero-shot classifier with device setting
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Define risk metrics and thresholds
RISK_THRESHOLDS = {
    "low": 5,
    "medium": 10,
    "high": 20
}

# Function to search NSE symbols via live API (using NSE India unofficial endpoint)
@st.cache_data
def search_nse_symbols_live(query):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        url = f"https://www.nseindia.com/api/search/autocomplete?q={query}"
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Will raise an error if the request fails
        data = response.json()

        # Check if data is returned
        if 'symbols' not in data:
            st.error("Error fetching data from NSE API. Please try again later.")
            return []

        # Extract stock symbols and map them to their names
        matches = [
            {"label": item['label'], "symbol": item['symbol']}
            for item in data['symbols'] if item['symbol'].endswith("EQ")
        ]
        
        if not matches:
            st.warning(f"No stocks found for '{query}'. Try another search term.")
        
        return matches

    except Exception as e:
        st.error(f"An error occurred while fetching data: {str(e)}")
        return []

# Fetch stock data
def fetch_stock_summary(symbol):
    stock = yf.Ticker(symbol + ".NS")
    hist = stock.history(period="6mo")

    if hist.empty:
        return None

    current_price = hist['Close'].iloc[-1]
    change = current_price - hist['Close'].iloc[0]
    pct_change = (change / hist['Close'].iloc[0]) * 100
    risk = "low" if abs(pct_change) < RISK_THRESHOLDS["low"] else (
           "medium" if abs(pct_change) < RISK_THRESHOLDS["medium"] else "high")

    return {
        "symbol": symbol,
        "current_price": current_price,
        "pct_change": pct_change,
        "risk": risk,
        "history": hist
    }

# Analyze multiple stocks
def analyze_portfolio(symbols):
    return [fetch_stock_summary(sym) for sym in symbols if fetch_stock_summary(sym) is not None]

# Classify recommendation using Hugging Face
def get_advice(text):
    labels = ["Buy", "Hold", "Avoid"]
    result = classifier(text, labels)
    return result['labels'][0]

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **NSE-listed Indian stocks**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Search box for user input
user_input = st.text_input("Enter stock name or code:")

if user_input:
    # Fetch matching stocks dynamically as the user types
    matched = search_nse_symbols_live(user_input)

    if matched:
        selected_symbol = st.selectbox("Select a stock:", [item['label'] for item in matched])

        # Get the corresponding symbol for the selected stock
        selected_symbol_code = next(item['symbol'] for item in matched if item['label'] == selected_symbol)

        if st.button("Analyze"):
            results = analyze_portfolio([selected_symbol_code])

            if not results:
                st.error("No data found. Please try another stock.")
            else:
                df = pd.DataFrame(results)

                st.subheader("📈 Summary Table")
                st.dataframe(df[["symbol", "current_price", "pct_change", "risk"]])

                st.subheader("🧠 AI-Powered Recommendation")
                for r in results:
                    prompt = (f"The stock {r['symbol']} has changed {r['pct_change']:.2f}% over 6 months. "
                              f"The current price is ₹{r['current_price']:.2f}. Risk level is {r['risk']}. Should I invest?")
                    recommendation = get_advice(prompt)
                    st.write(f"**{r['symbol']}**: {recommendation} — *{prompt}*")

                st.subheader("📉 6-Month Price Chart")
                for r in results:
                    st.write(f"### {r['symbol']}")
                    fig, ax = plt.subplots()
                    r['history']['Close'].plot(ax=ax, title=f"{r['symbol']} - 6M Closing Prices")
                    st.pyplot(fig)

    else:
        st.warning("No stocks found for the entered query. Try another name or code.")
else:
    st.info("Please enter a stock name or code to get started.")
