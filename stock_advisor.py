import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process

# Ensure device compatibility
try:
    device = 0 if torch.cuda.is_available() else -1
except:
    device = -1

# Load Hugging Face zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Define risk metrics and thresholds
RISK_THRESHOLDS = {
    "low": 5,
    "medium": 10,
    "high": 20
}

# Fetch stock data from Yahoo Finance
def fetch_stock_summary(symbol):
    stock = yf.Ticker(symbol)
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

# Function to get stock symbols dynamically (example: can be extended)
def get_yahoo_stock_symbols(query):
    # Placeholder: return a list of symbols for the demo (in real-world, this can be fetched from a larger symbol dataset or API)
    all_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS", "SBIN.NS", "BAJAJ-AUTO.NS", "BHARTIARTL.NS"]
    return [symbol for symbol in all_symbols if query.lower() in symbol.lower()]

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Search for symbols dynamically from Yahoo (limited example list)
user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None  # Initialize selected_symbol

if user_search:
    # Get stock symbols matching the search query
    suggestions = get_yahoo_stock_symbols(user_search)
    if suggestions:
        # Fuzzy matching for better UX
        matches = process.extract(user_search, suggestions, limit=5)
        match_list = [match[0] for match in matches if match[1] > 50]  # Filter based on match score

        if match_list:
            selected_symbol = st.selectbox("Suggestions:", match_list)

# Allow the user to proceed with the analysis if a symbol is selected
if selected_symbol:
    results = analyze_portfolio([selected_symbol])

    if not results:
        st.error("No data found. Please try another stock symbol.")
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
