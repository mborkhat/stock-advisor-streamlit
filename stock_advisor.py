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

# Fuzzy matching support for Yahoo tickers
def fuzzy_match_stock(user_input, stock_list):
    matches = process.extract(user_input, stock_list, limit=5)
    return [match[0] for match in matches]

# Define a static list of popular Yahoo Finance tickers with display names
yahoo_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS"
}

# Streamlit UI
st.title("\U0001F4C8 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Autocomplete with fuzzy search
user_search = st.text_input("Type stock name (e.g., reliance, infosys, tata...)")
selected_symbol = None

if user_search:
    suggestions = fuzzy_match_stock(user_search, list(yahoo_stocks.keys()))
    matched = st.selectbox("Select a matching stock:", suggestions)
    if matched:
        selected_symbol = yahoo_stocks[matched]

if selected_symbol and st.button("Analyze"):
    results = analyze_portfolio([selected_symbol])

    if not results:
        st.error("No data found. Please try another stock symbol.")
    else:
        df = pd.DataFrame(results)

        st.subheader("\U0001F4C8 Summary Table")
        st.dataframe(df[["symbol", "current_price", "pct_change", "risk"]])

        st.subheader("\U0001F9E0 AI-Powered Recommendation")
        for r in results:
            prompt = (f"The stock {r['symbol']} has changed {r['pct_change']:.2f}% over 6 months. "
                      f"The current price is ₹{r['current_price']:.2f}. Risk level is {r['risk']}. Should I invest?")
            recommendation = get_advice(prompt)
            st.write(f"**{r['symbol']}**: {recommendation} — *{prompt}*")

        st.subheader("\U0001F4C9 6-Month Price Chart")
        for r in results:
            st.write(f"### {r['symbol']}")
            fig, ax = plt.subplots()
            r['history']['Close'].plot(ax=ax, title=f"{r['symbol']} - 6M Closing Prices")
            st.pyplot(fig)
