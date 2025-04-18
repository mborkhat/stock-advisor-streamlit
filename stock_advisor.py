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

# Function to search stock symbols on Yahoo Finance
def search_yahoo_finance(query):
    # Search Yahoo Finance for the query (e.g., "Reliance", "INFY", etc.)
    try:
        search_result = yf.Tickers(query)
        return search_result.tickers
    except Exception as e:
        return None

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Search for symbols dynamically using Yahoo Finance
user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None  # Initialize selected_symbol

if user_search:
    # Search for stock data on Yahoo Finance
    tickers = search_yahoo_finance(user_search)
    if tickers:
        matches = [ticker for ticker in tickers if user_search.lower() in ticker.info['longName'].lower()]
        if matches:
            st.write("Suggestions: ", [match.info['symbol'] for match in matches])
            selected_symbol = matches[0].info['symbol']  # Automatically choose the first match

# Allow the user to proceed with the analysis
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
