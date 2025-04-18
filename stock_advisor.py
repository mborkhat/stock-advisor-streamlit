import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process
import requests

# Ensure device compatibility (CPU if CUDA is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Fetch 52 week high and low
    week_52_high = stock.info.get('fiftyTwoWeekHigh', 'N/A')
    week_52_low = stock.info.get('fiftyTwoWeekLow', 'N/A')

    return {
        "symbol": symbol,
        "current_price": current_price,
        "pct_change": pct_change,
        "risk": risk,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
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

# Fetch Nifty 50 symbols dynamically
def get_nifty_50_symbols():
    nifty_50_symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS", "SBIN.NS", 
        "BAJAJ-AUTO.NS", "BHARTIARTL.NS", "M&M.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS",
        "HUL.NS", "AXISBANK.NS", "MARUTI.NS", "ULTRACEMCO.NS", "WIPRO.NS", "SUNPHARMA.NS",
        "HCLTECH.NS", "ONGC.NS", "BAJAJFINSV.NS", "TITAN.NS", "NTPC.NS", "ADANIGREEN.NS",
        "POWERGRID.NS", "ASIANPAINT.NS", "JSWSTEEL.NS", "DRREDDY.NS", "INDUSINDBK.NS", 
        "BHEL.NS", "VEDL.NS", "SHREECEM.NS", "HINDALCO.NS", "M&MFIN.NS", "EICHERMOT.NS", 
        "GAIL.NS", "COALINDIA.NS", "BOSCHLTD.NS", "HDFCBANK.NS", "CIPLA.NS", "MARICO.NS", 
        "UPL.NS", "RECLTD.NS", "TECHM.NS", "DIVISLAB.NS", "PIDILITIND.NS", "MOTHERSUMI.NS", 
        "TATAMOTORS.NS", "TCS.NS"
    ]
    return nifty_50_symbols

# Function to fetch stock symbols from user search
def get_yahoo_stock_symbols(query):
    tickers = get_nifty_50_symbols()
    
    # Match query with tickers (use fuzzy matching here)
    matched_tickers = process.extract(query, tickers, limit=5)
    return [match[0] for match in matched_tickers if match[1] > 50]

# Fetch latest news articles related to the stock
def fetch_stock_news(symbol):
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=news"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        news = data['quoteSummary']['result'][0].get('news', [])
        articles = []
        for item in news:
            articles.append(item['title'])
        return articles
    return []

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Search for symbols dynamically from Yahoo Finance
user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None  # Initialize selected_symbol

if user_search:
    # Get stock symbols matching the search query
    suggestions = get_yahoo_stock_symbols(user_search)
    if suggestions:
        # Display suggestions to the user
        selected_symbol = st.selectbox("Suggestions:", suggestions)

# Allow the user to proceed with the analysis if a symbol is selected
if selected_symbol:
    result = fetch_stock_summary(selected_symbol)

    if not result:
        st.error("No data found. Please try another stock symbol.")
    else:
        st.subheader("📈 Stock Summary")
        st.write(f"**{result['symbol']}**: Current price ₹{result['current_price']:.2f}")
        st.write(f"**52 Week High**: ₹{result['week_52_high']}, **52 Week Low**: ₹{result['week_52_low']}")
        st.write(f"Performance over 6 months: {result['pct_change']:.2f}%")
        st.write(f"Risk level: {result['risk']}")

        # Display AI recommendation
        prompt = (f"The stock {result['symbol']} has changed {result['pct_change']:.2f}% over 6 months. "
                  f"The current price is ₹{result['current_price']:.2f}. Risk level is {result['risk']}. Should I invest?")
        recommendation = get_advice(prompt)
        st.write(f"**Recommendation**: {recommendation}")

        # Fetch and display the latest news articles
        st.subheader("📰 Latest News")
        articles = fetch_stock_news(result['symbol'])
        if articles:
            for article in articles[:5]:  # Display top 5 news articles
                st.write(f"- {article}")
        else:
            st.write("No news found for this stock.")

        # Display stock price chart
        st.subheader("📉 6-Month Price Chart")
        fig, ax = plt.subplots()
        result['history']['Close'].plot(ax=ax, title=f"{result['symbol']} - 6M Closing Prices")
        st.pyplot(fig)
