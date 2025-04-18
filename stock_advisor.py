import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process
from time import sleep
from yfinance.exceptions import YFRateLimitError
from cachetools import TTLCache

# Cache for storing stock data for 10 minutes to avoid repeated API calls
stock_cache = TTLCache(maxsize=100, ttl=600)

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
    # Check if data is already cached
    if symbol in stock_cache:
        return stock_cache[symbol]

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            return None
        
        # Get 52-week high and low
        week_52_high = stock.info.get('fiftyTwoWeekHigh', None)
        week_52_low = stock.info.get('fiftyTwoWeekLow', None)

        # Calculate risk based on percentage change
        current_price = hist['Close'].iloc[-1]
        change = current_price - hist['Close'].iloc[0]
        pct_change = (change / hist['Close'].iloc[0]) * 100
        risk = "low" if abs(pct_change) < RISK_THRESHOLDS["low"] else (
               "medium" if abs(pct_change) < RISK_THRESHOLDS["medium"] else "high")

        # Store the result in cache
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "pct_change": pct_change,
            "risk": risk,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "history": hist
        }
        stock_cache[symbol] = result
        return result
    
    except YFRateLimitError:
        st.error("Rate limit exceeded. Please try again later.")
        sleep(10)  # Retry after 10 seconds
        return None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

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
    # Get Nifty 50 symbols dynamically (can be changed for other indexes or lists)
    tickers = get_nifty_50_symbols()
    
    # Match query with tickers (use fuzzy matching here)
    matched_tickers = process.extract(query, tickers, limit=5)
    return [match[0] for match in matched_tickers if match[1] > 50]

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

    if result:
        st.subheader(f"📈 Stock Summary: {selected_symbol}")
        st.write(f"**Current Price**: ₹{result['current_price']:.2f}")
        st.write(f"**Percentage Change (6 months)**: {result['pct_change']:.2f}%")
        st.write(f"**52 Week High**: ₹{result['week_52_high']}")
        st.write(f"**52 Week Low**: ₹{result['week_52_low']}")

        # Show recommendation
        prompt = (f"The stock {result['symbol']} has changed {result['pct_change']:.2f}% over 6 months. "
                  f"The current price is ₹
