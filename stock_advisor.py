import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
import torch
from fuzzywuzzy import process
import requests
import re

# NEWS API KEY (replace with your key)
NEWS_API_KEY = "43519c8a11d042d39bf873d5d8cb0c6b"

# Ensure device compatibility (CPU only to avoid meta tensor errors)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # Force CPU
)

RISK_THRESHOLDS = {
    "low": 5,
    "medium": 10,
    "high": 20
}

def fetch_stock_summary(symbol, period):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)

    if hist.empty:
        return None

    current_price = hist['Close'].iloc[-1]
    change = current_price - hist['Close'].iloc[0]
    pct_change = (change / hist['Close'].iloc[0]) * 100
    risk = "low" if abs(pct_change) < RISK_THRESHOLDS["low"] else (
           "medium" if abs(pct_change) < RISK_THRESHOLDS["medium"] else "high")

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

def get_advice(text):
    labels = ["Buy", "Hold", "Avoid"]
    result = classifier(text, labels)
    return result['labels'][0]

def get_nifty_50_symbols():
    return [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS", "SBIN.NS",
        "BAJAJ-AUTO.NS", "BHARTIARTL.NS", "M&M.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS",
        "HUL.NS", "AXISBANK.NS", "MARUTI.NS", "ULTRACEMCO.NS", "WIPRO.NS", "SUNPHARMA.NS",
        "HCLTECH.NS", "ONGC.NS", "BAJAJFINSV.NS", "TITAN.NS", "NTPC.NS", "ADANIGREEN.NS",
        "POWERGRID.NS", "ASIANPAINT.NS", "JSWSTEEL.NS", "DRREDDY.NS", "INDUSINDBK.NS",
        "BHEL.NS", "VEDL.NS", "SHREECEM.NS", "HINDALCO.NS", "M&MFIN.NS", "EICHERMOT.NS",
        "GAIL.NS", "COALINDIA.NS", "BOSCHLTD.NS", "HDFCBANK.NS", "CIPLA.NS", "MARICO.NS",
        "UPL.NS", "RECLTD.NS", "TECHM.NS", "DIVISLAB.NS", "PIDILITIND.NS", "MOTHERSUMI.NS",
        "TATAMOTORS.NS"
    ]

def get_yahoo_stock_symbols(query):
    tickers = get_nifty_50_symbols()
    matched = process.extract(query, tickers, limit=5)
    return [m[0] for m in matched if m[1] > 50]

def fetch_stock_news(symbol):
    stock = yf.Ticker(symbol)
    company_name = stock.info.get("longName", re.sub(r'[\W_]+', ' ', symbol.replace('.NS', '')).strip())
    query = f'"{company_name}" AND stock'
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [{
            'title': a['title'],
            'source': a['source']['name'],
            'url': a['url'],
            'date': pd.to_datetime(a['publishedAt']).strftime('%Y-%m-%d %H:%M')
        } for a in data.get('articles', [])[:3]]
    return []

# Streamlit UI
st.title("\U0001F4C8 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

user_search = st.text_input("\U0001F50D Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None

time_range = st.selectbox("Select performance period:", (
    "7d", "6mo", "1y", "2y", "3y", "4y", "5y"), index=0)

if user_search:
    suggestions = get_yahoo_stock_symbols(user_search)
    if suggestions:
        selected_symbol = st.selectbox("Suggestions:", suggestions)

if selected_symbol:
    result = fetch_stock_summary(selected_symbol, time_range)

    if not result:
        st.error("No data found. Please try another stock symbol.")
    else:
        # Organize stock summary in a column layout for better mobile view
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("\U0001F4C8 Stock Summary")
            st.write(f"**{result['symbol']}**: Current price ₹{result['current_price']:.2f}")
            st.write(f"**52 Week High**: ₹{result['week_52_high']}, **52 Week Low**: ₹{result['week_52_low']}")
            st.write(f"Performance over {time_range}: {result['pct_change']:.2f}%")
            st.write(f"Risk level: {result['risk']}")

        with col2:
            prompt = (
                f"The stock {result['symbol']} has changed {result['pct_change']:.2f}% over {time_range}. "
                f"The current price is ₹{result['current_price']:.2f}. Risk level is {result['risk']}. Should I invest?"
            )
            recommendation = get_advice(prompt)
            st.write(f"**Recommendation**: {recommendation}")

        # Expandable section for news
        with st.expander("🔍 Latest News"):
            articles = fetch_stock_news(result['symbol'])
            if articles:
                for article in articles:
                    st.write(f"- **{article['title']}**")
                    st.write(f"  Source: {article['source']} | Date: {article['date']}")
                    st.write(f"  [Read more]({article['url']})")
            else:
                st.write("No news found for this stock.")

        # Display price chart with 7-day moving average if selected
        st.subheader(f"\U0001F4C9 {time_range} Price Chart")
        hist = result['history']
        
        if time_range == '7d':
            hist = hist.tail(7)  # Get only the last 7 days
        
        # Calculate the 7-day Moving Average
        hist['7_day_MA'] = hist['Close'].rolling(window=7).mean()
        
        # Plot the price chart and the moving average
        fig = go.Figure()

        # Plot stock price
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines+markers', name='Close Price', text=hist['Close'], hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'))
        
        # Plot 7-day Moving Average
        fig.add_trace(go.Scatter(x=hist.index, y=hist['7_day_MA'], mode='lines', name='7-Day Moving Average', line=dict(dash='dash')))

        fig.update_layout(
            title=f"{result['symbol']} - {time_range} Closing Prices",
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
