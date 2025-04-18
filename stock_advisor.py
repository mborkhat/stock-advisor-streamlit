import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process
import requests
import re
import plotly.graph_objects as go

# NEWS API KEY (replace with your key)
NEWS_API_KEY = "your_newsapi_key_here"

# Ensure device compatibility (CPU if CUDA is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

RISK_THRESHOLDS = {
    "low": 5,
    "medium": 10,
    "high": 20
}

def fetch_stock_summary(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="5y")  # Changed to 5 years

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
    company = re.sub(r'\W+', ' ', symbol.replace('.NS', '')).strip()
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [{
            'title': a['title'],
            'source': a['source']['name'],
            'url': a['url'],
            'date': pd.to_datetime(a['publishedAt']).strftime('%Y-%m-%d %H:%M')
        } for a in data.get('articles', [])[:5]]
    return []

# Streamlit UI
st.title("\U0001F4C8 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 5-year performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

user_search = st.text_input("\U0001F50D Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None

if user_search:
    suggestions = get_yahoo_stock_symbols(user_search)
    if suggestions:
        selected_symbol = st.selectbox("Suggestions:", suggestions)

if selected_symbol:
    result = fetch_stock_summary(selected_symbol)

    if not result:
        st.error("No data found. Please try another stock symbol.")
    else:
        st.subheader("\U0001F4C8 Stock Summary")
        st.write(f"**{result['symbol']}**: Current price ₹{result['current_price']:.2f}")
        st.write(f"**52 Week High**: ₹{result['week_52_high']}, **52 Week Low**: ₹{result['week_52_low']}")
        st.write(f"Performance over 5 years: {result['pct_change']:.2f}%")
        st.write(f"Risk level: {result['risk']}")

        prompt = (
            f"The stock {result['symbol']} has changed {result['pct_change']:.2f}% over 5 years. "
            f"The current price is ₹{result['current_price']:.2f}. Risk level is {result['risk']}. Should I invest?"
        )
        recommendation = get_advice(prompt)
        st.write(f"**Recommendation**: {recommendation}")

        st.subheader("\U0001F4F0 Latest News")
        articles = fetch_stock_news(result['symbol'])
        if articles:
            for article in articles:
                st.write(f"- **{article['title']}**")
                st.write(f"  Source: {article['source']} | Date: {article['date']}")
                st.write(f"  [Read more]({article['url']})")
        else:
            st.write("No news found for this stock.")

        # Chart Section
        st.subheader("📉 5-Year Price Chart")
        if not result['history'].empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result['history'].index,
                y=result['history']['Close'],
                mode='lines',
                name='Closing Price',
                line=dict(color='royalblue'),
                hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
            ))
            fig.update_layout(
                title=f"{result['symbol']} - 5Y Closing Prices",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price history not available.")
