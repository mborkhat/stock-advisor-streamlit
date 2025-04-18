import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process
import requests

# NewsAPI Key (replace with your actual API key)
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

# Ensure device compatibility (CPU if CUDA is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

RISK_THRESHOLDS = {"low": 5, "medium": 10, "high": 20}

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

def analyze_portfolio(symbols):
    return [fetch_stock_summary(sym) for sym in symbols if fetch_stock_summary(sym) is not None]

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
    matched_tickers = process.extract(query, tickers, limit=5)
    return [match[0] for match in matched_tickers if match[1] > 50]

# ✅ Fetch news using NewsAPI
def fetch_stock_news(symbol):
    company = symbol.replace('.NS', '')  # cleaner keyword for NewsAPI
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = []
        for item in data.get('articles', [])[:5]:
            articles.append({
                'title': item['title'],
                'source': item['source']['name'],
                'url': item['url'],
                'date': pd.to_datetime(item['publishedAt']).strftime('%Y-%m-%d %H:%M')
            })
        return articles
    else:
        return []

# 🌟 Streamlit App
st.title("📊 Indian Stock Portfolio Advisor (AI Powered)")

st.markdown("This app analyzes **Indian stocks** using Yahoo Finance and provides investment advice using Hugging Face 🤗.")

user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None

if user_search:
    suggestions = get_yahoo_stock_symbols(user_search)
    if suggestions:
        selected_symbol = st.selectbox("Suggestions:", suggestions)

if selected_symbol:
    result = fetch_stock_summary(selected_symbol)

    if not result:
        st.error("No data found. Try another stock.")
    else:
        st.subheader("📈 Stock Summary")
        st.write(f"**{result['symbol']}**: Current price ₹{result['current_price']:.2f}")
        st.write(f"**52 Week High**: ₹{result['week_52_high']}, **Low**: ₹{result['week_52_low']}")
        st.write(f"6-Month Change: {result['pct_change']:.2f}%, Risk: **{result['risk']}**")

        prompt = (
            f"The stock {result['symbol']} has changed {result['pct_change']:.2f}% over 6 months. "
            f"The current price is ₹{result['current_price']:.2f}. Risk level is {result['risk']}. Should I invest?"
        )
        recommendation = get_advice(prompt)
        st.write(f"**AI Recommendation**: 🧠 {recommendation}")

        st.subheader("📰 Latest News")
        articles = fetch_stock_news(result['symbol'])
        if articles:
            for article in articles:
                st.markdown(f"**[{article['title']}]({article['url']})**")
                st.write(f"{article['source']} | {article['date']}")
        else:
            st.info("No news found using NewsAPI.")

        st.subheader("📉 6-Month Price Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        result['history']['Close'].plot(ax=ax, title=f"{result['symbol']} - 6M Closing Prices")
        ax.set_ylabel('Price (₹)')
        ax.set_xlabel('Date')
        ax.grid(True)
        st.pyplot(fig)
