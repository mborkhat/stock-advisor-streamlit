import streamlit as st
import yfinance as yf
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fuzzywuzzy import process
import requests
from bs4 import BeautifulSoup

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

# Scrape stock symbols from 5Paisa
def fetch_5paisa_stock_symbols():
    url = 'https://www.5paisa.com/stocks/all'
    response = requests.get(url)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    stock_elements = soup.find_all('a', {'class': 'stock-name'})

    stock_symbols = []

    for element in stock_elements:
        symbol = element.get_text().strip()
        if symbol:  # Ensure it's not an empty string
            stock_symbols.append(symbol)

    return stock_symbols

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Fetch stock symbols dynamically from 5Paisa
stock_symbols = fetch_5paisa_stock_symbols()

user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.BO, TCS.BO)")
selected_symbol = None

if user_search:
    # Get stock symbols matching the search query
    suggestions = process.extract(user_search, stock_symbols, limit=5)
    if suggestions:
        # Display suggestions to the user
        selected_symbol = st.selectbox("Suggestions:", [match[0] for match in suggestions])

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
