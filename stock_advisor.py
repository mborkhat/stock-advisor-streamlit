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

# Fetch broader stock list dynamically from Yahoo (NSE)
@st.cache_data
def get_indian_stock_symbols():
    try:
        nse_500 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500")[1]
        nse_500["Symbol"] = nse_500["Symbol"].astype(str).str.strip() + ".NS"
        symbol_dict = dict(zip(nse_500['Company Name'], nse_500['Symbol']))

        # Also include reverse mapping from symbol to itself for symbol search
        symbol_dict.update({v: v for v in symbol_dict.values()})

        return symbol_dict
    except:
        return {}

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Load symbol dictionary
symbol_dict = get_indian_stock_symbols()
search_pool = list(symbol_dict.keys())

# Freeform text box with fuzzy dropdown-style feedback
user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")
selected_symbol = None

if user_search:
    # Fuzzy matching for suggestions
    suggestions = process.extract(user_search, search_pool, limit=5)
    matches = [f"{match} ({symbol_dict[match]})" for match, score in suggestions if score > 50]

    if matches:
        chosen = st.selectbox("Suggestions:", matches)
        top_match = chosen.split("(")[0].strip()
        selected_symbol = symbol_dict.get(top_match)

# Analyze the selected stock when button is clicked
if selected_symbol and st.button("Analyze"):
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
