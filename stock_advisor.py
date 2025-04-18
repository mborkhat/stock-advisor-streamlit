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

# Use a CSV-based NSE symbol lookup to avoid API issues
@st.cache_data
def load_nse_symbols():
    url = "https://raw.githubusercontent.com/justindujardin/nse-india/main/nse_stocks.csv"
    df = pd.read_csv(url)
    df = df[df['series'] == 'EQ']  # Only EQ stocks
    df['label'] = df['nameOfCompany'] + " (" + df['symbol'] + ")"
    return dict(zip(df['label'], df['symbol']))

# Function to search and match stock names dynamically
def search_stock(query, symbols_dict):
    results = process.extract(query, symbols_dict.keys(), limit=10)
    return [res[0] for res in results]

# Fetch stock data
def fetch_stock_summary(symbol):
    stock = yf.Ticker(symbol + ".NS")
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

# Classify recommendation using Hugging Face
def get_advice(text):
    labels = ["Buy", "Hold", "Avoid"]
    result = classifier(text, labels)
    return result['labels'][0]

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **NSE-listed Indian stocks**, evaluates 6-month performance, and gives investment advice using Hugging Face transformers (100% free tech).
""")

# Load symbols
symbols_dict = load_nse_symbols()

# Autocomplete stock search input
user_input = st.text_input("Enter a stock name:")
if user_input:
    # Search for similar stock names using fuzzy matching
    matched_stocks = search_stock(user_input, symbols_dict)
    
    if matched_stocks:
        selected_stock = st.selectbox("Select a stock:", options=matched_stocks)
        selected_symbol = symbols_dict[selected_stock]
        
        if st.button("Analyze"):
            results = [fetch_stock_summary(selected_symbol)]
            
            if not results:
                st.error("No data found. Please try another stock.")
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
    else:
        st.warning("No matching stocks found. Please try a more specific search.")
else:
    st.write("Please enter a stock name to get started.")
