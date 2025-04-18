import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# Fetch stock data from Yahoo Finance
def fetch_stock_summary(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")
    
    # Fetch 52-week high and low
    try:
        week_52_high = stock.info['fiftyTwoWeekHigh']
        week_52_low = stock.info['fiftyTwoWeekLow']
    except KeyError:
        week_52_high, week_52_low = None, None

    if hist.empty:
        return None

    current_price = hist['Close'].iloc[-1]
    change = current_price - hist['Close'].iloc[0]
    pct_change = (change / hist['Close'].iloc[0]) * 100
    risk = "low" if abs(pct_change) < 5 else ("medium" if abs(pct_change) < 10 else "high")

    # Get the latest news articles related to the stock
    news = stock.news[:5]  # Get the top 5 latest news articles

    return {
        "symbol": symbol,
        "current_price": current_price,
        "pct_change": pct_change,
        "risk": risk,
        "history": hist,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "news": news
    }

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
    matched_tickers = process.extract(query, tickers, limit=5)
    return [match[0] for match in matched_tickers if match[1] > 50]

# Streamlit UI
st.title("📊 Indian Stock Portfolio Advisor (Free AI Powered)")

st.markdown("""
This app analyzes **Indian stocks from Yahoo Finance**, evaluates 6-month performance, and provides investment insights like 52-week high/low and the latest news.
""")

user_search = st.text_input("🔍 Type stock name or symbol (e.g., Reliance, INFY.NS, TCS.NS)")

if user_search:
    suggestions = get_yahoo_stock_symbols(user_search)
    
    if suggestions:
        selected_symbol = st.selectbox("Select stock symbol", suggestions)
    else:
        st.error("Stock not found. Please try again with a different symbol.")

    if selected_symbol:
        st.write(f"Analyzing **{selected_symbol}**...")

        result = fetch_stock_summary(selected_symbol)
        if result:
            df = pd.DataFrame([result])
            st.subheader("📈 Stock Summary Table")
            st.dataframe(df[["symbol", "current_price", "pct_change", "risk"]])

            # Display 52-week high and low
            st.subheader("📊 52-Week High and Low")
            st.write(f"**52-Week High**: ₹{result['week_52_high']:.2f}")
            st.write(f"**52-Week Low**: ₹{result['week_52_low']:.2f}")

            st.subheader("🧠 AI-Powered Trend Suggestion")
            trend = "Uptrend" if result["pct_change"] > 0 else "Downtrend" if result["pct_change"] < 0 else "Neutral"
            st.write(f"Trend Suggestion: {trend} based on a {result['pct_change']:.2f}% change over the last 6 months.")

            st.subheader("📰 Latest News Articles")
            for news_item in result["news"]:
                st.write(f"- **{news_item['title']}**: {news_item['link']}")

            st.subheader("📉 6-Month Price Chart")
            
            # Check if the stock data contains valid 'Close' prices before plotting
            if not result['history'].empty and 'Close' in result['history']:
                fig, ax = plt.subplots(figsize=(10, 6))
                result['history']['Close'].plot(ax=ax, title=f"{selected_symbol} - 6M Closing Prices", grid=True)
                
                # Add 52-week high and low as horizontal lines
                if result['week_52_high'] and result['week_52_low']:
                    ax.axhline(result['week_52_high'], color='green', linestyle='--', label=f"52-Week High: ₹{result['week_52_high']}")
                    ax.axhline(result['week_52_low'], color='red', linestyle='--', label=f"52-Week Low: ₹{result['week_52_low']}")
                
                # Customize the plot with labels and legends
                ax.set_ylabel("Price (₹)")
                ax.set_xlabel("Date")
                ax.legend(loc="best")
                st.pyplot(fig)
            else:
                st.error("No valid stock data available to plot.")
        else:
            st.error("No data found for the selected stock.")
