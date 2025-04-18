import yfinance as yf
import matplotlib.pyplot as plt
import requests
import streamlit as st
from datetime import datetime

# Fetch stock data
def fetch_stock_summary(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    
    # Get stock data
    stock_info = stock.info
    current_price = stock_info.get('regularMarketPrice')
    week_52_high = stock_info.get('fiftyTwoWeekHigh')
    week_52_low = stock_info.get('fiftyTwoWeekLow')
    
    return current_price, week_52_high, week_52_low

# Fetch stock news using NewsAPI
def get_stock_news(stock_symbol):
    api_key = 'YOUR_NEWSAPI_KEY'  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={43519c8a11d042d39bf873d5d8cb0c6b}"
    response = requests.get(url)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        news_articles = []
        
        for article in articles[:5]:  # Top 5 recent articles
            news_articles.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"]
            })
        
        return news_articles
    else:
        return f"Error fetching news: {response.status_code}"

# Streamlit UI
st.title('Stock Advisor')

# User input for stock symbol
selected_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):")

if selected_symbol:
    # Fetch stock summary
    current_price, week_52_high, week_52_low = fetch_stock_summary(selected_symbol)

    # Display stock information
    st.write(f"### Current Price: ₹{current_price}")
    st.write(f"### 52 Week High: ₹{week_52_high}")
    st.write(f"### 52 Week Low: ₹{week_52_low}")

    # Plotting stock price chart
    stock_data = yf.download(selected_symbol, period="1y", interval="1d")
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label="Close Price")
    plt.title(f"{selected_symbol} - Stock Price Over the Last Year")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    
    # Show 52 week high and low on the plot
    plt.axhline(y=week_52_high, color='green', linestyle='--', label=f"52 Week High: ₹{week_52_high}")
    plt.axhline(y=week_52_low, color='red', linestyle='--', label=f"52 Week Low: ₹{week_52_low}")
    plt.legend(loc='best')
    st.pyplot(plt)

    # Get latest news
    news = get_stock_news(selected_symbol)

    if isinstance(news, str):
        st.error(news)  # Show error if news fetching fails
    else:
        st.write("### Latest News")
        for article in news:
            st.write(f"**{article['title']}**")
            st.write(f"Published At: {article['published_at']}")
            st.write(f"[Read more]({article['url']})")
            st.write(f"Description: {article['description']}")
            st.write("-" * 100)

else:
    st.write("Please enter a stock symbol.")
