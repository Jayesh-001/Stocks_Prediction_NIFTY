import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def load_and_preprocess_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        return None
    
    df = df.dropna()
    
    # Calculate Technical Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df = df.dropna()
    return df