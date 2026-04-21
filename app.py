# app.py

import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

# Import our custom modules
from config import NIFTY_50, FEATURES
from data_utils import load_and_preprocess_data
from ml_pipeline import train_and_predict

# UI CONFIGURATION
st.set_page_config(page_title="NIFTY 50 Deep Learning Predictor", layout="wide")
st.title("📈 NIFTY 50 Quant Predictor")
st.markdown("Trains an LSTM with **Early Stopping** and evaluates **Directional Accuracy** on a validation set.")

# SIDEBAR CONTROLS
st.sidebar.header("Model Parameters")
selected_company = st.sidebar.selectbox("Select Asset", list(NIFTY_50.keys()))
ticker = NIFTY_50[selected_company]

years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
start_date = date.today().replace(year=date.today().year - years)
end_date = date.today()

# Increased max epochs so Early Stopping has room to trigger
epochs = st.sidebar.slider("Max Training Epochs", 20, 200, 100)
lookback_window = st.sidebar.slider("Lookback Window (Days)", 30, 90, 60)
patience = st.sidebar.slider("Early Stopping Patience", 5, 30, 15)

# DATA FETCHING
df = load_and_preprocess_data(ticker, start_date, end_date)

if df is None:
    st.error("Failed to fetch data. Please try another stock.")
else:
    # EDA PLOT
    st.subheader(f"Historical Closing Price & Moving Averages: {selected_company}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Close Price', color='#1f77b4')
    ax.plot(df.index, df['SMA_20'], label='20-Day SMA', color='#ff7f0e', alpha=0.7)
    ax.plot(df.index, df['SMA_50'], label='50-Day SMA', color='#2ca02c', alpha=0.7)
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # ML PIPELINE TRIGGER
    if st.button("🚀 Train LSTM & Predict Tomorrow's Price"):
        
        st.write("### Training Neural Network...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Call our clean, extracted ML function
        current_price, predicted_price, best_epoch, dir_acc = train_and_predict(
            df=df, 
            features=FEATURES, 
            lookback_window=lookback_window, 
            epochs=epochs,
            patience=patience,
            st_progress_bar=progress_bar, 
            st_status_text=status_text
        )

        st.success(f"Training finalized! Model restored to best weights from Epoch {best_epoch}.")

        # RESULTS CALCULATION & DISPLAY
        price_diff = predicted_price - current_price
        pct_change = (price_diff / current_price) * 100
        direction = "UP 🔼" if price_diff > 0 else "DOWN 🔽"
        color = "normal" if price_diff > 0 else "inverse"

        st.markdown("---")
        st.subheader("🔮 Tomorrow's Prediction")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Actual Close", f"₹{current_price:.2f}")
        col2.metric("LSTM Predicted Close", f"₹{predicted_price:.2f}", delta=f"₹{price_diff:.2f}", delta_color=color)
        col3.metric("Expected Move", f"{pct_change:.2f}%", delta=direction, delta_color=color)
        
        st.markdown("---")
        st.subheader("📊 Model Validation Metrics")
        col4, col5 = st.columns(2)
        
        col4.metric("Directional Accuracy (Validation Set)", f"{dir_acc:.1f}%")
        col5.metric("Peak Predictive Epoch", f"{best_epoch} / {epochs}")
            
        st.info("**Disclaimer:** Deep Learning models trained on limited epochs in a web app are for educational purposes only. Do not use this for actual financial trading.")