# 📈 NIFTY 50 Quant Predictor (PyTorch + Streamlit)

A full-stack Deep Learning web application that predicts short-term stock price movements for NIFTY 50 companies. Built with **PyTorch** and **Streamlit**, this tool dynamically trains a Long Short-Term Memory (LSTM) neural network on the fly, evaluates its directional accuracy, and forecasts the next day's closing price.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)

## ✨ Features

- **On-the-Fly Training:** Trains a custom PyTorch LSTM model in the browser based on user-defined hyperparameters (Epochs, Lookback Window).
- **Technical Feature Engineering:** Automatically calculates and feeds MACD, RSI, and Simple Moving Averages (20 & 50-day) into the neural network.
- **Smart Early Stopping:** Uses an 85/15 Train-Validation split to monitor validation loss, halting training to prevent overfitting and restoring the best model weights.
- **Quantitative Evaluation:** Evaluates the model based on **Directional Accuracy** (did it correctly predict an UP or DOWN movement?) rather than just standard regression metrics.
- **Interactive UI:** Sleek Streamlit dashboard for selecting assets, adjusting history lengths, and viewing historical price overlays.

## 🗂️ Project Structure

This project was transitioned from an exploratory Jupyter Notebook (`Stonks.ipynb`) into a production-ready modular architecture:

```text
├── app.py                # Main Streamlit application and UI frontend
├── config.py             # Configuration file mapping NIFTY 50 companies to tickers
├── data_utils.py         # Data acquisition (yfinance) and Technical Indicator math
├── ml_pipeline.py        # Core engine: sequence generation, training loop, and evaluation
├── models.py             # PyTorch LSTM neural network class definition
├── requirements.txt      # Python package dependencies
└── Stonks.ipynb          # Original exploratory data analysis and prototyping notebook
```

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone [https://github.com/yourusername/nifty-quant-predictor.git](https://github.com/yourusername/nifty-quant-predictor.git)
cd nifty-quant-predictor
```

**2. Create a virtual environment (Recommended)**
_Using Conda:_

```bash
conda create -n quant_env python=3.9
conda activate quant_env
```

_Using venv:_

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## 🧠 How the Model Works

1.  **Data Fetching:** Downloads daily historical data via `yfinance`.
2.  **Feature Scaling:** Uses `MinMaxScaler` to normalize features (Close, SMAs, MACD, RSI) between (0, 1) to ensure stable gradient descent. A dedicated scaler is kept for the 'Close' price to inverse-transform final predictions.
3.  **Sequence Generation:** Converts 2D tabular data into 3D overlapping time-series sequences `(Batch, Sequence Length, Features)`.
4.  **LSTM Forward Pass:** The recurrent layers process the sequence to capture time-based market momentum, passing the final hidden state through a Dropout layer and a Linear layer to output a continuous price prediction.
5.  **Directional Accuracy:** Validates the model by multiplying the actual historical price move by the predicted move. A positive product indicates the model correctly guessed the direction of the market.

## ⚠️ Disclaimer

This project is for **educational and portfolio demonstration purposes only**. Deep learning models trained on a limited set of historical data and epochs are highly experimental. Do not use this application or its predictions to make real financial investments or trades.

```

```
