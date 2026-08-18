# 📈 AlphaTrade – Algorithmic Stock Prediction Engine

AlphaTrade is a Machine Learning based stock prediction and analysis system that uses historical market data, technical indicators, feature engineering, and ML models to generate stock price predictions and BUY/SELL/HOLD signals.

## 🚀 Features

- 📊 Historical stock data using Yahoo Finance
- 📈 Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- 🧠 Feature engineering and lag features
- 🤖 Machine Learning based price prediction
- 🔮 Next-day stock price prediction
- 📉 Backtesting and strategy performance analysis
- 📊 Interactive Streamlit dashboard
- 💹 Supports multiple stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
- 📋 Model performance and prediction reports

## 🛠️ Tech Stack

**Python • Pandas • NumPy • Scikit-learn • XGBoost • yFinance • TA • Matplotlib • Plotly • Streamlit • Joblib**

## 📁 Project Structure

```text
AlphaTrade/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── AAPL_model.pkl
│   ├── MSFT_model.pkl
│   └── ...
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_prediction.ipynb
│   └── 05_backtesting.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── indicators.py
│   ├── features.py
│   ├── models.py
│   ├── prediction.py
│   └── backtesting.py
│
├── requirements.txt
├── .gitignore
└── README.md
