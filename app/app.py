import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT_DIR)
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AlphaTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 AlphaTrade")

st.subheader(
    "Algorithmic Stock Prediction Engine"
)

st.markdown(
    """
    **AlphaTrade** combines historical stock data,
    technical indicators and machine learning to
    estimate the next price movement.
    """
)


# ============================================================
# CONSTANTS
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA"
]

DATA_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

MODEL_DIR = (
    ROOT_DIR
    / "models"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ AlphaTrade")


ticker = st.sidebar.selectbox(
    "Select Stock",
    TICKERS
)


threshold = st.sidebar.slider(
    "Signal Threshold (%)",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
)


predict_button = st.sidebar.button(
    "🚀 Predict",
    use_container_width=True
)


st.sidebar.markdown("---")


st.sidebar.write(
    "Selected Stock:"
)

st.sidebar.success(
    ticker
)


# ============================================================
# FILE PATHS
# ============================================================

FEATURE_FILE = (
    DATA_DIR
    / f"{ticker}_features.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / f"{ticker}_model.pkl"
)

FEATURE_COLUMNS_FILE = (
    MODEL_DIR
    / f"{ticker}_feature_columns.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_stock_data(
    file_path
):

    df = pd.read_csv(
        file_path
    )

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CHECK FEATURE DATA
# ============================================================

if not FEATURE_FILE.exists():

    st.error(
        f"""
        ❌ Feature data not found for {ticker}.

        Expected file:

        {FEATURE_FILE}

        Run the feature engineering notebook
        for this stock first.
        """
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_stock_data(
        FEATURE_FILE
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# ============================================================
# LATEST DATA
# ============================================================

latest = df.iloc[-1]

current_price = float(
    latest["Close"]
)


# ============================================================
# HEADER INFORMATION
# ============================================================

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Current Price
# ------------------------------------------------------------

with col1:

    st.metric(
        "💰 Current Price",
        f"${current_price:.2f}"
    )


# ------------------------------------------------------------
# Daily Change
# ------------------------------------------------------------

with col2:

    if len(df) >= 2:

        previous_price = float(
            df["Close"].iloc[-2]
        )

        daily_change = (
            current_price
            - previous_price
        )

        daily_change_pct = (
            daily_change
            / previous_price
        ) * 100

    else:

        daily_change_pct = 0

    st.metric(
        "📊 Daily Change",
        f"{daily_change_pct:.2f}%"
    )


# ------------------------------------------------------------
# Volume
# ------------------------------------------------------------

with col3:

    if "Volume" in df.columns:

        volume = int(
            latest["Volume"]
        )

        st.metric(
            "📦 Volume",
            f"{volume:,}"
        )

    else:

        st.metric(
            "📦 Volume",
            "N/A"
        )


# ------------------------------------------------------------
# Date
# ------------------------------------------------------------

with col4:

    latest_date = latest["Date"]

    st.metric(
        "📅 Latest Date",
        str(
            latest_date.date()
        )
    )


# ============================================================
# PRICE CHART
# ============================================================

st.header(
    f"📈 {ticker} Price Chart"
)


fig = go.Figure()


# ------------------------------------------------------------
# Candlestick
# ------------------------------------------------------------

fig.add_trace(

    go.Candlestick(

        x=df["Date"],

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"],

        name=ticker
    )
)


# ------------------------------------------------------------
# SMA 20
# ------------------------------------------------------------

if "SMA_20" in df.columns:

    fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["SMA_20"],

            mode="lines",

            name="SMA 20"
        )
    )


# ------------------------------------------------------------
# SMA 50
# ------------------------------------------------------------

if "SMA_50" in df.columns:

    fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["SMA_50"],

            mode="lines",

            name="SMA 50"
        )
    )


fig.update_layout(

    height=550,

    xaxis_title="Date",

    yaxis_title="Price",

    xaxis_rangeslider_visible=False,

    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.header(
    "📊 Technical Indicators"
)


indicator_columns = [

    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "BB_High",
    "BB_Low",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20"
]


available_indicators = [

    col
    for col in indicator_columns
    if col in df.columns
]


if available_indicators:

    indicator_cols = st.columns(
        4
    )

    for i, column in enumerate(
        available_indicators
    ):

        value = latest[column]

        if pd.isna(value):

            display_value = "N/A"

        else:

            display_value = (
                f"{float(value):.2f}"
            )

        with indicator_cols[
            i % 4
        ]:

            st.metric(
                column,
                display_value
            )

else:

    st.warning(
        "No technical indicators available."
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("---")

st.header(
    "🤖 Machine Learning Prediction"
)


if predict_button:

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not MODEL_FILE.exists():

        st.error(
            f"""
            ❌ Trained model not found.

            Expected:

            {MODEL_FILE}

            Train the model first.
            """
        )

    elif not FEATURE_COLUMNS_FILE.exists():

        st.error(
            f"""
            ❌ Feature columns file not found.

            Expected:

            {FEATURE_COLUMNS_FILE}
            """
        )

    else:

        try:

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            model = joblib.load(
                MODEL_FILE
            )


            # ------------------------------------------------
            # LOAD FEATURES
            # ------------------------------------------------

            feature_columns = joblib.load(
                FEATURE_COLUMNS_FILE
            )


            # ------------------------------------------------
            # CHECK FEATURES
            # ------------------------------------------------

            missing_features = [

                col

                for col in feature_columns

                if col not in df.columns
            ]


            if missing_features:

                st.error(
                    "Missing features:"
                )

                st.write(
                    missing_features
                )

                st.stop()


            # ------------------------------------------------
            # LATEST INPUT
            # ------------------------------------------------

            X_latest = (

                latest[
                    feature_columns
                ]

                .to_frame()

                .T
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            predicted_price = float(

                model.predict(
                    X_latest
                )[0]
            )


            # ------------------------------------------------
            # EXPECTED RETURN
            # ------------------------------------------------

            expected_return = (

                (
                    predicted_price
                    - current_price
                )
                / current_price

            ) * 100


            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            if expected_return > threshold:

                signal = "BUY"

            elif expected_return < -threshold:

                signal = "SELL"

            else:

                signal = "HOLD"


            # =================================================
            # PREDICTION CARDS
            # =================================================

            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "Current Price",
                    f"${current_price:.2f}"
                )


            with col2:

                price_difference = (
                    predicted_price
                    - current_price
                )

                st.metric(
                    "Predicted Price",
                    f"${predicted_price:.2f}",
                    delta=f"${price_difference:.2f}"
                )


            with col3:

                st.metric(
                    "Expected Return",
                    f"{expected_return:.2f}%"
                )


            with col4:

                st.metric(
                    "Signal",
                    signal
                )


            # =================================================
            # SIGNAL DISPLAY
            # =================================================

            st.markdown("---")


            if signal == "BUY":

                st.success(
                    f"""
                    🟢 BUY SIGNAL

                    {ticker} is predicted to move
                    upward.

                    Expected Return:
                    {expected_return:.2f}%
                    """
                )


            elif signal == "SELL":

                st.error(
                    f"""
                    🔴 SELL SIGNAL

                    {ticker} is predicted to move
                    downward.

                    Expected Return:
                    {expected_return:.2f}%
                    """
                )


            else:

                st.warning(
                    f"""
                    🟡 HOLD SIGNAL

                    No strong upward or downward
                    prediction.

                    Expected Return:
                    {expected_return:.2f}%
                    """
                )


            # =================================================
            # PREDICTION DETAILS
            # =================================================

            st.subheader(
                "Prediction Details"
            )


            prediction_df = pd.DataFrame({

                "Metric": [

                    "Stock",

                    "Date",

                    "Current Price",

                    "Predicted Price",

                    "Expected Return",

                    "Signal",

                    "Threshold"
                ],

                "Value": [

                    ticker,

                    str(
                        latest_date.date()
                    ),

                    f"${current_price:.2f}",

                    f"${predicted_price:.2f}",

                    f"{expected_return:.2f}%",

                    signal,

                    f"{threshold:.2f}%"
                ]
            })


            st.dataframe(

                prediction_df,

                use_container_width=True,

                hide_index=True
            )


            # =================================================
            # PREDICTION VISUAL
            # =================================================

            st.subheader(
                "Current vs Predicted Price"
            )


            prediction_chart = go.Figure()


            prediction_chart.add_trace(

                go.Bar(

                    x=[
                        "Current Price",
                        "Predicted Price"
                    ],

                    y=[
                        current_price,
                        predicted_price
                    ],

                    text=[
                        f"${current_price:.2f}",
                        f"${predicted_price:.2f}"
                    ],

                    textposition="auto"
                )
            )


            prediction_chart.update_layout(

                height=400,

                yaxis_title="Price",

                showlegend=False
            )


            st.plotly_chart(

                prediction_chart,

                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


else:

    st.info(
        "👈 Select a stock and click "
        "**🚀 Predict** to generate the ML prediction."
    )


# ============================================================
# RECENT DATA
# ============================================================

st.markdown("---")

st.header(
    "📋 Recent Market Data"
)


st.dataframe(

    df.tail(10),

    use_container_width=True,

    hide_index=True
)


# ============================================================
# BACKTESTING
# ============================================================

st.markdown("---")

st.header(
    "📊 Backtesting Performance"
)


BACKTEST_FILE = (

    DATA_DIR
    / f"{ticker}_backtest_results.csv"
)


if BACKTEST_FILE.exists():

    try:

        backtest = pd.read_csv(
            BACKTEST_FILE
        )


        if "Date" in backtest.columns:

            backtest["Date"] = (
                pd.to_datetime(
                    backtest["Date"]
                )
            )


        if (

            "Strategy_Equity"
            in backtest.columns

            and

            "Buy_Hold_Equity"
            in backtest.columns

        ):

            equity_fig = go.Figure()


            equity_fig.add_trace(

                go.Scatter(

                    x=backtest["Date"],

                    y=backtest[
                        "Strategy_Equity"
                    ],

                    mode="lines",

                    name="AlphaTrade"
                )
            )


            equity_fig.add_trace(

                go.Scatter(

                    x=backtest["Date"],

                    y=backtest[
                        "Buy_Hold_Equity"
                    ],

                    mode="lines",

                    name="Buy & Hold"
                )
            )


            equity_fig.update_layout(

                title=(
                    f"{ticker} Portfolio "
                    "Equity Curve"
                ),

                height=450,

                xaxis_title="Date",

                yaxis_title="Portfolio Value",

                hovermode="x unified"
            )


            st.plotly_chart(

                equity_fig,

                use_container_width=True
            )


        else:

            st.warning(
                "Backtest file does not contain "
                "equity curve columns."
            )


    except Exception as e:

        st.error(
            f"Unable to load backtest: {e}"
        )

else:

    st.info(
        f"""
        Backtesting data for {ticker}
        is not available yet.

        Run the backtesting notebook first.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "📈 AlphaTrade | Algorithmic Stock Prediction Engine"
)

st.caption(
    "⚠️ Educational/research project. "
    "Predictions are not financial advice."
)