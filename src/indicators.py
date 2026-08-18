import pandas as pd


def add_technical_indicators(df):
    """
    Add technical indicators to stock dataset.
    """

    df = df.copy()

    # -------------------------
    # Moving Averages
    # -------------------------

    df["SMA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["SMA_200"] = (
        df["Close"]
        .rolling(window=200)
        .mean()
    )

    # EMA
    df["EMA_20"] = (
        df["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # -------------------------
    # RSI
    # -------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI_14"] = 100 - (
        100 / (1 + rs)
    )

    # -------------------------
    # MACD
    # -------------------------

    ema_12 = (
        df["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema_26 = (
        df["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_Histogram"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # -------------------------
    # Bollinger Bands
    # -------------------------

    rolling_mean = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    df["BB_Middle"] = rolling_mean

    df["BB_Upper"] = (
        rolling_mean +
        2 * rolling_std
    )

    df["BB_Lower"] = (
        rolling_mean -
        2 * rolling_std
    )

    # -------------------------
    # Daily Return
    # -------------------------

    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    # -------------------------
    # Volatility
    # -------------------------

    df["Volatility_20"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    return df