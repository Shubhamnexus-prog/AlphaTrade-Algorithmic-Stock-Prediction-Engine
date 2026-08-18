from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path=None
):
    """
    Load trained machine learning model.
    """

    if model_path is None:
        model_path = MODEL_DIR / "best_model.pkl"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}\n\n"
            "Run 03_model_training.ipynb first."
        )

    model = joblib.load(
        model_path
    )

    return model


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

def load_feature_columns(
    feature_path=None
):
    """
    Load feature columns used during model training.
    """

    if feature_path is None:
        feature_path = (
            MODEL_DIR /
            "feature_columns.pkl"
        )

    feature_path = Path(
        feature_path
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found:\n"
            f"{feature_path}\n\n"
            "Save feature_columns.pkl during model training."
        )

    feature_columns = joblib.load(
        feature_path
    )

    return feature_columns


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data(
    ticker="AAPL"
):
    """
    Load processed feature dataset.
    """

    file_path = (
        DATA_DIR /
        f"{ticker}_features.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARE LATEST DATA
# ============================================================

def prepare_latest_input(
    df,
    feature_columns
):
    """
    Prepare latest row for prediction.
    """

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing model features:\n"
            + str(missing_columns)
        )

    latest_row = df.iloc[-1]

    X_latest = (
        latest_row[
            feature_columns
        ]
        .to_frame()
        .T
    )

    return latest_row, X_latest


# ============================================================
# PREDICT PRICE
# ============================================================

def predict_price(
    model,
    X_latest
):
    """
    Predict next stock price.
    """

    prediction = model.predict(
        X_latest
    )

    return float(
        prediction[0]
    )


# ============================================================
# EXPECTED RETURN
# ============================================================

def calculate_expected_return(
    current_price,
    predicted_price
):
    """
    Calculate expected percentage return.
    """

    if current_price == 0:
        return 0.0

    expected_return = (
        (
            predicted_price
            - current_price
        )
        / current_price
    ) * 100

    return float(
        expected_return
    )


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(
    expected_return,
    threshold=0.5
):
    """
    Generate BUY / SELL / HOLD signal.
    """

    if expected_return > threshold:

        return "BUY"

    elif expected_return < -threshold:

        return "SELL"

    return "HOLD"


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def make_prediction(
    ticker="AAPL",
    threshold=0.5
):
    """
    Complete AlphaTrade prediction pipeline.
    """

    # Load model
    model = load_model()

    # Load feature columns
    feature_columns = (
        load_feature_columns()
    )

    # Load processed data
    df = load_feature_data(
        ticker
    )

    # Prepare latest row
    latest_row, X_latest = (
        prepare_latest_input(
            df,
            feature_columns
        )
    )

    # Current price
    current_price = float(
        latest_row["Close"]
    )

    # Predicted price
    predicted_price = (
        predict_price(
            model,
            X_latest
        )
    )

    # Expected return
    expected_return = (
        calculate_expected_return(
            current_price,
            predicted_price
        )
    )

    # Signal
    signal = generate_signal(
        expected_return,
        threshold
    )

    result = {

        "ticker": ticker,

        "date": latest_row["Date"],

        "current_price":
            current_price,

        "predicted_price":
            predicted_price,

        "expected_return":
            expected_return,

        "signal":
            signal
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    try:

        result = make_prediction(
            ticker="AAPL",
            threshold=0.5
        )

        print()
        print("=" * 60)
        print("           ALPHATRADE PREDICTION")
        print("=" * 60)

        print(
            f"Ticker           : "
            f"{result['ticker']}"
        )

        print(
            f"Date             : "
            f"{result['date']}"
        )

        print(
            f"Current Price    : "
            f"${result['current_price']:.2f}"
        )

        print(
            f"Predicted Price  : "
            f"${result['predicted_price']:.2f}"
        )

        print(
            f"Expected Return  : "
            f"{result['expected_return']:.2f}%"
        )

        print(
            f"Signal            : "
            f"{result['signal']}"
        )

        print("=" * 60)

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )