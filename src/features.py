import sys
from pathlib import Path

import pandas as pd


# Add src folder to Python import path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indicators import add_technical_indicators


def create_features(input_file, output_file):
    """
    Create technical indicators,
    lag features, momentum features
    and prediction target.
    """

    print("Loading cleaned dataset...")

    df = pd.read_csv(input_file)

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------
    # Technical Indicators
    # --------------------------------

    df = add_technical_indicators(df)

    # --------------------------------
    # Lag Features
    # --------------------------------

    df["Close_Lag_1"] = df["Close"].shift(1)
    df["Close_Lag_2"] = df["Close"].shift(2)
    df["Close_Lag_3"] = df["Close"].shift(3)
    df["Close_Lag_5"] = df["Close"].shift(5)
    df["Close_Lag_10"] = df["Close"].shift(10)

    # --------------------------------
    # Return Lag Features
    # --------------------------------

    df["Return_Lag_1"] = df["Daily_Return"].shift(1)
    df["Return_Lag_2"] = df["Daily_Return"].shift(2)
    df["Return_Lag_5"] = df["Daily_Return"].shift(5)

    # --------------------------------
    # Momentum
    # --------------------------------

    df["Momentum_5"] = (
        df["Close"] / df["Close"].shift(5) - 1
    )

    df["Momentum_10"] = (
        df["Close"] / df["Close"].shift(10) - 1
    )

    df["Momentum_20"] = (
        df["Close"] / df["Close"].shift(20) - 1
    )

    # --------------------------------
    # Target
    # --------------------------------

    # Next trading day's closing price
    df["Target"] = df["Close"].shift(-1)

    # --------------------------------
    # Remove NaN
    # --------------------------------

    df = df.dropna().reset_index(drop=True)

    # --------------------------------
    # Save Dataset
    # --------------------------------

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nFeature engineering completed!")
    print("--------------------------------")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nFeatures:")
    print(df.columns.tolist())

    print("\nSaved to:")
    print(output_file)

    return df


if __name__ == "__main__":

    # --------------------------------
    # Project Root
    # --------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --------------------------------
    # Input / Output
    # --------------------------------

    input_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "AAPL_cleaned.csv"
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "AAPL_features.csv"
    )

    # --------------------------------
    # Check Input
    # --------------------------------

    if not input_file.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{input_file}\n\n"
            "Run preprocessing.py first."
        )

    # --------------------------------
    # Create Features
    # --------------------------------

    df = create_features(
        input_file,
        output_file
    )

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nFinal Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum().sum())


