import pandas as pd
from pathlib import Path


def clean_stock_data(input_file, output_file):
    """
    Clean raw stock market data.
    """

    print("Loading raw dataset...")

    df = pd.read_csv(input_file)

    # Remove unwanted index columns
    unwanted_columns = [
        col for col in df.columns
        if str(col).startswith("Unnamed")
    ]

    if unwanted_columns:
        df.drop(columns=unwanted_columns, inplace=True)

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"])

    # Convert numeric columns
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Handle missing values
    df[numeric_columns] = df[numeric_columns].ffill()

    # Remove remaining missing rows
    df.dropna(inplace=True)

    # Save cleaned data
    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nCleaning completed!")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_file}")

    return df


if __name__ == "__main__":

    input_file = "data/raw/AAPL_historical.csv"
    output_file = "data/processed/AAPL_cleaned.csv"

    df = clean_stock_data(
        input_file,
        output_file
    )

    print("\nCleaned dataset:")
    print(df.head())