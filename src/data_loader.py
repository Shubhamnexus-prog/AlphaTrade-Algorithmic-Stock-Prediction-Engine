import yfinance as yf
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"


def download_stock_data(
    ticker="AAPL",
    start="2015-01-01",
    end=None
):
    """
    Download historical stock data from Yahoo Finance.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Downloading {ticker} stock data...")

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    # Handle yfinance MultiIndex
    if hasattr(data.columns, "nlevels"):

        if data.columns.nlevels > 1:

            data.columns = (
                data.columns
                .get_level_values(0)
            )

    data.reset_index(
        inplace=True
    )

    file_path = (
        RAW_DIR /
        f"{ticker}_historical.csv"
    )

    data.to_csv(
        file_path,
        index=False
    )

    print(
        f"Data saved: {file_path}"
    )

    print(
        f"Rows: {len(data)}"
    )

    return data


if __name__ == "__main__":

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA"
    ]

    for ticker in tickers:

        try:

            download_stock_data(
                ticker=ticker,
                start="2015-01-01"
            )

        except Exception as e:

            print(
                f"Failed {ticker}: {e}"
            )