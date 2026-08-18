import pandas as pd
import numpy as np


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(
    file_path
):
    """
    Load model test predictions.
    """

    df = pd.read_csv(
        file_path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df.sort_values("Date")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# CREATE EXPECTED RETURN
# ============================================================

def calculate_expected_return(
    df
):
    """
    Calculate expected return from predicted price.
    """

    df = df.copy()

    df["Expected_Return"] = (
        (
            df["Predicted"]
            - df["Actual"]
        )
        / df["Actual"]
    )

    return df


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signals(
    df,
    threshold=0.005
):
    """
    Generate BUY / HOLD / SELL numeric signals.

    +1 = BUY
     0 = HOLD
    -1 = SELL
    """

    df = df.copy()

    if "Expected_Return" not in df.columns:

        df = calculate_expected_return(
            df
        )

    df["Signal"] = np.where(
        df["Expected_Return"] > threshold,
        1,
        np.where(
            df["Expected_Return"] < -threshold,
            -1,
            0
        )
    )

    return df


# ============================================================
# CALCULATE RETURNS
# ============================================================

def calculate_returns(
    df
):
    """
    Calculate market and strategy returns.
    """

    df = df.copy()

    df["Actual_Return"] = (
        df["Actual"]
        .pct_change()
    )

    df["Actual_Return"] = (
        df["Actual_Return"]
        .fillna(0)
    )

    # Use previous day's signal
    # to avoid look-ahead bias.
    df["Strategy_Return"] = (
        df["Signal"].shift(1)
        * df["Actual_Return"]
    )

    df["Strategy_Return"] = (
        df["Strategy_Return"]
        .fillna(0)
    )

    df["Buy_Hold_Return"] = (
        df["Actual_Return"]
    )

    return df


# ============================================================
# EQUITY CURVE
# ============================================================

def calculate_equity_curve(
    df,
    initial_capital=100000
):
    """
    Calculate strategy and buy-and-hold
    portfolio values.
    """

    df = df.copy()

    df["Strategy_Equity"] = (
        initial_capital
        * (
            1
            + df["Strategy_Return"]
        ).cumprod()
    )

    df["Buy_Hold_Equity"] = (
        initial_capital
        * (
            1
            + df["Buy_Hold_Return"]
        ).cumprod()
    )

    return df


# ============================================================
# MAX DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    equity
):
    """
    Calculate maximum drawdown.
    """

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    return float(
        drawdown.min() * 100
    )


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe_ratio(
    returns
):
    """
    Calculate annualized Sharpe ratio.
    """

    std = returns.std()

    if std == 0 or pd.isna(std):

        return 0.0

    sharpe = (
        returns.mean()
        / std
    ) * np.sqrt(252)

    return float(
        sharpe
    )


# ============================================================
# WIN RATE
# ============================================================

def calculate_win_rate(
    df
):
    """
    Calculate strategy win rate.
    """

    active_trades = df[
        df["Signal"] != 0
    ]

    if len(active_trades) == 0:

        return 0.0

    winning = (
        active_trades[
            "Strategy_Return"
        ] > 0
    ).sum()

    return float(
        winning
        / len(active_trades)
        * 100
    )


# ============================================================
# COMPLETE BACKTEST
# ============================================================

def run_backtest(
    df,
    initial_capital=100000,
    threshold=0.005
):
    """
    Run complete AlphaTrade backtest.
    """

    df = df.copy()

    df = calculate_expected_return(
        df
    )

    df = generate_signals(
        df,
        threshold
    )

    df = calculate_returns(
        df
    )

    df = calculate_equity_curve(
        df,
        initial_capital
    )

    final_strategy_value = (
        df["Strategy_Equity"].iloc[-1]
    )

    final_buy_hold_value = (
        df["Buy_Hold_Equity"].iloc[-1]
    )

    strategy_return = (
        final_strategy_value
        / initial_capital
        - 1
    ) * 100

    buy_hold_return = (
        final_buy_hold_value
        / initial_capital
        - 1
    ) * 100

    max_drawdown = (
        calculate_max_drawdown(
            df["Strategy_Equity"]
        )
    )

    sharpe = (
        calculate_sharpe_ratio(
            df["Strategy_Return"]
        )
    )

    win_rate = (
        calculate_win_rate(
            df
        )
    )

    number_of_trades = (
        df["Signal"]
        .diff()
        .abs()
        .fillna(0)
        .gt(0)
        .sum()
    )

    results = {

        "Initial Capital":
            initial_capital,

        "Final Strategy Value":
            final_strategy_value,

        "Final Buy & Hold Value":
            final_buy_hold_value,

        "Strategy Return (%)":
            strategy_return,

        "Buy & Hold Return (%)":
            buy_hold_return,

        "Maximum Drawdown (%)":
            max_drawdown,

        "Sharpe Ratio":
            sharpe,

        "Number of Trades":
            int(number_of_trades),

        "Win Rate (%)":
            win_rate
    }

    return df, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    file_path = (
        "data/processed/"
        "AAPL_test_predictions.csv"
    )

    df = load_predictions(
        file_path
    )

    backtest_df, results = (
        run_backtest(
            df,
            initial_capital=100000,
            threshold=0.005
        )
    )

    print("\n")
    print("=" * 60)
    print("          ALPHATRADE BACKTEST")
    print("=" * 60)

    for key, value in results.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key:<30}: {value:.2f}"
            )

        else:

            print(
                f"{key:<30}: {value}"
            )

    print("=" * 60)