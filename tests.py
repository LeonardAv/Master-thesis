import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from Leveraged_Path_Code import leveragedPath

# Example leverage setup (adapt if needed)
leverage_values = [1, 1.28, 2, 3, 4, 5]
leverage_labels = ['1x', 'Worst-Case', '2x', 'Mid-Case', '4x', '5x']

colors = [
    "#000000",  # black (1x)
    "#1C3879",  # medium navy-blue (2x)
    "#2878B5",  # bright steel blue (3x)
    "#41AB5D",  # muted green (4x)
    "#A6D96A",  # light olive (5x)
    "#FDAE61",  # soft orange (Worst-Case)
    "#911D09"   # dark red (Mid-Case)
]

def days_to_recovery_for_leverage(lev,
                                  start_date="2005-01-01",
                                  end_date="2025-01-01",
                                  start_crisis="2020-02-01",
                                  end_crisis="2022-02-01"):
    """
    For a given leverage lev, compute:
    - Pre-crisis peak level of the leveraged path
    - Date of this peak
    - First recovery date where price >= pre-crisis peak
    - Number of days between peak and recovery (calendar + trading)
    """
    
    data_long = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
    data_stress = yf.Ticker("^GSPC").history(start=start_crisis, end=end_crisis)["Close"].dropna()

    initial_value_long = data_long.iloc[0]
    initial_value_stress = data_stress.iloc[0]

    path_leveraged_long_before_stress = leveragedPath(data_long, lev, initial_value_long)
    path_leveraged_just_before_stress = leveragedPath(data_stress, lev, initial_value_stress)

    lowest_point_crisis_idx = path_leveraged_just_before_stress["price"].idxmin()
    lowest_point_crisis_price = path_leveraged_just_before_stress.loc[lowest_point_crisis_idx, "price"]
    lowest_point_crisis_date = pd.to_datetime(path_leveraged_just_before_stress.loc[lowest_point_crisis_idx, "date"])
    
    path_before_lowest_point = pd.DataFrame(path_leveraged_just_before_stress)
    path_before_lowest_point["date"] = pd.to_datetime(path_before_lowest_point["date"])
    path_before_lowest_point = path_before_lowest_point[path_before_lowest_point['date'] < lowest_point_crisis_date]
    path_after_lowest_point = pd.DataFrame(path_leveraged_just_before_stress)
    path_after_lowest_point["date"] = pd.to_datetime(path_after_lowest_point["date"])
    path_after_lowest_point = path_after_lowest_point[path_after_lowest_point['date'] > lowest_point_crisis_date]
    
    if len(path_before_lowest_point) != 0:
        idx_max_before_stress = path_before_lowest_point["price"].idxmax()
        max_price_before_stress = path_before_lowest_point.loc[idx_max_before_stress, "price"]
        max_price_before_stress_date = path_before_lowest_point.loc[idx_max_before_stress, "date"]
    else:
        max_price_before_stress = lowest_point_crisis_price
        max_price_before_stress_date = lowest_point_crisis_date
        
    path_recovered = path_after_lowest_point[path_after_lowest_point["price"] >= max_price_before_stress]
    print(path_recovered)

    if path_recovered.empty:
        recovery_date = None
        calendar_days = None
        trading_days = None
    else:
        recovery_date = path_recovered["date"].iloc[0]
        print(recovery_date)
        calendar_days = (recovery_date - max_price_before_stress_date).days
        #trading_days = len(after_peak.loc[:recovery_date]) - 1  # exclude peak day itself

    return {
        "leverage": lev,
        "peak_date": max_price_before_stress_date,
        "peak_price": max_price_before_stress,
        "recovery_date": recovery_date,
        "calendar_days": calendar_days,
        #"trading_days": trading_days,
    }

def days_to_recovery_all_leverages(leverage_values,
                                   start_date="2005-01-01",
                                   end_date="2025-01-01",
                                   start_crisis="2020-02-01",
                                   end_crisis="2022-02-01"):
    """
    Compute days to recovery for a list of leverages.
    Returns a DataFrame with results and optionally a bar plot.
    """
    results = []
    for lev in leverage_values:
        res = days_to_recovery_for_leverage(
            lev,
            start_date=start_date,
            end_date=end_date,
            start_crisis=start_crisis,
            end_crisis=end_crisis,
        )
        results.append(res)

    df = pd.DataFrame(results)
    return df

# ---- Example usage ----
#start_date_comparisons = "2005-01-01"
#end_date_comparisons   = "2025-01-01"
#start_crisis           = "2020-02-01",
#end_crisis             = "2022-02-01"

recovery_df = days_to_recovery_all_leverages(
    leverage_values
)

print(recovery_df)


'''
# Optional: bar plot of trading days to recovery
plt.figure(figsize=(8, 5))
plt.bar(
    [str(lv) for lv in recovery_df["leverage"]],
    recovery_df["trading_days"],
    color="steelblue",
    edgecolor="black"
)
plt.ylabel("Trading Days to Recovery")
plt.xlabel("Leverage Ratio")
plt.title("Days to Recover Pre-COVID Peak by Leverage Strategy")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
'''