import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 14
#

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14}

leverage_values = [1, 2, 3, 4, 5, 1.28, 3]
leverage_values_ordered = [1, 1.28, 2, 3, 4, 5]
leverage_labels = ['1x', '2x', '3x', '4x', '5x', 'Worst-Case', 'Mid-Case']
leverage_labels_ordered = ['1x', 'Worst-Case', '2x', 'Mid-Case', '4x', '5x']

colors = [
    "#000000",  # black (1x)
    "#1C3879",  # medium navy-blue (2x)
    "#2878B5",  # bright steel blue (3x)
    "#41AB5D",  # muted green (4x)
    "#A6D96A",  # light olive (5x)
    "#FDAE61",  # soft orange (Worst-Case)
    "#911D09"   # neutral grey (Mid-Case)
]

start_date = "2000-01-01"
end_date = "2025-01-01"

def leveragedPath(data, leverage, initial_value):
    path = []
    previous_value = initial_value
    
    for i, (date, price) in enumerate(data.items()):
        if i == 0:
            path.append({"date": date, "price": previous_value})
            prev_price = price
        else:
            daily_return = (price / prev_price) - 1
            leveraged_return = leverage * daily_return
            new_value = previous_value * (1 + leveraged_return)
            path.append({"date": date, "price": new_value})
            prev_price = price
            previous_value = new_value

    return pd.DataFrame(path)



def plotpaths(start_date, case, filename, end_date = "2025-01-01"):
    # Download data
    dataIndexSP500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"]

    initial_value = dataIndexSP500.iloc[0]

    leveraged_index = leveragedPath(dataIndexSP500, 1, initial_value)
    leveraged_2x_sp = leveragedPath(dataIndexSP500, 2, initial_value)
    leveraged_3x_sp = leveragedPath(dataIndexSP500, 3, initial_value)
    leveraged_4x_sp = leveragedPath(dataIndexSP500, 4, initial_value)
    leveraged_5x_sp = leveragedPath(dataIndexSP500, 5, initial_value)

    leveraged_worst_case = leveragedPath(dataIndexSP500, 1.28, initial_value)
    leveraged_mid_case = leveragedPath(dataIndexSP500, 3, initial_value)

    result_unlev = dataIndexSP500.iloc[-1]
    result_2x = leveragedPath(dataIndexSP500, 2, initial_value).iloc[-1]['price']
    result_3x = leveragedPath(dataIndexSP500, 3, initial_value).iloc[-1]['price']
    result_4x = leveragedPath(dataIndexSP500, 4, initial_value).iloc[-1]['price']
    result_5x = leveragedPath(dataIndexSP500, 5, initial_value).iloc[-1]['price']

    result_worst_case = leveragedPath(dataIndexSP500, 1.28, initial_value).iloc[-1]['price']
    result_mid_case = leveragedPath(dataIndexSP500, 3, initial_value).iloc[-1]['price']

    values = [result_unlev, result_2x, result_3x, result_4x, result_5x, result_worst_case, result_mid_case]
    if case == "final_values":
        # Plot final values bar chart
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, len(leverage_labels) - 1, len(leverage_labels)) * 0.7
        bars = plt.bar(x, values, color=colors, width=0.5)
        for bar in bars:
            height = bar.get_height()
            bar_color = bar.get_facecolor()
            plt.text(
                bar.get_x() + bar.get_width() / 2,   # x position (center of bar)
                height + max(values) * 0.015,        # y position (slightly above bar)
                f'{height:,.0f}',                    # formatted value
                ha='center', va='bottom',
                fontsize=8, color=bar_color, fontweight='bold'
            )
        plt.xticks(x, leverage_labels, rotation=45, ha='right')
        plt.title("Final Portfolio Value for Different Leverage Strategies (start_date = " + start_date + ")", fontdict=font)
        plt.ylabel("Final Value", fontdict=font)
        plt.xlabel("Leverage Ratio", fontdict=font)
        plt.grid(True, axis='y', alpha=0.4)
        plt.tight_layout()
        #plt.show()
        plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"✅ Figure saved as: {filename}")
    elif case == "all_paths":
        # Plot 1 (Different leverages)
        plt.figure(figsize=(12, 6))
        plt.plot(dataIndexSP500.index, dataIndexSP500.values, label="S&P 500 (^GSPC) - unleveraged", color=colors[0])
        plt.plot(leveraged_2x_sp["date"], leveraged_2x_sp["price"], label="Synthetic 2x Leverage", color=colors[1])
        plt.plot(leveraged_3x_sp["date"], leveraged_3x_sp["price"], label="Synthetic 3x Leverage", color=colors[2])
        plt.plot(leveraged_4x_sp["date"], leveraged_4x_sp["price"], label="Synthetic 4x Leverage", color=colors[3])
        plt.plot(leveraged_5x_sp["date"], leveraged_5x_sp["price"], label="Synthetic 5x Leverage", color=colors[4])
        plt.title("Different leveraged S&P 500 Strategies", fontdict=font)
        plt.xlabel("Date", fontdict=font)
        plt.ylabel("Portfolio Value", fontdict=font)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"✅ Figure saved as: {filename}")
    elif case == "worst_vs_mid":
        # Plot 2 (Compare worst against mid and unleveraged)
        plt.figure(figsize=(12, 6))
        plt.plot(leveraged_worst_case["date"], leveraged_worst_case["price"], label="Leverage under worst case", color=colors[5])
        plt.plot(leveraged_mid_case["date"], leveraged_mid_case["price"], label="Leverage under mid case", color=colors[6])
        plt.plot(dataIndexSP500.index, dataIndexSP500.values, label="S&P 500 (^GSPC) - unleveraged", color=colors[0])
        plt.title("Different leverage strategies for the S&P 500", fontdict=font)
        plt.xlabel("Date", fontdict=font)
        plt.ylabel("Value", fontdict=font)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"✅ Figure saved as: {filename}")
   

def plot_procentage_return(start_date, filename, end_date="2025-01-01"):
    data = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
    initial_value = data.iloc[0]
    investment_time_year = (data.index[-1] - data.index[0]).days / 365.25
    returns = []
    for lev in leverage_values:
        if lev == 1:
            final = data.iloc[-1]
        else:
            path = leveragedPath(data, lev, initial_value)
            final = path.iloc[-1]["price"]

        # Annualized percentage return
        percent_return = ((final / initial_value) ** (1 / investment_time_year) - 1) * 100
        returns.append(percent_return)
    # --- Plot ---
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, len(leverage_labels) - 1, len(leverage_labels)) * 0.7
    bars = plt.bar(x, returns, color=colors, width=0.5)      
    for bar in bars:
            height = bar.get_height()
            bar_color = bar.get_facecolor()
            plt.text(
                bar.get_x() + bar.get_width() / 2,   # x position (center of bar)
                height + max(returns) * 0.015,       # y position (slightly above bar)
                f'{height:,.0f}',                    # formatted value
                ha='center', va='bottom',
                fontsize=8, color=bar_color, fontweight='bold'
            )
    plt.xticks(x, leverage_labels, rotation=45, ha='right')    
    plt.title(f"Annualized Return by Leverage Strategy (start_date = {start_date})", fontdict=font)
    plt.ylabel("Annualized Return (%)", fontdict=font)
    plt.xlabel("Leverage Ratio", fontdict=font)
    plt.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close() 
    print(f"✅ Figure saved as: {filename}")
     
        
def plot_procentage_returns(start_dates, leverage_values_for_plot, leverage_labels_for_plot, filename, end_date="2025-01-01"):
    strategy_returns = []
    for start_date in start_dates:
        data = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
        initial_value = data.iloc[0]
        investment_time_year = (data.index[-1] - data.index[0]).days / 365.25
        returns = []
        for lev in leverage_values_for_plot:
            if lev == 1:
                final = data.iloc[-1]
            else:
                path = leveragedPath(data, lev, initial_value)
                final = path.iloc[-1]["price"]
            percent_return = ((final / initial_value) ** (1 / investment_time_year) - 1) * 100
            returns.append(percent_return)
        strategy_returns.append(returns)
    # Convert to numpy array
    strategy_returns = np.array(strategy_returns)  # shape (n_start_dates, n_strategies)
    # Plot grouped bar plot (grouped by start date)
    x = np.arange(len(start_dates))  # start date positions
    bar_width = 0.12
    plt.figure(figsize=(12, 6))
    for i in range(len(leverage_labels_for_plot)):
        # center around each group
        offset = (i - (len(leverage_labels_for_plot)-1)/2) * bar_width
        #find correct color (search in leverage_labels the index of the label in leverage_labels and get the color from colors)
        j = leverage_labels.index(leverage_labels_for_plot[i])
        plt.bar(x + offset, strategy_returns[:, i], width=bar_width, label=leverage_labels_for_plot[i], color=colors[j])

    # Formatting
    plt.xticks(x, [d[:4] for d in start_dates])
    plt.xlabel("Start Date", fontdict=font)
    plt.ylabel("Annualized Return (%)", fontdict=font)
    plt.title("Annualized Percentage Return by Start Date and Leverage Strategy", fontdict=font)
    plt.legend(title="Strategy", loc="best")
    plt.grid(True, axis='y')
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close() 
    print(f"✅ Figure saved as: {filename}")
    

def plot_procentage_returns_different_starts_worst_mid(start_dates, filename, end_date="2025-01-01"):
        
    mid_case_returns = []
    worst_case_returns = []
    for start_date in start_dates:
        data = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
        initial_value = data.iloc[0]
        investment_time_year = (data.index[-1] - data.index[0]).days / 365.25

        # Mid case
        mid_path = leveragedPath(data, 3, initial_value)
        mid_final = mid_path.iloc[-1]["price"]
        mid_return = ((mid_final / initial_value) ** (1 / investment_time_year) - 1) * 100
        mid_case_returns.append(mid_return)

        # Worst case
        worst_path = leveragedPath(data, 1.28, initial_value)
        worst_final = worst_path.iloc[-1]["price"]
        worst_return = ((worst_final / initial_value) ** (1 / investment_time_year) - 1) * 100
        worst_case_returns.append(worst_return)
    # Plotting
    x = np.arange(len(start_dates))  # start date positions
    bar_width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, mid_case_returns, width=bar_width, label="Mid Case (3x Leverage)", color=colors[6])
    plt.bar(x + bar_width/2, worst_case_returns, width=bar_width, label="Worst Case (1.28x Leverage)", color=colors[5])
    # Formatting
    plt.xticks(x, [d[:4] for d in start_dates])
    plt.xlabel("Start Date", fontdict=font)
    plt.ylabel("Annualized Return (%)", fontdict=font)
    plt.title("Annualized Percentage Return for Mid and Worst Case Leverage Strategies", fontdict=font)
    plt.legend(title="Strategy", loc="upper left")
    plt.grid(True, axis='y')
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")
    
    
def plot_stacked_paths_selected_leverages(multiple_dates, filename, selected_leverages=[1, 1.28, 3, 5], end_date="2025-01-01"):
    n = len(multiple_dates)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

    if n == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i, date in enumerate(multiple_dates):
        data = yf.Ticker("^GSPC").history(start=date, end=end_date)["Close"].dropna()
        initial_value = data.iloc[0]
        paths = {}
        lev_labels = []
        for lev_value in selected_leverages:
            if lev_value == 1:
                lev_label = '1x'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 1, initial_value)
            elif lev_value == 1.28:
                lev_label = 'Worst-Case'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 1.28, initial_value)
            elif lev_value == 2:
                lev_label = '2x'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 2, initial_value)
            elif lev_value == 3:
                lev_label = 'Mid-Case'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 3, initial_value)
            elif lev_value == 4:
                lev_label = '4x'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 4, initial_value)
            elif lev_value == 5:
                lev_label = '5x'
                lev_labels.append(lev_label)
                paths[lev_label] = leveragedPath(data, 5, initial_value)

        ax = axes[i]
        for label in lev_labels:
            df = paths[label]
            ax.plot(df['date'], df['price'], label=label)
        
        # colouring corresponding to leverage
        for j, label in enumerate(lev_labels):
            ax.lines[j].set_color(colors[leverage_labels.index(label)])
        ax.set_title(f"Starting from {date}", fontdict=font)
        ax.set_ylabel("Value", fontdict=font)
        ax.grid(True)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.suptitle("Performance of Different Leverages from Various Starting Points", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")


def compute_historical_VaR(price_series, confidence_level):
    # Compute daily returns
    daily_returns = price_series.pct_change().dropna()

    # Compute historical VaR at the (1 - confidence_level) quantile
    var_level = (1 - confidence_level)
    VaR = np.quantile(daily_returns, var_level)

    return VaR



def plot_var_histogram(price_series, title, filename, confidence_level=0.95, bins=50):
    daily_returns = price_series.pct_change().dropna()
    var_value = np.quantile(daily_returns, (1 - confidence_level))

    plt.figure(figsize=(10, 5))
    plt.hist(daily_returns, bins=bins, color="skyblue", edgecolor="black")
    plt.axvline(var_value, color='red', linestyle='--', label=f'{int((1-confidence_level)*100)}% VaR: {var_value:.2%}')
    plt.title(title, fontdict=font)
    plt.xlabel("Daily Return", fontdict=font)
    plt.ylabel("Frequency", fontdict=font)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")
    
    
    
def plot_returns_with_var_threshold(price_series, date_series, leverage_val, title, filename, confidence_level=0.95):
    returns = price_series.pct_change().dropna()
    var_value = np.quantile(returns, (1 - confidence_level))
    
    breaches = returns < var_value
    
    date_series = pd.Index(date_series)
    aligned_dates = date_series[1:]
    breach_dates = aligned_dates[breaches.values]
    
    col_index = leverage_values.index(leverage_val)
    color = colors[col_index]

    plt.figure(figsize=(12, 5))
    plt.plot(aligned_dates, returns.values, label="Daily Return", color=color)
    plt.axhline(var_value, color='red', linestyle='--', label=f'{int((1-confidence_level)*100)}% VaR: {var_value:.2%}')
    plt.fill_between(aligned_dates, returns.values, var_value, where=(returns.values < var_value), color='red', alpha=0.3)
    
    print(f"Breaches: ", breaches)
    print(f"Number of VaR breaches: {breaches.sum()}")
    print(f"Number of total observations: {len(returns)}")
    print(f"Proportion of VaR breaches: {breaches.sum() / len(returns):.2%}")
    #plt.scatter(returns.index[breaches], returns[breaches], color='red', s=10, label="VaR breaches")
    
    plt.title(title, fontdict=font)
    plt.xlabel("Date", fontdict=font)
    plt.ylabel("Return", fontdict=font)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    filename1 = filename + "_Daily_Returns.pdf"
    plt.savefig(filename1, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename1}")
    

    # Additional plot of breaches timeline
    plt.figure(figsize=(12, 2.5))
    plt.scatter(breach_dates, np.ones_like(breach_dates),
                color=color, s=12, label="VaR breach")

    plt.yticks([])
    plt.title("Timeline of VaR Breaches", fontdict=font)
    plt.xlabel("Date", fontdict=font)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    #plt.show()
    filename2 = filename + "_Timeline_VaR_Breaches.pdf"
    plt.savefig(filename2, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename2}")
    
    
    print("Date seies:", breach_dates)
    # Additional plot of breaches histogram
    plt.figure(figsize=(10, 5))
    plt.hist(breach_dates.year, bins=range(breach_dates.year.min(), breach_dates.year.max() + 2),
             color=color, edgecolor="black")
    plt.title("Histogram of VaR Breaches by Year", fontdict=font)
    plt.xlabel("Year", fontdict=font)
    plt.ylabel("Number of Breaches", fontdict=font)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    filename3 = filename + "_Histogram_VaR_Breaches.pdf"
    plt.savefig(filename3, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename3}")




def plot_overlaid_var_histograms(paths, labels_for_plot, values_for_plot, filename, confidence_level=0.95, bins=200):
    """
    Plot overlaid histograms of daily returns for multiple strategies with VaR lines,
    each colored and labeled consistently.
    """
    plt.figure(figsize=(10, 6))
    
    

    for i, (path, label) in enumerate(zip(paths, labels_for_plot)):
        returns = path["price"].pct_change().dropna()
        var = np.quantile(returns, (1 - confidence_level))
        
        col_index = leverage_labels.index(label)
        color = colors[col_index]
        
        # Plot histogram
        plt.hist(
            returns,
            bins=bins,
            density=True,
            alpha=0.80,
            histtype="stepfilled",
            linewidth=2,
            color=color,
            label=f"{label} (VaR: {var:.2%})",
        )
        
        # Plot matching VaR line
        plt.axvline(var, linestyle='--', linewidth=2, color=color)

    plt.title(f"Overlaid Daily Return Distributions with {int((1-confidence_level)*100)}% VaR", fontdict=font)
    plt.xlabel("Daily Return", fontdict=font)
    plt.ylabel("Density", fontdict=font)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")




def plot_max_drawdowns(dates, filename):
    def get_mdd_per_year(lev):
        mdd_list = []
        for i in range(len(dates)-1):
            start_date = dates[i]
            end_date = dates[i+1]
            
            dataIndexSP500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
            initial_value = dataIndexSP500.iloc[0]
            path = leveragedPath(dataIndexSP500, lev, initial_value)
            
            idx_max = path["price"].idxmax()
            max_price = path.loc[idx_max, "price"]
            max_date  = path.loc[idx_max, "date"]
            
            new_path = pd.DataFrame(path)
            new_path = new_path[new_path['date'] > max_date]
            if len(new_path) != 0:
                idx_min = new_path["price"].idxmin()
                min_price = new_path.loc[idx_min, "price"]
            else:
                min_price = max_price

            mdd = (min_price - max_price ) / max_price
            
            mdd_list.append(mdd)
            
        return mdd_list

    plt.figure(figsize=(10, 5))

    x_labels = [d[:4] for d in dates[:-1]]  # years for x-axis
    x = range(len(x_labels))

    prev_mdd = np.zeros(len(x))  # start stacking from 0

    lev_plot = [1, 1.28, 2, 3, 4, 5]
    levl_labels = ['1x', 'Worst-Case', '2x', 'Mid-Case', '4x', '5x']
    colors = [
        "#000000",  # black (1x)
        "#FDAE61",  # soft orange (Worst-Case)
        "#1C3879",  # medium navy-blue (2x)
        "#911D09",   # dark red (Mid-Case)
        "#41AB5D",  # muted green (4x)
        "#A6D96A"  # light olive (5x)
    ]


    for lev, label, color in zip(lev_plot, levl_labels, colors):
        mdd_list = get_mdd_per_year(lev)
        plt.plot(x, mdd_list, marker='o', linestyle='-', color=color, label=label)
        '''
        mdd_list = np.array(get_mdd_per_year(lev))
        height = mdd_list - prev_mdd
        bottom = prev_mdd
        plt.bar(
            x,
            height,
            bottom=bottom,
            color=color,
            edgecolor="black",
            alpha=0.8,
            label=label,
        )
        prev_mdd = mdd_list
        '''
        
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks(x, x_labels)
    plt.title("Maximum Drawdown per Year for Different Leverage Ratios")
    plt.xlabel("Year")
    plt.ylabel("Max Drawdown (fraction of peak)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")
    


def plot_risk_return_metrics(filename, start_date_comparisons="2000-01-01", end_date_comparisons="2025-01-01"):
    def compute_sharpe_ratio(lev, start_date, end_date):
        # Download data
        dataIndexSP500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"]

        initial_value = dataIndexSP500.iloc[0]

        leveraged_path = leveragedPath(dataIndexSP500, lev, initial_value)

        # 1. Calculate daily returns
        daily_returns = leveraged_path["price"].pct_change().dropna()

        # 2. Define and convert the annual risk-free rate to a daily rate
        annual_risk_free_rate = 0.02
        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/252) - 1

        # 3. Calculate excess daily returns
        excess_daily_returns = daily_returns - daily_risk_free_rate

        # 4. Calculate the average of excess daily returns
        avg_excess_return = excess_daily_returns.mean()

        # 5. Calculate the standard deviation of excess daily returns (daily volatility)
        std_dev_excess_return = excess_daily_returns.std()

        # 6. Calculate the daily Sharpe Ratio
        daily_sharpe_ratio = avg_excess_return / std_dev_excess_return

        # 7. Annualize the Sharpe Ratio
        annualization_factor = np.sqrt(252) # Use 252 for daily trading data
        annualized_sharpe_ratio = daily_sharpe_ratio * annualization_factor

        #print(f"Annualized Sharpe Ratio for a {lev}x leveraged S&P500: {annualized_sharpe_ratio}")
        return annualized_sharpe_ratio


    def compute_avg_return(lev, start_date, end_date):
        data = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
        initial_value = data.iloc[0]
        investment_time_year = (data.index[-1] - data.index[0]).days / 365.25
        leveraged_path = leveragedPath(data, lev, initial_value)
        final_value = leveraged_path.iloc[-1]["price"]

        # Annualized percentage return
        percent_return = ((final_value / initial_value) ** (1 / investment_time_year) - 1) * 100
        return percent_return


    def value_at_risk(lev, confidence_level, start_date, end_date):
        data = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
        initial_value = data.iloc[0]
        leveraged_path = leveragedPath(data, lev, initial_value)
        daily_returns = leveraged_path["price"].pct_change().dropna()
        var_value = np.quantile(daily_returns, (1 - confidence_level))
        return var_value


    def get_avg_mdd(lev, dates):
        mdd_list = []
        for i in range(len(dates)-1):
            start_date = dates[i]
            end_date = dates[i+1]
            
            dataIndexSP500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"].dropna()
            initial_value = dataIndexSP500.iloc[0]
            leveraged_path = leveragedPath(dataIndexSP500, lev, initial_value)
            
            idx_max = leveraged_path["price"].idxmax()
            max_price = leveraged_path.loc[idx_max, "price"]
            max_date  = leveraged_path.loc[idx_max, "date"]
            
            new_path = pd.DataFrame(leveraged_path)
            new_path = new_path[new_path['date'] > max_date]
            if len(new_path) != 0:
                idx_min = new_path["price"].idxmin()
                min_price = new_path.loc[idx_min, "price"]
            else:
                min_price = max_price

            mdd = (min_price - max_price) / max_price
            
            mdd_list.append(mdd)
            
        mdd_avg = np.mean(mdd_list)
        return mdd_avg


    
    dates = pd.date_range(start=start_date_comparisons,
                                end=end_date_comparisons,
                                freq="YS").strftime("%Y-%m-%d").tolist()

    sharpe1x = compute_sharpe_ratio(1, start_date=start_date_comparisons, end_date=end_date_comparisons)
    sharpe128x = compute_sharpe_ratio(1.28, start_date=start_date_comparisons, end_date=end_date_comparisons)
    sharpe2x = compute_sharpe_ratio(2, start_date=start_date_comparisons, end_date=end_date_comparisons)
    sharpe3x = compute_sharpe_ratio(3, start_date=start_date_comparisons, end_date=end_date_comparisons)
    sharpe4x = compute_sharpe_ratio(4, start_date=start_date_comparisons, end_date=end_date_comparisons)
    sharpe5x = compute_sharpe_ratio(5, start_date=start_date_comparisons, end_date=end_date_comparisons)

    avg_return_1x = compute_avg_return(1, start_date=start_date_comparisons, end_date=end_date_comparisons)
    avg_return_128x = compute_avg_return(1.28, start_date=start_date_comparisons, end_date=end_date_comparisons)
    avg_return_2x = compute_avg_return(2, start_date=start_date_comparisons, end_date=end_date_comparisons)
    avg_return_3x = compute_avg_return(3, start_date=start_date_comparisons, end_date=end_date_comparisons)
    avg_return_4x = compute_avg_return(4, start_date=start_date_comparisons, end_date=end_date_comparisons)
    avg_return_5x = compute_avg_return(5, start_date=start_date_comparisons, end_date=end_date_comparisons)

    vaR_1x = value_at_risk(1, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)
    vaR_128x = value_at_risk(1.28, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)
    vaR_2x = value_at_risk(2, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)
    vaR_3x = value_at_risk(3, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)
    vaR_4x = value_at_risk(4, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)
    vaR_5x = value_at_risk(5, 0.95, start_date=start_date_comparisons, end_date=end_date_comparisons)

    avg_max_draw_1x = get_avg_mdd(1, dates)
    avg_max_draw_128x = get_avg_mdd(1.28, dates)
    avg_max_draw_2x = get_avg_mdd(2, dates)
    avg_max_draw_3x = get_avg_mdd(3, dates)
    avg_max_draw_4x = get_avg_mdd(4, dates)
    avg_max_draw_5x = get_avg_mdd(5, dates)

    sharpes = [sharpe1x, sharpe128x, sharpe2x, sharpe3x, sharpe4x, sharpe5x]
    avg_returns = [avg_return_1x, avg_return_128x, avg_return_2x, avg_return_3x, avg_return_4x, avg_return_5x]
    vaR_values = [vaR_1x, vaR_128x,  vaR_2x, vaR_3x, vaR_4x, vaR_5x]
    avg_max_drawdowns = [avg_max_draw_1x, avg_max_draw_128x, avg_max_draw_2x, avg_max_draw_3x, avg_max_draw_4x, avg_max_draw_5x]


    rows = [
        ("1x",    avg_return_1x,   avg_max_draw_1x,   vaR_1x,   sharpe1x),
        ("Worst", avg_return_128x, avg_max_draw_128x, vaR_128x, sharpe128x),
        ("2x",    avg_return_2x,   avg_max_draw_2x,   vaR_2x,   sharpe2x),
        ("3x",    avg_return_3x,   avg_max_draw_3x,   vaR_3x,   sharpe3x),
        ("4x",    avg_return_4x,   avg_max_draw_4x,   vaR_4x,   sharpe4x),
        ("5x",    avg_return_5x,   avg_max_draw_5x,   vaR_5x,   sharpe5x),
    ]

    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lccc c}")
    print(r"    \toprule")
    print(r"    Leverage & Avg Annual Return 2000--2025 & Max Drawdown & Daily VaR (5\%) & Sharpe \\")
    print(r"    \midrule")

    for lev, ret, mdd, var, sh in rows:
        ret_str = f"{ret:5.2f}\\%"
        mdd_str = f"{mdd*100:5.2f}\\%"
        var_str = f"{var*100:5.2f}\\%"
        sh_str  = f"{sh:5.2f}"
        print(f"    {lev} & {ret_str} & {mdd_str} & {var_str} & {sh_str} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Risk and return metrics for different leverage strategies.}")
    print(r"  \label{tab:risk_return_leverage}")
    print(r"\end{table}")



    # Create 4-panel figure
    x = np.arange(6)
    leverage_labels = ['1x', 'Worst-Case', '2x', 'Mid-Case', '4x', '5x']

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # ==== Plot 1: Avg Annual Return ====
    axs[0, 0].plot(x, avg_returns, marker='o', color='green')
    axs[0, 0].set_title('Average Annual Return')
    axs[0, 0].set_ylabel('Avg Annual Return (%)', color='green')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(leverage_labels, rotation=15)
    axs[0, 0].grid(True)

    # ==== Plot 2: Sharpe Ratio ====
    axs[0, 1].plot(x, sharpes, marker='s', color='purple')
    axs[0, 1].set_title('Sharpe Ratio')
    axs[0, 1].set_ylabel('Sharpe Ratio', color='purple')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(leverage_labels, rotation=15)
    axs[0, 1].grid(True)

    # ==== Plot 3: Daily VaR ====
    axs[1, 0].plot(x, vaR_values, marker='^', color='blue')
    axs[1, 0].set_title('Daily VaR (5%)')
    axs[1, 0].set_ylabel('Daily VaR (%)', color='blue')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(leverage_labels, rotation=15)
    axs[1, 0].grid(True)

    # ==== Plot 4: Max Drawdown ====
    axs[1, 1].plot(x, avg_max_drawdowns, marker='d', color='red')
    axs[1, 1].set_title('Average Max Drawdown')
    axs[1, 1].set_ylabel('Max Drawdown (%)', color='red')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(leverage_labels, rotation=15)
    axs[1, 1].grid(True)

    fig.suptitle("Risk and Return Metrics Across Leverage Strategies", fontsize=16)
    plt.tight_layout()
    #plt.show()
    filename1 = filename + "_4_Panel.pdf"
    plt.savefig(filename1, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename1}")




    # Create 3-panel figure
    x = np.arange(6)
    leverage_labels = ['1x', 'Worst-Case', '2x', 'Mid-Case', '4x', '5x']


    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # ==== Plot 1: Avg Annual Return ====
    axs[0].plot(x, avg_returns, marker='o', color='green')
    axs[0].set_title('Average Annual Return')
    axs[0].set_ylabel('Avg Annual Return (%)', color='green')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(leverage_labels, rotation=15)
    axs[0].grid(True)

    # ==== Plot 2: Max Drawdown ====
    axs[1].plot(x, avg_max_drawdowns, marker='d', color='red')
    axs[1].set_title('Average Max Drawdown')
    axs[1].set_ylabel('Max Drawdown (%)', color='red')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(leverage_labels, rotation=15)
    axs[1].grid(True)

    # ==== Plot 3: Daily VaR ====
    axs[2].plot(x, vaR_values, marker='^', color='blue')
    axs[2].set_title('Daily VaR (5%)')
    axs[2].set_ylabel('Daily VaR (%)', color='blue')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(leverage_labels, rotation=15)
    axs[2].grid(True)

    fig.suptitle("Risk and Return Metrics Across Leverage Strategies", fontsize=16)
    plt.tight_layout()
    #plt.show()
    filename2 = filename + "_3_Panel.pdf"
    plt.savefig(filename2, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename2}")
    
    
def days_to_recovery_for_leverage(lev,
                                  start_date="2005-01-01",
                                  end_date="2025-01-01",
                                  start_crisis="2020-02-01",
                                  end_crisis="2022-02-01"):
      
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

    if path_recovered.empty:
        recovery_date = None
        calendar_days = None
        trading_days = None
    else:
        recovery_date = path_recovered["date"].iloc[0]
        calendar_days = (recovery_date - max_price_before_stress_date).days
        trading_days = len(path_leveraged_just_before_stress[(path_leveraged_just_before_stress["date"] >= max_price_before_stress_date) &
                        (path_leveraged_just_before_stress["date"] <= recovery_date)]) - 1

    return {
        "leverage": lev,
        "peak_date": max_price_before_stress_date,
        "peak_price": max_price_before_stress,
        "recovery_date": recovery_date,
        "calendar_days": calendar_days,
        "trading_days": trading_days,
    }

def days_to_recovery_all_leverages(leverage_values,
                                   lev_labels,
                                   filename,
                                   start_date="2005-01-01",
                                   end_date="2025-01-01",
                                   start_crisis="2020-02-01",
                                   end_crisis="2022-02-01"
                                   ):
    
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

    recovery_df = pd.DataFrame(results)
    
    # Latex code for a table
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lcccc}")
    print(r"    \toprule")
    print(r"    Leverage & Peak Date & Peak Price & Recovery Date & Trading Days \\")
    print(r"    \midrule")

    for _, row in recovery_df.iterrows():
        lev = f"{row['leverage']:.2f}".replace(".00", "") + "x" if row['leverage']<=5 else "Worst"
        peak_date = str(row['peak_date'])[:10]
        rec_date  = str(row['recovery_date'])[:10]
        print(f"    {lev:8} & {peak_date} & {row['peak_price']:.2f} & {rec_date} & {int(row['trading_days'])} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Peak values and recovery durations for leveraged S\&P500 strategies during the 2020 Covid-19 crash.}")
    print(r"  \label{tab:covid_recovery_leverage}")
    print(r"\end{table}")

    # Plot
    x = np.arange(len(leverage_values))
    y = recovery_df['trading_days'].values

    plt.figure(figsize=(6.5,6))
    plt.plot(x, y, marker='o', linestyle='-', color='darkred')

    plt.title('Days to recover from the COVID crash by Leverage')
    plt.xlabel('Leverage Ratio', color='black')
    plt.ylabel('Trading Days to Recovery', color='black')

    plt.xticks(x, lev_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.4)

    plt.tight_layout()
    #plt.show()
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Figure saved as: {filename}")
    
    
  
  
  

# Set standart values and paths
dataIndexSP500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)["Close"]

initial_value = dataIndexSP500.iloc[0]

leveraged_index = leveragedPath(dataIndexSP500, 1, initial_value)
leveraged_2x_sp = leveragedPath(dataIndexSP500, 2, initial_value)
leveraged_3x_sp = leveragedPath(dataIndexSP500, 3, initial_value)
leveraged_4x_sp = leveragedPath(dataIndexSP500, 4, initial_value)
leveraged_5x_sp = leveragedPath(dataIndexSP500, 5, initial_value)
leveraged_worst_case = leveragedPath(dataIndexSP500, 1.28, initial_value)
leveraged_mid_case = leveragedPath(dataIndexSP500, 3, initial_value)


result_unlev = dataIndexSP500.iloc[-1]
result_2x = leveragedPath(dataIndexSP500, 2, initial_value).iloc[-1]['price']
result_3x = leveragedPath(dataIndexSP500, 3, initial_value).iloc[-1]['price']
result_4x = leveragedPath(dataIndexSP500, 4, initial_value).iloc[-1]['price']
result_5x = leveragedPath(dataIndexSP500, 5, initial_value).iloc[-1]['price']

result_worst_case = leveragedPath(dataIndexSP500, 1.28, initial_value).iloc[-1]['price']
result_mid_case = leveragedPath(dataIndexSP500, 3, initial_value).iloc[-1]['price']

values = [result_unlev, result_2x, result_3x, result_4x, result_5x, result_worst_case, result_mid_case]


# Run modules


plotpaths(start_date = "2005-01-01", case="final_values", filename="Final_Values_Leverages_2005.pdf")
plotpaths(start_date = "2005-01-01", case="all_paths", filename="Paths_All_cases_Leverages_2005.pdf")
plotpaths(start_date = "2005-01-01", case="worst_vs_mid", filename="Paths_Mid_Worst_Leverages_2005.pdf")

plotpaths(start_date = "2000-01-01", case="all_paths", filename="Paths_All_cases_Leverages_2000.pdf")
plotpaths(start_date = "2010-01-01", case="all_paths", filename="Paths_All_cases_Leverages_2010.pdf")
plotpaths(start_date = "2015-01-01", case="all_paths", filename="Paths_All_cases_Leverages_2015.pdf")
plotpaths(start_date = "2020-01-01", case="all_paths", filename="Paths_All_cases_Leverages_2020.pdf")

plot_stacked_paths_selected_leverages(multiple_dates = ["2000-01-01", "2005-01-01", "2010-01-01", "2020-01-01"], filename="Stacked_Paths_00_05_10_20.pdf")

plot_procentage_return(start_date = "2005-01-01", filename="Annualized_Return_by_Leverage_Strategy_2005.pdf")
plot_procentage_return(start_date = "2010-01-01", filename="Annualized_Return_by_Leverage_Strategy_2010.pdf")
plot_procentage_return(start_date = "2015-01-01", filename="Annualized_Return_by_Leverage_Strategy_2015.pdf")

plot_procentage_returns(start_dates = ["2000-01-01", "2005-01-01", "2010-01-01", "2020-01-01"], leverage_values_for_plot = [1, 1.28, 3], leverage_labels_for_plot=['1x', 'Worst-Case', 'Mid-Case'], filename="Annualized_Return_by_Leverage_Strategy_multiple_starts_worst_mid.pdf")
plot_procentage_returns(start_dates = ["2000-01-01", "2005-01-01", "2010-01-01", "2020-01-01"], leverage_values_for_plot = leverage_values, leverage_labels_for_plot = leverage_labels, filename="Annualized_Return_by_Leverage_Strategy_multiple_starts_all.pdf")

plotpaths(start_date = "1990-01-01", case="final_values", filename="Paths_Final_Values_Leverages_1990.pdf")
plotpaths(start_date = "2000-01-01", case="final_values", filename="Paths_Final_Values_Leverages_2000.pdf")
plotpaths(start_date = "2010-01-01", case="final_values", filename="Paths_Final_Values_Leverages_2010.pdf")
plotpaths(start_date = "2020-01-01", case="final_values", filename="Paths_Final_Values_Leverages_2020.pdf")

plot_procentage_returns_different_starts_worst_mid(start_dates = ["1990-01-01", "1995-01-01", "2000-01-01", "2005-01-01", "2010-01-01", "2020-01-01"], filename="Percentage_Returns_Mid_Worst_Multiple_Starts_General_Scenario.pdf")

plot_var_histogram(dataIndexSP500, title="Distribution of Daily Returns with VaR for the index of the ETF", filename="VaR_Histogram_Index_General_Scenario.pdf", confidence_level=0.95, bins=50)
plot_var_histogram(leveraged_worst_case["price"], title="Distribution of Daily Returns with VaR for worst-case leveraged ETF", filename="VaR_Histogram_Worst_Case_General_Scenario.pdf", confidence_level=0.95, bins=50)
plot_var_histogram(leveraged_mid_case["price"], title="Distribution of Daily Returns with VaR for mid-case leveraged ETF", filename="VaR_Histogram_Mid_Case_General_Scenario.pdf", confidence_level=0.95, bins=50)

plot_returns_with_var_threshold(leveraged_index["price"], leveraged_index["date"], 1, title="Daily Returns with Historical VaR Threshold for the index of the ETF", filename="Daily_Returns_VaR_Index_General_Scenario", confidence_level=0.95)
plot_returns_with_var_threshold(leveraged_mid_case["price"], leveraged_mid_case["date"], 3, title="Daily Returns with Historical VaR Threshold for mid-case leveraged ETF", filename="Daily_Returns_VaR_Mid_Case_General_Scenario", confidence_level=0.95)
plot_returns_with_var_threshold(leveraged_worst_case["price"], leveraged_worst_case["date"], 1.28, title="Daily Returns with Historical VaR Threshold for worst-case leveraged ETF", filename="Daily_Returns_VaR_Worst_Case_General_Scenario", confidence_level=0.95)

plot_overlaid_var_histograms(
    paths=[leveraged_index, leveraged_mid_case, leveraged_worst_case],
    labels_for_plot=["1x", "Mid-Case", "Worst-Case"],
    values_for_plot=[1, 3, 1.28],
    filename="Overlaid_VaR_Histogram_General_Scenario.pdf"
)

plot_max_drawdowns(dates = ['2007-01-01','2008-01-01','2009-01-01','2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01','2024-01-01','2025-01-01'], filename="Plot_max_drawdowns_General_Scenario.pdf")

plot_risk_return_metrics(filename="Plot_Risk_Return_Metrics")


#Stress scenario
invest_crash_date = "2020-01-15"
post_crash_date_2y5 = "2022-07-15"
post_crash_date_3y = "2023-01-15"
post_crash_date_4y = "2024-01-15"
dataIndexSP500_stress = yf.Ticker("^GSPC").history(start=invest_crash_date, end=post_crash_date_2y5)["Close"]
initial_value_stress = dataIndexSP500_stress.iloc[0]
leveraged_index_stress = leveragedPath(dataIndexSP500_stress, 1, initial_value_stress)
leveraged_2x_sp_stress = leveragedPath(dataIndexSP500_stress, 2, initial_value_stress)
leveraged_3x_sp_stress = leveragedPath(dataIndexSP500_stress, 3, initial_value_stress)
leveraged_4x_sp_stress = leveragedPath(dataIndexSP500_stress, 4, initial_value_stress)
leveraged_5x_sp_stress = leveragedPath(dataIndexSP500_stress, 5, initial_value_stress)
leveraged_worst_case_stress = leveragedPath(dataIndexSP500_stress, 1.28, initial_value_stress)
leveraged_mid_case_stress = leveragedPath(dataIndexSP500_stress, 3, initial_value_stress)


plot_procentage_returns_different_starts_worst_mid(start_dates = ["2019-07-01", "2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"], filename="Percentage_Returns_Mid_Worst_Multiple_Starts_Stress_Scenario.pdf", end_date=post_crash_date_2y5)

plot_overlaid_var_histograms(
    paths=[leveraged_index_stress, leveraged_mid_case_stress, leveraged_worst_case_stress],
    labels_for_plot=["1x", "Mid-Case", "Worst-Case"],
    values_for_plot=[1, 3, 1.28],
    filename="Overlaid_VaR_Histogram_Stress_Scenario.pdf"
)
plot_var_histogram(leveraged_index_stress["price"], title="Distribution of Daily Returns with VaR for the unleveraged ETF", filename="VaR_Histogram_Index_Stress_Scenario.pdf", confidence_level=0.95, bins=50)
plot_var_histogram(leveraged_worst_case_stress["price"], title="Distribution of Daily Returns with VaR for worst-case leveraged ETF", filename="VaR_Histogram_Worst_Case_Stress_Scenario.pdf", confidence_level=0.95, bins=50)
plot_var_histogram(leveraged_mid_case_stress["price"], title="Distribution of Daily Returns with VaR for mid-case leveraged ETF", filename="VaR_Histogram_Mid_Case_Stress_Scenario.pdf", confidence_level=0.95, bins=50)

plot_risk_return_metrics(filename="Plot_Risk_Return_Metrics_Stress_Scenario", start_date_comparisons=invest_crash_date, end_date_comparisons=post_crash_date_2y5)

plot_returns_with_var_threshold(leveraged_index_stress["price"], leveraged_index_stress["date"], 1, title="Daily Returns with Historical VaR Threshold for the index of the ETF", filename="Daily_Returns_VaR_Index_Stress_Scenario", confidence_level=0.95)
plot_returns_with_var_threshold(leveraged_mid_case_stress["price"], leveraged_mid_case_stress["date"], 3, title="Daily Returns with Historical VaR Threshold for mid-case leveraged ETF", filename="Daily_Returns_VaR_Mid_Case_Stress_Scenario", confidence_level=0.95)
plot_returns_with_var_threshold(leveraged_worst_case_stress["price"], leveraged_worst_case_stress["date"], 1.28, title="Daily Returns with Historical VaR Threshold for worst-case leveraged ETF", filename="Daily_Returns_VaR_Worst_Case_Stress_Scenario", confidence_level=0.95)
plot_max_drawdowns(dates = ['2019-01-01','2020-01-01','2021-01-01','2022-01-01'], filename="Plot_max_drawdowns_Stress_Scenario.pdf")
plotpaths(start_date = invest_crash_date, case="all_paths", filename="Paths_All_Leverages_2020_2025.pdf", end_date="2025-11-20")
plotpaths(start_date = invest_crash_date, case="final_values", filename="Paths_Final_Values_2020_2023.pdf", end_date="2023-08-01")
plotpaths(start_date = invest_crash_date, case="all_paths", filename="Paths_All_Leverages_2020_2023.pdf", end_date="2023-08-01")
plotpaths(start_date = invest_crash_date, case="all_paths", filename="Paths_All_Leverages_2020_2024.pdf", end_date="2024-08-01")

plotpaths(start_date = invest_crash_date, case="all_paths", filename="Paths_All_Leverages_2020_2022.pdf", end_date="2022-07-01")
plotpaths(start_date = invest_crash_date, case="final_values", filename="Paths_Final_Values_2020_2022.pdf", end_date="2022-07-01")
plot_procentage_return(start_date = invest_crash_date, filename="Annualized_Return_by_Leverage_Strategy_2020_2022.pdf", end_date="2022-07-01")
plot_procentage_returns(start_dates = ["2019-07-15", "2020-01-15", "2020-07-15", "2021-01-15", "2021-07-15", "2022-01-15"], leverage_values_for_plot = leverage_values, leverage_labels_for_plot = leverage_labels, filename="Percentage_Returns_All_Leverages_Multiple_Starts_Stress_Scenario.pdf", end_date=post_crash_date_2y5)
plot_procentage_returns_different_starts_worst_mid(start_dates = ["2019-07-15", "2020-01-15", "2020-07-15", "2021-01-15", "2021-07-15", "2022-01-15"], filename="Percentage_Returns_Mid_Worst_Multiple_Starts_Stress_Scenario.pdf", end_date=post_crash_date_2y5)

plot_stacked_paths_selected_leverages(multiple_dates = [invest_crash_date, "2020-07-15", "2021-01-15", "2021-07-15"], filename="Stacked_Paths_20_21_21_Some_Lev.pdf", selected_leverages=[1, 1.28, 3, 5], end_date="2023-07-15")
plot_stacked_paths_selected_leverages(multiple_dates = [invest_crash_date, "2020-07-15", "2021-01-15", "2021-07-15"], filename="Stacked_Paths_20_21_21_All_Lev.pdf", selected_leverages=[1, 1.28,2, 3, 4, 5], end_date="2023-07-15")

days_to_recovery_all_leverages(leverage_values_ordered, lev_labels=leverage_labels_ordered, filename="Plot_Days_To_Recover.pdf", start_date="2005-01-01", end_date="2025-01-01", start_crisis=invest_crash_date, end_crisis="2022-02-01")


print(f"✅ All Evaluations and Plots are finished")