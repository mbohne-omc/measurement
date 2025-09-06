import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.ticker as mticker


# mpl.rcParams['grid.color'] = "#C9F3DE"

DEFAULT_COLOR_PALETTE = [
    "#2CC076", "#262833", "#18663E", "#7EECB5",
    "#C9F3DE", "#FF7A06", "#FCC996", "#ECEE82",
    "#65C1DE", "#F55858"
]

DEFAULT_COLORS = {
    "ad_spend": "#2CC076",
    "roi": "#FF7A06",
    "grid": "#C9F3DE",
    "background": "#ECEE82",
    "text": "#262833",
}

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=DEFAULT_COLOR_PALETTE)

def get_time_metrics(weekly_prepped_data_df, 
                date_column, 
                metric_columns, 
                metric_name='metric', 
                sum_all_channels=True):
    """
    Calculate annual metrics, past year's metrics, past 90 days, and past 30 days 
    metrics from weekly prepped data.

    Parameters:
    - weekly_prepped_data_df (pd.DataFrame): DataFrame with weekly data.
    - date_column (str): Name of the date column in the DataFrame.
    - metric_columns (list): List of columns to calculate the annual metric for.
    - metric_name (str): Name of the metric for the output column (default: 'metric').
    - sum_all_channels (bool): If True, aggregates across all metric columns.

    Returns:
    - pd.DataFrame: A DataFrame with annual metrics for each year and rows for 
      the past year's, past 90 days', and past 30 days' metrics.
    """
    # Create a copy of relevant data and extract the year
    rel_df = weekly_prepped_data_df[[date_column] + metric_columns].copy()
    rel_df['year'] = rel_df[date_column].dt.year

    # Group by year and sum metrics
    annual_metrics = rel_df.groupby('year')[metric_columns].sum()

    if sum_all_channels:
        # Aggregate across all metric columns
        annual_metrics['total'] = annual_metrics.sum(axis=1)
        annual_metrics = annual_metrics[['total']]  # Keep only the total column
        annual_metrics.columns = [metric_name]
    else:
        # Add metric_name to each column
        annual_metrics.columns = [f'{col}_{metric_name}' for col in annual_metrics.columns]


    # Rename the index to 'Time Period'
    annual_metrics.index.name = 'Time Period'

    # Get the most recent date
    last_date = rel_df[date_column].max()

    # Helper function to calculate metrics for a specific period
    def calculate_period_metrics(start_date, end_date, period_label):
        period_data = rel_df[(rel_df[date_column] > start_date) & (rel_df[date_column] <= end_date)]
        period_metrics = period_data[metric_columns].sum()

        if sum_all_channels:
            # Aggregate across all metric columns
            period_metric = period_metrics.sum()
            return pd.DataFrame({'Time Period': [period_label], metric_name: [period_metric]})
        else:
            # Add metric_name to each column and add as row
            return pd.DataFrame(
                {'Time Period': [period_label], 
                 **{f'{col}_{metric_name}': [value] for col, value in period_metrics.items()}}
            )

    # Define periods
    one_year_back = last_date - pd.DateOffset(years=1)
    ninety_days_back = last_date - pd.DateOffset(days=90)
    thirty_days_back = last_date - pd.DateOffset(days=30)

    # Calculate metrics for each period
    past_year_row = calculate_period_metrics(one_year_back, last_date, 'Past Year')
    past_90_days_row = calculate_period_metrics(ninety_days_back, last_date, 'Past 90 Days')
    past_30_days_row = calculate_period_metrics(thirty_days_back, last_date, 'Past 30 Days')

    # Reset index and append all period metrics
    annual_metrics_df = annual_metrics.reset_index()
    annual_metrics_df = pd.concat(
        [annual_metrics_df, past_year_row, past_90_days_row, past_30_days_row], 
        ignore_index=True
    )

    return annual_metrics_df

def calculate_total_metrics(run_mmm, outcome_name):

    # Calculate total metrics
    total_spend_df = get_time_metrics(
        weekly_prepped_data_df=run_mmm.prepped_data_df,
        date_column=run_mmm.date_column,
        metric_columns=list(run_mmm.channel_dict.values()),
        metric_name="ad_spend",
        sum_all_channels=True
    )

    total_y_var_df = get_time_metrics(
        weekly_prepped_data_df=run_mmm.prepped_data_df,
        date_column=run_mmm.date_column,
        metric_columns=list(run_mmm.outcome_dict.values()),
        metric_name="outcome",
        sum_all_channels=True
    )

    total_contribution_df = get_time_metrics(
        weekly_prepped_data_df=run_mmm.model_output['contributions_over_time'].reset_index(),
        date_column='date',
        metric_columns=list(run_mmm.channel_dict.values()),
        metric_name="attributed_outcome",
        sum_all_channels=True
    )

    # Ensure column consistency for merging
    total_contribution_df.rename(columns={"total": "attributed_outcome"}, inplace=True)

    # Combine total-level metrics
    total_combined_df = total_spend_df.merge(
        total_y_var_df, on="Time Period", how="outer"
    ).merge(
        total_contribution_df, on="Time Period", how="outer"
    )

    # Add ROI column
    total_combined_df['roi'] = total_combined_df['attributed_outcome'] / total_combined_df['ad_spend']

    # Rearrange rows to prioritize "Past 30 Days," "Past 90 Days," "Past Year," and then years in descending order
    time_order = ["Past 30 Days", "Past 90 Days", "Past Year"]
    years = sorted(
        [x for x in total_combined_df["Time Period"].unique() if isinstance(x, int)], reverse=True
    )
    time_order += years

    total_combined_df["Time Period"] = pd.Categorical(
        total_combined_df["Time Period"],
        categories=time_order,
        ordered=True
    )
    total_combined_df = total_combined_df.sort_values("Time Period").reset_index(drop=True)

    # Rename columns
    total_combined_df.rename(
        columns={
            "ad_spend": "Ad Spend",
            "outcome": f"{outcome_name}",
            "attributed_outcome": f"Attributed {outcome_name}",
            "roi": "ROI"
        },
        inplace=True
    )

    return total_combined_df


def create_time_summary(run_mmm):
    # Collect the name
    outcome_name=list(run_mmm.outcome_dict.keys())[0]
    
    # Calculate total metrics
    total_combined_df = calculate_total_metrics(run_mmm, outcome_name)

    # Extract time order from total_combined_df
    time_order = total_combined_df["Time Period"].cat.categories

    # Calculate channel metrics
    channel_combined_df = calculate_channel_metrics(run_mmm, time_order)

    return {
        "total": total_combined_df,
        "channel": channel_combined_df
    }

def calculate_channel_metrics(run_mmm, time_order):
    # Combine channel-level metrics for spend
    channel_spend_df = get_time_metrics(
        weekly_prepped_data_df=run_mmm.prepped_data_df,
        date_column=run_mmm.date_column,
        metric_columns=list(run_mmm.channel_dict.values()),
        metric_name="ad_spend",
        sum_all_channels=False
    )

    # Combine channel-level metrics for attributed outcome
    channel_contribution_df = get_time_metrics(
        weekly_prepped_data_df=run_mmm.model_output['contributions_over_time'].reset_index(),
        date_column='date',
        metric_columns=list(run_mmm.channel_dict.values()),
        metric_name="attributed_outcome",
        sum_all_channels=False
    )

    # Calculate ROI as attributed outcome divided by spend
    roi_columns = {}
    for channel in run_mmm.channel_dict.values():
        spend_col = f"{channel}_ad_spend"
        contribution_col = f"{channel}_attributed_outcome"
        if spend_col in channel_spend_df.columns and contribution_col in channel_contribution_df.columns:
            roi_columns[channel] = channel_contribution_df[contribution_col] / channel_spend_df[spend_col]

    channel_roi_df = pd.DataFrame(roi_columns)
    channel_roi_df["Time Period"] = channel_spend_df["Time Period"]
    channel_roi_df = channel_roi_df[["Time Period"] + [col for col in roi_columns.keys()]]

    # Rename columns of contribution and ad spend to remove suffixes
    channel_contribution_df.columns = channel_contribution_df.columns.str.replace("_attributed_outcome", "")
    channel_spend_df.columns = channel_spend_df.columns.str.replace("_ad_spend", "")

    # Apply the same row ordering as total_combined_df
    for df in [channel_spend_df, channel_contribution_df, channel_roi_df]:
        df["Time Period"] = pd.Categorical(
            df["Time Period"],
            categories=time_order,
            ordered=True
        )
        df.sort_values("Time Period", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return {
        "spend": channel_spend_df,
        "attributed_outcome": channel_contribution_df,
        "roi": channel_roi_df
    }


def calculate_total_metrics_transposed(df):
    # Transpose the DataFrame
    transposed_df = df.transpose()

    # Set the first row as column names
    transposed_df.columns = transposed_df.iloc[0]  # First row as column names
    transposed_df = transposed_df[1:]  # Drop the first row as it is now column names

    # Reset the index for clarity and rename it to "Metric"
    transposed_df.reset_index(inplace=True)
    transposed_df.rename(columns={"index": "Metric"}, inplace=True)

    # Explicitly drop any index name
    transposed_df = transposed_df.rename_axis(None, axis=1)

    return transposed_df

def get_mroi(mmm, dollars_spent=100):        

    n_channels = len(mmm.channel_columns)
    spend = np.ones(n_channels) * dollars_spent
    new_spend_contributions_sample = mmm.new_spend_contributions(spend=spend, one_time = True)
    new_spend_contributions = new_spend_contributions_sample.mean(dim=['chain', 'draw', 'time_since_spend']).values.flatten()

    # Create a DataFrame with expected monetary values and the customer_id
    new_spend_contributions_df = pd.DataFrame({
        'channel': mmm.channel_columns,
        'mroi': new_spend_contributions
    })

    return new_spend_contributions_df

def plot_total_ad_spend_and_roi(total_combined_df, roi_var, plt_size=(10, 6), colors=None):
    """
    Create a two-axis plot for Ad Spend and ROI over time with automatic scaling for Ad Spend and ROI.

    Parameters:
        df (DataFrame): DataFrame containing the data with columns 'Time Period', 'Ad Spend', and 'ROI'.
        plt_size (tuple): Size of the plot.
        colors (dict or list): Custom color palette. Defaults to a predefined palette.
    """
    df = total_combined_df.copy()
    # Default colors
    if colors is None:
        colors = {
            "ad_spend": "#2CC076",  # Green
            "roi": "#262833",      # Orange
            "grid": "#18663E"      # Light green for grid
        }

    # Ensure the "Time Period" column is of type string
    df["Time Period"] = df["Time Period"].astype(str)

    # Reverse the order of Time Period for plotting
    df = df.iloc[::-1]

    # Define multiplier map
    multiplier_map = {
        1: "",
        10: "tens",
        100: "hundreds",
        1_000: "thousands",
        10_000: "ten thousands",
        100_000: "hundred thousands",
        1_000_000: "millions",
        10_000_000: "ten millions",
        100_000_000: "hundred millions",
        1_000_000_000: "billions",
    }

    # Determine the scale for Ad Spend 
    min_ad_spend = df["Ad Spend"].min()
    ad_spend_scale = 1
    for scale in multiplier_map.keys():
        if min_ad_spend >= scale:
            ad_spend_scale = scale
    ad_spend_unit = multiplier_map.get(ad_spend_scale, f"{ad_spend_scale:,}")

    multiplier_map = {
        1: "",
        10: "ten",
        100: "hundred",
        1_000: "thousand",
        10_000: "ten thousand",
        100_000: "hundred thousand",
        1_000_000: "million",
        10_000_000: "ten million",
        100_000_000: "hundred million",
        1_000_000_000: "billion",
    }

    # Determine the scale for ROI
    min_roi = df["ROI"].min()
    roi_scale = 1
    while min_roi * roi_scale < 1 and min_roi > 0:
        roi_scale *= 10

    roi_unit = multiplier_map.get(roi_scale, f"{roi_scale:,}")

    # Adjust Ad Spend and ROI values based on their scales
    df["Ad Spend (scaled)"] = df["Ad Spend"] / ad_spend_scale
    df["ROI (scaled)"] = (df["ROI"] * roi_scale).round(0)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=plt_size)

    # Bar chart for Ad Spend
    ax1.bar(df["Time Period"], df["Ad Spend (scaled)"], color=colors["ad_spend"], label="Ad Spend")
    ax1.set_ylabel(f"Ad Spend ($ in {ad_spend_unit})", color=colors["ad_spend"])
    ax1.tick_params(axis='y', labelcolor=colors["ad_spend"])

    # Tilt x-axis labels
    ax1.set_xticklabels(df["Time Period"], rotation=45)

    # Line chart for ROI
    ax2 = ax1.twinx()
    ax2.plot(df["Time Period"], df["ROI (scaled)"], color=colors["roi"], marker='o', label="ROI")
    ax2.set_ylabel(f"{roi_var} (per {roi_unit} $ Spent)", color=colors["roi"])
    ax2.tick_params(axis='y', labelcolor=colors["roi"])

    # Apply grid color
    # ax1.grid(color=colors["grid"], linestyle='--', linewidth=2.0)
    ax2.grid(False)

    # Title
    plt.title("Overall: Ad Spend and ROI", fontsize=16, fontweight='bold')
    fig.tight_layout()

    # Show the plot
    plt.show()


def plot_category_ad_spend_and_roi(ad_spend, roi, color_palette, roi_var, plt_size=(10, 6)):
    """
    Create side-by-side plots:
    - Stacked bar chart for Ad Spend by category.
    - Line chart for ROI by category with reversed time axis and y-axis scaling.

    Parameters:
    - ad_spend (pd.DataFrame): DataFrame with ad spend data, indexed by 'Time Period'.
    - roi (pd.DataFrame): DataFrame with ROI data, indexed by 'Time Period'.
    - color_palette (list): List of hex colors for customizing the charts.
    - roi_var (str): Label for ROI y-axis.

    Returns:
    - None: Displays the plots.
    """
    # Reverse the order of the Time Period for plotting
    # roi = roi.iloc[::-1]
    ad_spend = ad_spend.iloc[::-1]

    # Move values to index for analysis
    total_df = ad_spend.iloc[:, 1:].sum(axis=1)
    min_ad_spend = total_df.min()
    ad_spend.index = ad_spend['Time Period']
    ad_spend.drop(columns='Time Period', inplace=True)
    roi.index = roi['Time Period']
    roi.drop(columns='Time Period', inplace=True)

    # Guarantee float type
    ad_spend = ad_spend.astype(float)
    roi = roi.astype(float)

    # Define multiplier map
    multiplier_map = {
        1: "",
        10: "tens",
        100: "hundreds",
        1_000: "thousands",
        10_000: "ten thousands",
        100_000: "hundred thousands",
        1_000_000: "millions",
        10_000_000: "ten millions",
        100_000_000: "hundred millions",
        1_000_000_000: "billions",
    }
    
    # Determine the scale for Ad Spend
    min_ad_spend = total_df.min()
    ad_spend_scale = 1
    for scale in multiplier_map.keys():
        if min_ad_spend >= scale:
            ad_spend_scale = scale
    ad_spend_unit = multiplier_map.get(ad_spend_scale, f"{ad_spend_scale:,}")

    # Define multiplier map
    multiplier_map = {
        1: "",
        10: "ten",
        100: "hundred",
        1_000: "thousand",
        10_000: "ten thousand",
        100_000: "hundred thousand",
        1_000_000: "million",
        10_000_000: "ten million",
        100_000_000: "hundred million",
        1_000_000_000: "billion",
    }

    # Determine the scale for ROI
    min_roi = roi.min().min()
    roi_scale = 1
    while min_roi * roi_scale < 1 and min_roi > 0:
        roi_scale *= 10

    roi_unit = multiplier_map.get(roi_scale, f"{roi_scale:,}")

    # Adjust Ad Spend and ROI values based on their scales
    scaled_ad_spend = ad_spend / ad_spend_scale
    scaled_roi = (roi * roi_scale).round(0)

    # Convert index to string explicitly
    scaled_ad_spend.index = scaled_ad_spend.index.astype(str)
    scaled_roi.index = scaled_roi.index.astype(str)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=plt_size)

    # Set overall title
    fig.suptitle("Channel: Ad Spend and ROI", fontsize=16, fontweight='bold')

    # Stacked bar chart for Ad Spend
    bars = scaled_ad_spend.plot(kind='bar', stacked=True, color=color_palette, ax=axs[0], legend = False)
    axs[0].set_title("Ad Spend")
    axs[0].set_ylabel(f"Ad Spend (in {ad_spend_unit} $)")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_xlabel("")

    # Line chart for ROI
    lines = []
    for column, color in zip(scaled_roi.columns, color_palette):
        line, = axs[1].plot(
            scaled_roi.index, 
            scaled_roi[column], 
            label=column, 
            color=color, 
            marker='o',
        )
        lines.append(line)
    axs[1].set_title("ROI")
    axs[1].set_ylabel(f"{roi_var} (per {roi_unit} $ Spent)")
    axs[1].set_xticks(np.arange(len(scaled_roi.index)))
    axs[1].set_xticklabels(scaled_roi.index, rotation=45)
    axs[1].invert_xaxis()  # Reverse the x-axis
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Unified legend below the title but above the plots
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in color_palette]
    labels = scaled_ad_spend.columns.tolist()  # Assumes both ad spend and ROI have the same labels
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        ncol=len(color_palette), 
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        title=""
    )

    # Adjust layout and show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the legend and title
    plt.show()

def plot_dual_axis_bar_chart(data, 
                             percent_columns, 
                             secondary_column, 
                             colors=['#2CC076', '#262833'], 
                             series_labels=None,
                             title="Dual Axis Bar Chart", 
                             ylabel1="Percentage", 
                             ylabel2="Secondary Axis",
                             figsize = (10, 6)):
    """
    Plots a bar chart with two columns side by side for percentages and a third column on a secondary y-axis.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        percent_columns (list): List of two column names for the side-by-side percentages.
        secondary_column (str): Column name for the secondary y-axis.
        colors (list): List of colors for the bars. Defaults to Matplotlib's default colors.
        series_labels (list): List of labels for the percentage columns. Defaults to column names if None.
        title (str): Title of the plot.
        ylabel1 (str): Label for the primary y-axis.
        ylabel2 (str): Label for the secondary y-axis.
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']  # Default colors for the percentage bars

    if series_labels is None:
        series_labels = percent_columns

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)

    # Add a secondary axis
    ax2 = ax1.twinx()

    # Bar width and positions
    bar_width = 0.4
    x = range(len(data))

    # Plot percentage columns as side-by-side bars
    bars1 = ax1.bar(
        [i - bar_width / 2 for i in x],
        data[percent_columns[0]],
        width=bar_width,
        label=series_labels[0],
        color=colors[0],
    )
    bars2 = ax1.bar(
        [i + bar_width / 2 for i in x],
        data[percent_columns[1]],
        width=bar_width,
        label=series_labels[1],
        color=colors[1],
    )

    # Plot secondary column with enhanced visibility
    ax2.plot(x, data[secondary_column], color='none', marker='o', linestyle='-', linewidth=0, label=secondary_column)

    # Set labels, titles, and legends
    fig.suptitle(title, fontsize=16, y=0.95)  # Title at the very top
    
    ax1.set_ylabel(ylabel1, fontsize=12)
    ax2.set_ylabel(ylabel2, fontsize=12)

    # Set x-axis labels to the index of the data
    ax1.set_xticks(x)
    ax1.set_xticklabels(data.index, rotation=45)

    # Format the secondary y-axis as dollars
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

    # Unified legend below the title but above the plots
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    handles.append(plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=8, label=secondary_column))
    labels = series_labels
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        ncol=len(handles), 
        bbox_to_anchor=(0.5, 0.93),
        frameon=False
    )

    # Adjust layout
    # ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the title and legend
    plt.show()


def create_barbell_chart(
    df,
    col1,
    col2,
    reference_col=None,
    labels=None,
    title="Barbell Chart",
    xlabel="Value",
    ylabel=None,
    legend_labels=("Series 1", "Series 2", "Reference"),
    output_path=None,
):
    """
    Creates a barbell chart from a DataFrame for two columns with an optional reference column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col1 (str): Name of the first column (e.g., "initial_percent").
        col2 (str): Name of the second column (e.g., "budget_1_percent").
        reference_col (str, optional): Name of the reference column (e.g., "initial_total").
        labels (list, optional): Custom labels for the y-axis. If None, the index is used.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis (set to None to remove).
        legend_labels (tuple): Labels for the two series and the reference in the legend.
        output_path (str): Path to save the chart as an image (optional).

    Returns:
        None
    """
    # Sort the DataFrame by col2 in descending order
    df = df.sort_values(by=col2, ascending=True)

    # Labels for y-axis
    if labels is None:
        labels = df.index.astype(str)

    # Extract values
    values_col1 = df[col1]
    values_col2 = df[col2]
    reference_values = df[reference_col] if reference_col else None

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.8))

    # Adjust layout to create space for title, legend, and plot
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Move the title to the very top and add more space from the legend
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    # Draw barbell lines
    for i, (val1, val2) in enumerate(zip(values_col1, values_col2)):
        ax.plot([val1, val2], [i, i], color="gray", linewidth=2, alpha=0.7)

    # Plot points for the two columns
    ax.scatter(values_col1, range(len(df)), color="blue", label=legend_labels[0], zorder=3)
    ax.scatter(values_col2, range(len(df)), color="green", label=legend_labels[1], zorder=3)

    # Add a hollow red point in the legend for the reference column
    ax.scatter([], [], edgecolor="red", facecolor="none", label=legend_labels[2], zorder=3)

    # Add reference text annotations with red background and white text
    if reference_col:
        for i, ref in enumerate(reference_values):
            formatted_ref = f"${ref:,.0f}"  # Format as currency with no decimals and commas
            ax.text(
                103,  # Position slightly beyond max x-axis range
                i,
                formatted_ref,
                va="center",
                ha="center",
                fontsize=10,
                color="white",
                bbox=dict(facecolor="red", edgecolor="none", boxstyle="round,pad=0.3"),
            )

    # Add y-axis labels (categories)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if ylabel:  # Only add y-axis label if provided
        ax.set_ylabel(ylabel)

    # Add legend below the title with extra space
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)

    # Modify legend text background and text color for the reference column
    legend_texts = legend.get_texts()
    for i, text in enumerate(legend_texts):
        if i == 2:  # Reference column is the third legend label
            text.set_backgroundcolor("red")  # Red background
            text.set_color("white")  # White text

    # Hide unnecessary chart elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Set x-axis range
    ax.set_xlim(0, 100)

    # Save the chart if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)

    # Show the plot
    plt.show()

def save_image(img, filepath):
    """Save a PIL image to a specified file path."""
    img.save(filepath, format='PNG')

def plot_category_ad_spend_and_roi(ad_spend, roi, roi_var, color_palette = DEFAULT_COLOR_PALETTE, plt_size=(10, 6)):
    """
    Create side-by-side plots:
    - Stacked bar chart for Ad Spend by category.
    - Line chart for ROI by category with reversed time axis and y-axis scaling.

    Parameters:
    - ad_spend (pd.DataFrame): DataFrame with ad spend data, indexed by 'Time Period'.
    - roi (pd.DataFrame): DataFrame with ROI data, indexed by 'Time Period'.
    - color_palette (list): List of hex colors for customizing the charts.
    - roi_var (str): Label for ROI y-axis.

    Returns:
    - None: Displays the plots.
    """
    # Reverse the order of the Time Period for plotting
    # roi = roi.iloc[::-1]
    ad_spend = ad_spend.iloc[::-1]

    # Move values to index for analysis
    total_df = ad_spend.iloc[:, 1:].sum(axis=1)
    min_ad_spend = total_df.min()
    ad_spend.index = ad_spend['Time Period']
    ad_spend.drop(columns='Time Period', inplace=True)
    roi.index = roi['Time Period']
    roi.drop(columns='Time Period', inplace=True)

    # Guarantee float type
    ad_spend = ad_spend.astype(float)
    roi = roi.astype(float)

    # Define multiplier map
    multiplier_map = {
        1: "",
        10: "tens",
        100: "hundreds",
        1_000: "thousands",
        10_000: "ten thousands",
        100_000: "hundred thousands",
        1_000_000: "millions",
        10_000_000: "ten millions",
        100_000_000: "hundred millions",
        1_000_000_000: "billions",
    }
    
    # Determine the scale for Ad Spend
    min_ad_spend = total_df.min()
    ad_spend_scale = 1
    for scale in multiplier_map.keys():
        if min_ad_spend >= scale:
            ad_spend_scale = scale
    ad_spend_unit = multiplier_map.get(ad_spend_scale, f"{ad_spend_scale:,}")

    # Define multiplier map
    multiplier_map = {
        1: "",
        10: "ten",
        100: "hundred",
        1_000: "thousand",
        10_000: "ten thousand",
        100_000: "hundred thousand",
        1_000_000: "million",
        10_000_000: "ten million",
        100_000_000: "hundred million",
        1_000_000_000: "billion",
    }

    # Determine the scale for ROI
    min_roi = roi.min().min()
    roi_scale = 1
    while min_roi * roi_scale < 1 and min_roi > 0:
        roi_scale *= 10

    roi_unit = multiplier_map.get(roi_scale, f"{roi_scale:,}")

    # Adjust Ad Spend and ROI values based on their scales
    scaled_ad_spend = ad_spend / ad_spend_scale
    scaled_roi = (roi * roi_scale).round(0)

    # Convert index to string explicitly
    scaled_ad_spend.index = scaled_ad_spend.index.astype(str)
    scaled_roi.index = scaled_roi.index.astype(str)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=plt_size)

    # Set overall title
    fig.suptitle("Channel: Ad Spend and ROI", fontsize=16, fontweight='bold')

    # Stacked bar chart for Ad Spend
    bars = scaled_ad_spend.plot(kind='bar', stacked=True, color=color_palette, ax=axs[0], legend = False)
    axs[0].set_title("Ad Spend")
    axs[0].set_ylabel(f"Ad Spend (in {ad_spend_unit} $)")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_xlabel("")

    # Line chart for ROI
    lines = []
    for column, color in zip(scaled_roi.columns, color_palette):
        line, = axs[1].plot(
            scaled_roi.index, 
            scaled_roi[column], 
            label=column, 
            color=color, 
            marker='o',
        )
        lines.append(line)
    axs[1].set_title("ROI")
    axs[1].set_ylabel(f"{roi_var} (per {roi_unit} $ Spent)")
    axs[1].set_xticks(np.arange(len(scaled_roi.index)))
    axs[1].set_xticklabels(scaled_roi.index, rotation=45)
    axs[1].invert_xaxis()  # Reverse the x-axis
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Unified legend below the title but above the plots
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in color_palette]
    labels = scaled_ad_spend.columns.tolist()  # Assumes both ad spend and ROI have the same labels
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        ncol=len(color_palette), 
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        title=""
    )

    # Adjust layout and show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the legend and title
    plt.show()

def plot_mroi(df, 
              x_column="channel", 
              y_column="mroi", 
              variable_name="Leads", 
              colors=None, 
              plot_size=(8, 5), 
              dollars_spent=100, 
              decimal_places=1, 
              prefix_dollar=False):
    """
    Plots a line graph of the given dataframe with customized title and colors.

    Parameters:
    df (pd.DataFrame): The input dataframe containing at least two columns.
    x_column (str): The column name to be used for the X-axis (default: 'channel').
    y_column (str): The column name to be used for the Y-axis (default: 'mroi').
    variable_name (str): The name of the variable being analyzed (default: 'Leads').
    colors (list): List of colors for the line plot (default: None, uses a default color).
    plot_size (tuple): Tuple specifying the figure size (width, height) in inches (default: (8, 5)).
    dollars_spent (int or float): Dollar amount used in the title (default: 100).
    decimal_places (int): Number of decimal places for the annotation values (default: 1).
    prefix_dollar (bool): Whether to prefix annotation values with '$' (default: False).
    """
    df = df.copy()
    df[y_column] = df[y_column] * 10

    # Define title
    title = f"{variable_name} per Marginal ${dollars_spent:,.0f} Spent"

    # Default color if none provided
    if colors is None:
        colors = ['#2A9D8F']

    # Plotting
    plt.figure(figsize=plot_size)
    plt.plot(df[x_column], df[y_column], marker='o', linestyle='-', color=colors[0])
    plt.title(title)
    plt.ylabel(f"{variable_name} (per ${dollars_spent:,.0f})")

    # Tilt X-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add values as annotations
    for x, y in zip(df[x_column], df[y_column]):
        label = f"{y:.{decimal_places}f}"
        if prefix_dollar:
            label = f"${label}"
        plt.text(x, y + max(df[y_column]) * 0.02, label, ha='center', fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def extract_valid_metrics_and_roas(data: dict) -> dict:
    """
    Extracts ROAS dictionary and valid summary metric means from MMM output data.

    Parameters:
    -----------
    data : dict
        A dictionary with MMM model results keyed by integer or identifier.

    Returns:
    --------
    dict
        A JSON-style dictionary with the same keys as the input,
        each mapping to a dict with:
            - 'mean_roas_dict'
            - each valid summary metric (e.g., 'rmse', 'mae', 'mape') as its own key
    """
    results = {}

    for key, model_data in data.items():
        entry = {}

        # Extract ROAS
        entry['mean_roas_dict'] = model_data.get('mean_roas_dict', {})

        # Extract valid summary metric means with full precision
        summary_metrics = model_data.get('summary_metrics', {})
        for metric, stats in summary_metrics.items():
            mean = stats.get('mean', None)
            if mean is not None and np.isfinite(mean):
                entry[metric] = float(mean)  # Explicitly cast to float for full-precision consistency

        results[key] = entry

    return results

def get_mroi(mmm, dollars_spent=100):        

    n_channels = len(mmm.channel_columns)
    spend = np.ones(n_channels) * dollars_spent
    new_spend_contributions_sample = mmm.new_spend_contributions(spend=spend, one_time = True)
    new_spend_contributions = new_spend_contributions_sample.mean(dim=['chain', 'draw', 'time_since_spend']).values.flatten()

    # Create a DataFrame with expected monetary values and the customer_id
    new_spend_contributions_df = pd.DataFrame({
        'channel': mmm.channel_columns,
        'mroi': new_spend_contributions
    })

    return new_spend_contributions_df
