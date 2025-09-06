import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL import Image

def plot_time_series(df, date_col, ncols=3):
    """Plot all time series in a DataFrame, display the plot, and return the plot as a PIL image."""
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Select only numeric columns (ignoring non-numeric columns)
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns

    # Set up the number of rows and columns for subplots
    num_plots = len(numeric_cols)
    num_cols = ncols  # Number of columns for subplots
    num_rows = (num_plots + num_cols - 1) // num_cols  # Adjusted to correctly calculate rows

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))

    # Flatten the axes array to simplify indexing when plotting
    axes = axes.flatten()

    # Plot each numeric column as a time series
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df[date_col], df[col])  # Removed the marker argument
        axes[i].set_title(f'Time Series of {col}')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col)
        axes[i].grid(True)

        # Rotate x-axis labels to be vertical
        for label in axes[i].get_xticklabels():
            label.set_rotation(90)

    # Remove any unused axes if there are fewer plots than axes
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Convert the buffer to a PIL image
    img = Image.open(buf)
    buf.close()

    return img


def read_csv(
    filename,
    date_type='daily',
    date_col="date",
    year_col=None,
    week_col=None,
    week_end_col=None,     # NEW: optional "Week Ending" column
    range_col=None         # NEW: optional "Date" range column like "m/d/yyyy - m/d/yyyy"
):
    """
    Read CSV and return daily, weekly, and monthly aggregations.

    Supports:
      - daily rows (date_col holds a day)
      - weekly rows from:
          a) (year_col, week_col), OR
          b) start date + optional week_end_col, OR
          c) a single range string column like "12/26/2022 - 1/1/2023"
      - monthly rows (date_col must be month start)

    For weekly and monthly inputs, values are distributed evenly across days in the period.

    Returns a dict with original (dailyized), daily_aggregate, weekly_aggregate, monthly_aggregate.
    """
    df = pd.read_csv(filename)

    # Normalize column names a bit (strip whitespace); keep originals otherwise
    df.columns = [c.strip() for c in df.columns]

    # Drop rows with missing date_col/range anchors
    if date_type == 'weekly' and range_col:
        df = df[df[range_col].notnull()].copy()
    else:
        df = df[df[date_col].notnull()].copy()

    # Fill missing values in numeric columns with 0
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)

    # ---------- WEEKLY ----------
    if date_type == 'weekly':
        # Case A: year/week numbers provided
        if year_col and week_col:
            df['start_date'] = df.apply(
                lambda row: datetime.strptime(
                    f"{int(row[year_col])}-W{int(row[week_col])}-1", "%Y-W%W-%w"
                ),
                axis=1
            )
            df['end_date'] = df['start_date'] + pd.Timedelta(days=6)

        # Case B: a single range string like "m/d/yyyy - m/d/yyyy"
        elif range_col and range_col in df.columns:
            def _parse_range(s):
                # Split on '-' with optional spaces; keep robustness for various formats
                parts = re.split(r'\s*-\s*', str(s))
                if len(parts) != 2:
                    return pd.NaT, pd.NaT
                s0, s1 = parts[0].strip(), parts[1].strip()
                # Try common US formats first
                for fmt in ("%m/%d/%Y", "%m/%d/%y"):
                    try:
                        d0 = datetime.strptime(s0, fmt)
                        d1 = datetime.strptime(s1, fmt)
                        return d0, d1
                    except ValueError:
                        pass
                # Fallback to pandas parser
                try:
                    d0 = pd.to_datetime(s0, errors='coerce')
                    d1 = pd.to_datetime(s1, errors='coerce')
                    return d0.to_pydatetime() if pd.notna(d0) else pd.NaT, \
                           d1.to_pydatetime() if pd.notna(d1) else pd.NaT
                except Exception:
                    return pd.NaT, pd.NaT

            parsed = df[range_col].apply(_parse_range)
            df['start_date'] = parsed.apply(lambda t: t[0])
            df['end_date']   = parsed.apply(lambda t: t[1])

        # Case C: explicit start (date_col) and optional week_end_col
        else:
            # Parse start date
            df['start_date'] = pd.to_datetime(df[date_col], errors='coerce')
            if week_end_col and week_end_col in df.columns:
                df['end_date'] = pd.to_datetime(df[week_end_col], errors='coerce')
            else:
                df['end_date'] = df['start_date'] + pd.Timedelta(days=6)

        # Drop invalid rows
        df = df[df['start_date'].notnull() & df['end_date'].notnull()].copy()

        # Compute inclusive length of the week in days
        df['period_days'] = (df['end_date'] - df['start_date']).dt.days + 1
        df = df[df['period_days'] > 0].copy()

        # Identify numeric metric columns to distribute (exclude helper cols)
        exclude = {'start_date', 'end_date', 'period_days', date_col, year_col, week_col, week_end_col, range_col}
        exclude = {c for c in exclude if c in df.columns}
        metric_cols = df.select_dtypes(include='number').columns.difference(list(exclude))

        # Distribute weekly totals evenly across the actual number of days for that week
        for col in metric_cols:
            df[col] = df[col] / df['period_days']

        # Expand to daily rows
        daily_rows = []
        for _, row in df.iterrows():
            start = pd.Timestamp(row['start_date'])
            for k in range(int(row['period_days'])):
                new_row = row.copy()
                new_row['date'] = start + pd.Timedelta(days=k)
                daily_rows.append(new_row)

        df = pd.DataFrame(daily_rows)

    # ---------- MONTHLY ----------
    elif date_type == 'monthly':
        # Parse and validate that date_col marks month starts
        df['month_start'] = pd.to_datetime(df[date_col], errors='coerce', format="%m/%d/%y")
        # Fallback if year is 4 digits
        mask_bad = df['month_start'].isna()
        if mask_bad.any():
            df.loc[mask_bad, 'month_start'] = pd.to_datetime(df.loc[mask_bad, date_col], errors='coerce', format="%m/%d/%Y")

        df = df[df['month_start'].notnull()].copy()
        df['year']  = df['month_start'].dt.year
        df['month'] = df['month_start'].dt.month
        df['days_in_month'] = df['month_start'].dt.daysinmonth

        # Validate: rows should represent month totals (month_start should be day=1)
        if not (df['month_start'].dt.day == 1).all():
            raise ValueError(
                "For monthly data, each row must be a month total with date_col on the 1st of the month. "
                "Your file looks weekly (e.g., 12/26/22). Use date_type='weekly' instead."
            )

        # Distribute month totals evenly across days in that month
        exclude_cols = {'year', 'month', 'days_in_month'}
        metric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)
        for col in metric_cols:
            df[col] = df[col] / df['days_in_month']

        # Expand monthly data into daily rows (true days in month)
        daily_rows = []
        for _, row in df.iterrows():
            year  = int(row['year'])
            month = int(row['month'])
            days  = int(row['days_in_month'])
            for day in range(1, days + 1):
                new_row = row.copy()
                new_row['date'] = datetime(year, month, day)
                daily_rows.append(new_row)

        df = pd.DataFrame(daily_rows)

    # ---------- DAILY ----------
    else:  # 'daily'
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['date'].notnull()].copy()

    # Drop helper columns if present
    df.drop(columns=[c for c in ['month_start','year','month','days_in_month','start_date','end_date','period_days'] if c in df.columns],
            inplace=True, errors='ignore')

    # Standardize types (user-defined)
    df = standardize_data_types(df, 'date')

    # Analyze time series (user-defined)
    daily_df, weekly_df = analyze_time_series(df=df, date_col='date')

    # Monthly aggregation from dailyized df
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    monthly_df = df.groupby('month').sum(numeric_only=True).reset_index()

    # Cleanup
    df.drop(columns=['month'], errors='ignore', inplace=True)
    daily_df.drop(columns=['week'], errors='ignore', inplace=True)

    return {
        "original": df,                # dailyized
        "daily_aggregate": daily_df,
        "weekly_aggregate": weekly_df,
        "monthly_aggregate": monthly_df,
    }


def standardize_data_types(df, date_col):
    """Standardize the date column to datetime and other columns to float."""
    # Convert 'date_col' to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Convert other columns to float, safely coercing errors to NaN
    for col in df.columns:
        if col != date_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# def standardize_data_types(df, date_col):
#     """Standardize the date column to datetime and other columns to float."""
#     # Convert 'date_col' to datetime
#     df[date_col] = pd.to_datetime(df[date_col])

#     # Convert other columns to float, except for the date column
#     df = df.apply(lambda col: col.astype(float) if col.name != date_col else col)

#     return df

def analyze_time_series(df, date_col="date"):
    """Analyze the DataFrame to determine daily and weekly aggregates, dropping any partial weeks."""
    # Ensure the 'date_col' is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort the DataFrame by date
    df = df.sort_values(by=date_col)

    # Check if the DataFrame has daily data by calculating the differences
    date_diffs = df[date_col].diff().dt.days
    is_daily = date_diffs.iloc[1:].eq(1).all()  # Ignore the first date

    # Initialize aggregates
    daily_aggregate = df if is_daily else None

    print('Original data passed was daily.' if is_daily else 'Original data passed was not daily.')

    # Add a 'week' column representing the start date of each week (Monday)
    df['week'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)

    # Drop partial weeks (weeks with fewer than 7 days)
    if is_daily:
    
        # If data is daily, filter out weeks that are not complete
        df = df.groupby('week').filter(lambda x: len(x) == 7)

    # Group by 'week', summing only numeric columns
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    weekly_aggregate = df.groupby('week')[numeric_cols].sum().reset_index()

    # Return the original (daily) data and the weekly aggregate
    return daily_aggregate, weekly_aggregate

# def analyze_time_series(df, date_col="date"):
#     """Analyze the DataFrame to determine daily and weekly aggregates."""
#     # Ensure the 'date_col' is in datetime format
#     df[date_col] = pd.to_datetime(df[date_col])

#     # Sort the DataFrame by date
#     df = df.sort_values(by=date_col)

#     # Check if the DataFrame has daily data by calculating the differences
#     date_diffs = df[date_col].diff().dt.days
#     is_daily = date_diffs.iloc[1:].eq(1).all()  # Ignore the first date

#     # Check if the DataFrame has weekly data
#     is_weekly = date_diffs.iloc[1:].mod(7).eq(0).all()

#     # Initialize aggregates
#     daily_aggregate = df if is_daily else None
#     weekly_aggregate = None

#     if is_weekly:
#         print('Original data passed was weekly.')
#         # If data is weekly, use it as the weekly aggregate
#         df['week'] = df[date_col]
#         weekly_aggregate = df
#     else:
#         # Resample the data to weekly frequency if the data is daily
#         print('Original data passed was daily.')

#         # Add a 'week' column representing the start date of each week
#         df['week'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)

#         # Group by 'week', summing only numeric columns
#         numeric_cols = df.select_dtypes(include=['float', 'int']).columns
#         weekly_aggregate = df.groupby('week')[numeric_cols].sum().reset_index()

#     # Return the original (daily) data and the weekly aggregate
#     return daily_aggregate, weekly_aggregate

def simulate_channel_spend(spend_data, modeled_channels, simulated_weekly_spend_mean, simulated_weekly_spend_noise):
    total_spend = spend_data[modeled_channels].sum(axis=1)

    # Simulate base spend
    n_periods = len(total_spend)  # Number of months or periods
    base_spend = np.full(n_periods, simulated_weekly_spend_mean)

    # Add noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(loc=0, scale=simulated_weekly_spend_noise, size=n_periods) 

    # Add noise to the base spend
    spend_with_noise = base_spend + noise

    # Floor the spend at 0
    spend_with_noise = spend_with_noise.clip(min=0)

    # Adjust the average back to 250
    adjusted_spend = spend_with_noise * (simulated_weekly_spend_mean / spend_with_noise.mean())

    return adjusted_spend