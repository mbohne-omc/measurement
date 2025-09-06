import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymc_marketing.mmm import MMM
import glob
import os
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100

SEED = 42

def read_saved_model(dir, client):

    load_path = dir + client + '/'

    # Get a list of all models in the directory with the pattern 'model_*.nc'
    model_files = glob.glob(os.path.join(load_path, "model_*.nc"))

    # Sort the files by modification date (newest first) and get the most recent one
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Most recent model: {latest_model}")
        return MMM.load(latest_model)
    else:
        print("No models found in the directory.")


def plot_side_by_side_bar(df):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Set the index to be the channels
    channels = df.index
    
    # Set bar width
    bar_width = 0.35

    # Create x locations for the bars
    r1 = range(len(channels))  # Initial
    r2 = [x + bar_width for x in r1]  # Optimized

    # Create the bars
    plt.bar(r1, df['initial'], width=bar_width, color='blue', edgecolor='grey', label='Initial Spend')
    plt.bar(r2, df['optimized'], width=bar_width, color='orange', edgecolor='grey', label='Optimized Spend')

    # Add labels
    plt.xlabel('Channels', fontweight='bold', fontsize=14)
    plt.xticks([r + bar_width / 2 for r in range(len(channels))], channels)  # Center the ticks
    plt.ylabel('Spend Amount', fontweight='bold', fontsize=14)
    plt.title('Initial vs Optimized Spend by Channel', fontweight='bold', fontsize=16)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    return plt

def plot_before_after(
    before,
    after,
    before_label="Before",
    after_label="After",
    colors=None,
    y_axis_name="Value",
    kpi_type="leads",
    period="weekly",
    plot_size=(6, 4),
):
    """
    Plots a bar chart comparing two values ('before' and 'after'), with formatted titles and axis labels.

    Parameters:
    - before (float): The value before the change.
    - after (float): The value after the change.
    - before_label (str): Label for the 'before' bar.
    - after_label (str): Label for the 'after' bar.
    - colors (list): List of two colors for the bars.
    - y_axis_name (str): Label for the Y-axis.
    - kpi_type (str): KPI type ('leads', 'revenue', 'new customers').
    - period (str): Time granularity ('weekly', 'monthly', 'quarterly', 'annually').
    - plot_size (tuple): Figure size (width, height).
    """

    values = [before, after]
    labels = [before_label, after_label]

    # Default colors
    if colors is None:
        colors = ["#2CC076", "#262833"]

    # Title formatting
    title = f"Expected {period.title()} {kpi_type.title()} After Optimization"

    # Y-axis label formatting
    plt.figure(figsize=plot_size)
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel(f"{period.title()} {y_axis_name}")
    plt.title(title)

    # Set y-axis limit
    plt.ylim(0, max(values) * 1.2)

    # Format y-axis tick labels
    ax = plt.gca()
    if kpi_type.lower() == "revenue":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Formatting values on top of bars
    for bar, val in zip(bars, values):
        if kpi_type.lower() == "revenue":
            val_formatted = f"${val:,.0f}"
        else:
            val_formatted = f"{val:,.0f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            val_formatted,
            ha="center",
            fontweight="bold",
        )

    plt.show()

def run_optimization_workflow(mmm, spend_data, channel_dict, n_weeks=4, budget_levels=[1], threshold=0, max_attempts=5):
    """
    Runs the complete MMM optimization workflow including:
    - Budget optimization
    - Checking and rerunning optimization until improvement is achieved

    Parameters:
    - mmm: MMM model object
    - spend_data: DataFrame containing spend data
    - channel_dict: Dictionary mapping original channel names to model channels
    - n_weeks: Number of weeks for optimization
    - budget_levels: List of budget levels for optimization
    - threshold: Minimum acceptable improvement threshold
    - max_attempts: Max number of optimization attempts

    Returns:
    - Dictionary containing final responses and merged spend DataFrame
    """

    optimizer = OptimizeSpend(mmm=mmm, spend_data=spend_data.rename(columns=channel_dict), n_weeks=n_weeks)

    # Run initial budget optimization
    optimized_budgets_df = optimizer.run_budget_optimization(
        budget_levels=budget_levels,
        n_weeks=n_weeks,
        budget_bounds=optimizer.budget_bounds
    )

    # Merge spend DataFrames
    spend_df = optimized_budgets_df['spend_percent'].merge(
        optimized_budgets_df['spend_total'], 
        left_index=True, 
        right_index=True, 
        suffixes=('_percent', '_total')
    )

    # Check and rerun optimization if necessary
    response_initial_sum, response_optimized_sum, response_delta_sum = check_and_rerun_optimization(
        optimizer, threshold=threshold, max_attempts=max_attempts
    )

    return {
        "response_initial_sum": response_initial_sum,
        "response_optimized_sum": response_optimized_sum,
        "response_delta_sum": response_delta_sum,
        "spend_df": spend_df
    }

def check_and_rerun_optimization(optimizer, threshold=0, max_attempts=5):
    """
    Checks if the initial response is greater than the optimized response.
    If so, reruns the optimization iteratively until the optimized response exceeds the initial
    or the maximum attempts are reached.

    Parameters:
    - optimizer (OptimizeSpend): The optimization object.
    - threshold (float): The threshold for rerunning optimization.
    - max_attempts (int): Maximum number of optimization rerun attempts.

    Returns:
    - tuple: Sums of responses (response_initial_sum, response_optimized_sum, response_delta_sum).
    """
    attempts = 0

    while attempts < max_attempts:
        response_initial, response_optimized, response_delta = optimizer.response_comp(n_weeks=optimizer.n_weeks)

        response_initial_sum = response_initial.sum()
        response_optimized_sum = response_optimized.sum()
        response_delta_sum = response_delta.sum()

        if response_initial_sum <= response_optimized_sum + threshold:
            print(f"Optimized response is satisfactory after {attempts + 1} attempt(s).")
            break

        print(f"Attempt {attempts + 1}: Initial response ({response_initial_sum}) higher than optimized ({response_optimized_sum}). Rerunning optimization...")

        optimizer.allocate_budget(
            total_budget=optimizer.total_budget,
            n_weeks=optimizer.n_weeks,
            budget_bounds=optimizer.budget_bounds
        )

        attempts += 1

    if attempts == max_attempts:
        print("Maximum optimization attempts reached.")

    return response_initial_sum, response_optimized_sum, response_delta_sum

class OptimizeSpend:
    def __init__(self, 
                 mmm,
                 spend_data,
                 n_weeks = 12,
                 total_budget = None,
                 budget_bounds_prop_change = None
                 ):
        
        self.mmm = mmm
        self.n_weeks = n_weeks
        self.spend_date_raw = spend_data[mmm.channel_columns]

        # Last n weeks ad spend data
        spend_data_t = self.spend_date_raw.iloc[-self.n_weeks:]
        self.total_budget_by_channel = spend_data_t.sum(axis=0)
        
        # Initial budget per channel as dictionary
        self.initial_budget_dict = self.total_budget_by_channel.to_dict()

        # Set the budget based on history if not present
        if total_budget is None:
            self.total_budget = spend_data_t.sum(axis=0).sum().round(2)
        else:
            self.total_budget = total_budget
            
        print(f"Total budget for optimization: {self.total_budget}")

        # Set boundaries for budget optimization dynamically
        # if budget_bounds_prop_change is None:
        #     self.budget_bounds = {
        #         channel: [0, self.total_budget] for channel in self.initial_budget_dict.keys()
        #     }
        # else:
        #     # Create a new dictionary with adjusted bounds
        #     self.budget_bounds = {
        #         channel: [
        #             value - value * budget_bounds_prop_change,  # Lower bound (x% less)
        #             value + value * budget_bounds_prop_change   # Upper bound (x% more)
        #         ]
        #         for channel, value in self.total_budget_by_channel.items()
        #     }

        if budget_bounds_prop_change is None:
            budget_bounds_prop_change = 1  # try 50% flex around history

        self.budget_bounds = {
            channel: [
                max(0.0, float(val * (1 - budget_bounds_prop_change))),
                float(val * (1 + budget_bounds_prop_change)),
            ]
            for channel, val in self.total_budget_by_channel.items()
        }

    def combine_before_after(self, dict_spend_initial, dict_spend_optimized):

        # Calculate totals for each dictionary
        total_initial = sum(dict_spend_initial.values())
        total_optimized = sum(dict_spend_optimized.values())

        # Calculate percentage contributions
        percent_initial = {k: (v / total_initial) * 100 if total_initial != 0 else 0 for k, v in dict_spend_initial.items()}
        percent_optimized = {k: (v / total_optimized) * 100 if total_optimized != 0 else 0 for k, v in dict_spend_optimized.items()}

        # Create individual DataFrames
        df_absolute = pd.DataFrame({
                            'initial': dict_spend_initial,
                            'optimized': dict_spend_optimized
            })

        df_percent = pd.DataFrame({
                            'initial': percent_initial,
                            'optimized': percent_optimized
            })

        # Combine into a dictionary
        dfs = {'spend_total': df_absolute, 'spend_percent': df_percent}

        return dfs
    
    def run_budget_optimization(self, budget_levels, n_weeks, budget_bounds):
        """
        Run budget optimization for different budget levels and combine results into a unified DataFrame.

        Parameters:
            self: The instance containing MMM and other required methods.
            budget_levels: List of total_budget levels to test.
            n_weeks: Number of weeks for the optimization.
            budget_bounds: Bounds for the budget allocation.

        Returns:
            pd.DataFrame: Unified DataFrame with original and optimized results for all budget levels.
        """
        # Capture the original budget
        original_budget = self.initial_budget_dict
        total_original = sum(original_budget.values())

        # Initialize results storage
        results_absolute = pd.DataFrame({"initial": original_budget})
        results_percentage = pd.DataFrame({"initial": {k: (v / total_original) * 100 if total_original != 0 else 0 for k, v in original_budget.items()}})

        # Iterate through budget levels
        for budget in budget_levels:
            # Run budget allocation
            optimized_budget = self.allocate_budget(
                total_budget = self.total_budget * budget, n_weeks=n_weeks, budget_bounds=budget_bounds, custom_constraints=[]
            )
            
            # Calculate totals and percentages
            total_optimized = sum(optimized_budget.values())
            absolute_results = pd.Series(optimized_budget, name=f"budget_{budget}")
            percentage_results = pd.Series(
                {k: (v / total_optimized) * 100 if total_optimized != 0 else 0 for k, v in optimized_budget.items()},
                name=f"budget_{budget}",
            )

            # Append results to the DataFrames
            results_absolute = pd.concat([results_absolute, absolute_results], axis=1)
            results_percentage = pd.concat([results_percentage, percentage_results], axis=1)

        # Return both absolute and percentage DataFrames
        return {"spend_total": results_absolute, "spend_percent": results_percentage}
    
    def allocate_budget(self, 
                        total_budget ,
                        n_weeks,
                        budget_bounds,
                         custom_constraints = []):
        
        self.response = self.mmm.allocate_budget_to_maximize_response(
                        budget = total_budget,
                        num_periods = n_weeks,
                        time_granularity = "weekly",
                        budget_bounds = budget_bounds,
                         custom_constraints = custom_constraints
                        )

        # dfs_before_after = self.combine_before_after(dict_spend_initial = self.initial_budget_dict, 
        #                                                 dict_spend_optimized = self.mmm.optimal_allocation_dict
        #                                               )
        # return dfs_before_after
        # dict_spend_optimized = self.mmm.optimal_allocation_dict                                       
        dict_spend_optimized = dict(zip(self.mmm.optimal_allocation.channel.values, self.mmm.optimal_allocation.values))

        return dict_spend_optimized

    def response_comp(self,
                      n_weeks
                      ):
        last_date = self.mmm.X["week"].max()

        # New dates starting from last in dataset
        new_dates = pd.date_range(start=last_date, periods=1 + n_weeks, freq="W-MON")[1:]

        initial_budget_scenario = pd.DataFrame(
            {
                "week": new_dates,
            }
        )

        # Same channel spends as last day
        for channel in self.initial_budget_dict:
            initial_budget_scenario[channel] = self.initial_budget_dict[channel]
        
        # MPB Come back to this! Some of this should always be on!
        for control in self.mmm.control_columns:
            initial_budget_scenario[control] = 0

        response_initial_budget = self.mmm.sample_posterior_predictive(
            X_pred=initial_budget_scenario, extend_idata=False
        )

        y_response_original_scale_optimize = (
            self.response["y"] * self.mmm.get_target_transformer()["scaler"].scale_
        )

        response_initial = response_initial_budget["y"].mean(dim="sample").data
        response_optimized = y_response_original_scale_optimize.mean(dim="sample").data
        response_delta = response_optimized - response_initial

        return response_initial, response_optimized, response_delta
    
    