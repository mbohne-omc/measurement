import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation, MichaelisMentenSaturation
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.prior import Prior
from pymc_marketing.mmm.evaluation import calculate_metric_distributions, compute_summary_metrics, summarize_metric_distributions
from sklearn.model_selection import train_test_split
import datetime 
from itertools import product
import copy

warnings.filterwarnings("ignore", category=FutureWarning)

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100

SEED = 42
rng: np.random.Generator = np.random.default_rng(seed=SEED)

def save_model(mmm, dir, client):

    save_path = dir + client + '/'
    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save the model with the date appended
    model_name = f"""{save_path}model_{current_date}.nc"""
    mmm.save(model_name)

class OutOfSample:
    def __init__(self,
                 mmm,
                 n_new_weeks = 5
                 ):
        
        self.X_out_of_sample, self.y_out_of_sample = self.out_of_sample_predictions(mmm = mmm, n_new_weeks = n_new_weeks)
        self.out_of_sample_fig, self.out_of_sample_ax = self.plot_samples(mmm, self.X_out_of_sample, self.y_out_of_sample)
        self.new_spend_contributions_fig = self.plot_new_spend_contributions(mmm, spends = [0.3, 0.5, 1, 2])

    def plot_new_spend_contributions(self, mmm, spends = [0.3, 0.5, 1, 2]):

        fig, axes = plt.subplots(
            nrows=len(spends),
            ncols=1,
            figsize=(11, 14),
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        axes = axes.flatten()

        for ax, spend in zip(axes, spends, strict=True):
            mmm.plot_new_spend_contributions(spend_amount=spend, progressbar=False, ax=ax)

        fig.suptitle("New Spend Contribution Simulations", fontsize=18, fontweight="bold");
        return fig

    def plot_samples(self, mmm, X_out_of_sample, y_out_of_sample):
        fig, ax = plt.subplots()
        self.plot_in_sample(mmm.X, mmm.y, ax=ax)
        self.plot_out_of_sample(
            X_out_of_sample, y_out_of_sample, ax=ax, label="out of sample", color="C0"
        )
        ax.legend(loc="upper left");
        plt.title("In-Sample and Out-of-Sample Predictions")
        plt.xlabel("Week")
        plt.ylabel("Values")
        return fig, ax

    def plot_in_sample(self, X, y, ax, n_points: int = 52):
        (
            pd.DataFrame(y, columns=["y"])
            .set_index(X["week"])
            .iloc[-n_points:]
            .plot(ax=ax, marker="o", color="black", label="Actuals")
        )
        return ax
    
    def plot_out_of_sample(self, X_out_of_sample, y_out_of_sample, ax, color, label):
        y_out_of_sample_groupby = y_out_of_sample["y"].to_series().groupby("date")

        lower, upper = quantiles = [0.025, 0.975]
        conf = y_out_of_sample_groupby.quantile(quantiles).unstack()
        ax.fill_between(
            X_out_of_sample["week"].dt.to_pydatetime(),
            conf[lower],
            conf[upper],
            alpha=0.25,
            color=color,
            label=f"{label} interval",
        )

        mean = y_out_of_sample_groupby.mean()
        mean.plot(ax=ax, marker="o", label=label, color=color, linestyle="--")
        ax.set(ylabel="Original Target Scale", title="Out of sample predictions for MMM")
        return ax

    def out_of_sample_predictions(self, mmm, n_new_weeks):

        # Assuming mmm.X is your original DataFrame
        last_date = mmm.X["week"].max()

        # New dates starting from the last date in the dataset
        new_dates = pd.date_range(start=last_date, periods=1 + n_new_weeks, freq="W-MON")[1:]

        # Create a new DataFrame with the new dates
        X_out_of_sample = pd.DataFrame({
            "week": new_dates,
        })

        # Get the last values for all columns except "week"
        last_values = mmm.X.iloc[-1].drop("week")

        # Add these last values to the new DataFrame for the number of new rows
        for col in last_values.index:
            X_out_of_sample[col] = last_values[col]

        X_out_of_sample["t"] = range(len(mmm.X), len(mmm.X) + n_new_weeks)

        y_out_of_sample = mmm.sample_posterior_predictive(
            X_pred=X_out_of_sample, extend_idata=False, include_last_observations=True
        )

        return X_out_of_sample, y_out_of_sample

class RunMMM:
    def __init__(self, 
                 prepped_data_df,
                 outcome_dict,
                 channel_dict,
                 extra_dict = {"t": "t"},
                 prop_training = .9,
                 date_column = 'week',
                 model_config = False,
                 sampler_config ={
                    "tune": 1000,
                    "draws": 1000,
                    "target_accept": 0.85,
                    "cores": 2,
                    "chains": 2,
                    "random_seed": SEED,
                    "progress_bar": True,
                    },
                prior_sigma_scale_candidates = [1],
                yearly_seasonality_candidates = [2],
                max_weeks_candidates = [8]
                ):
        
        """Initialize the DataAnalyzer class with a DataFrame."""

        # Class inputs
        self.channel_dict = channel_dict 
        self.outcome_dict = outcome_dict
        self.extra_dict = extra_dict
        self.model_config = model_config
        self.sampler_config = sampler_config
        self.prior_sigma_scale_candidates = prior_sigma_scale_candidates
        self.yearly_seasonality_candidates = yearly_seasonality_candidates
        self.max_weeks_candidates = max_weeks_candidates

        if extra_dict:
            self.extra_dict
            self.combined_dict = {**channel_dict, **outcome_dict, **extra_dict}
        else:
            self.combined_dict = {**channel_dict, **outcome_dict}

        # Clean terms for use throughout the class
        self.prop_training = prop_training
        self.date_column = date_column
        self.channel_columns = sorted(list(channel_dict.values()))
        self.extra_columns = sorted(list(extra_dict.values())) if extra_dict else None
        self.outcome_column = list(outcome_dict.values())[0]
        self.all_used_columns = self.channel_columns + self.extra_columns + [self.outcome_column] + [self.date_column]
        self.X_columns = self.channel_columns + self.extra_columns + [self.date_column]

        # Preprocess the data
        self.prepped_data_df = self.preprocess_data(prepped_data_df = prepped_data_df, 
                                                    combined_dict = self.combined_dict, 
                                                    date_column = self.date_column, 
                                                    all_used_columns = self.all_used_columns, 
                                                    outcome_column = self.outcome_column)

        """# ... fill in options"""
        if prop_training > 0: 
            self.train_df, self.test_df = self.split_data(self.prepped_data_df)
        else: 
            self.train_df = self.prepped_data_df

        # print(self.train_df.describe())

        # Create MMM configs to test as candidate
        if self.model_config:
            """self.mmm_candidates = ..."""
        else:
            self.mmm_candidates = self.create_mmm_candidates(data = self.train_df, channel_columns = self.channel_columns)

        # Fit and evaluate candidate models
        self.model_fits = self.fit_and_evaluate_candidates(self.train_df, self.test_df)

        # Extract valid metrics and ROAS from the model results for LLM
        self.valid_metrics_and_roas = self.extract_valid_metrics_and_roas(self.model_fits)

        # Select the best model
        # MPB update here
        self.best_mmm_candidate = self.select_best_model(self.model_fits, metric="rmse")

        # Refit the best model on all data
        self.final_mmm = self.fit_mmm(self.best_mmm_candidate, self.prepped_data_df[self.X_columns], self.prepped_data_df[self.outcome_column])

        # Store model evaluation results on all data
        self.final_mmm_evaluation = self.evaluate_model_on_test_data(self.final_mmm, self.prepped_data_df)

        # Store model diagnostics for the final model
        self.final_mmm_diagnostics = self.store_model_diagnostics(self.final_mmm, self.prepped_data_df)
        
        # Collect model output
        self.model_output = self.model_output(self.final_mmm)

    def extract_valid_metrics_and_roas(self, data: dict) -> dict:
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

            # Extract valid summary metric means
            summary_metrics = model_data.get('summary_metrics', {})
            for metric, stats in summary_metrics.items():
                mean = stats.get('mean', None)
                if mean is not None and np.isfinite(mean):
                    entry[metric] = mean

            results[key] = entry

        return results


    def fit_and_evaluate_candidates(self, train_df, test_df):
        """
        Fits each candidate MMM model on the training data and evaluates it using Bayesian posterior predictive sampling.
        
        Args:
            train_df (pd.DataFrame): Training dataset.
            test_df (pd.DataFrame): Test dataset.

        Returns:
            dict: Dictionary containing:
                - fitted_model: The trained MMM model.
                - final_model: The corresponding untouched model from final_mmm_candidates.
                - summary_metrics: Output of compute_summary_metrics (for ranking models).
                - metric_distributions: Output of calculate_metric_distributions (for deeper analysis).
                - distribution_summaries: Output of summarize_metric_distributions (for statistical insights).
        """
        results = {}

        test_mmm_candidates = self.mmm_candidates["test_mmm_candidates"]
        final_mmm_candidates = self.mmm_candidates["final_mmm_candidates"]

        for i, (test_mmm, final_mmm) in enumerate(zip(test_mmm_candidates, final_mmm_candidates)):
            print(f"\n🚀 Fitting candidate model {i+1}/{len(test_mmm_candidates)}...")

            try:
                # Fit the model on training data
                fitted_mmm = self.fit_mmm(test_mmm, train_df[self.X_columns], train_df[self.outcome_column])

                # Get ROAS
                roas_plot, mean_roas_dict = self.get_roas_plot(fitted_mmm)

                # Evaluate the model
                evaluation_results = self.evaluate_model_on_test_data(fitted_mmm, test_df)

                # Store results
                results[i] = {
                    "fitted_model": fitted_mmm,  # Trained model
                    "final_model": final_mmm,  # Untrained model for final fit
                    "mean_roas_dict": mean_roas_dict,  # Mean ROAS values
                    **evaluation_results,  # Store evaluation outputs
                }

            except Exception as e:
                print(f"⚠️ Model {i+1} failed to fit: {e}\nSkipping this model.")
                continue

        print("\n✅ Model fitting & evaluation complete.")
        return results  
        
    def evaluate_model_on_test_data(self, mmm, test_df, hdi_prob = 0.89):
            """
            Evaluates the fitted MMM model on test data using Bayesian posterior predictive samples.

            Args:
                mmm (MMM): Fitted MMM model.
                test_df (pd.DataFrame): Test dataset.

            Returns:
                dict: Dictionary containing:
                    - summary_metrics (compute_summary_metrics output, for model selection)
                    - metric_distributions (full metric distributions)
                    - distribution_summaries (summarized metric distributions)
            """
            print("\nGenerating posterior predictive samples...")
            
            # Generate posterior predictive samples for test data
            posterior_preds = mmm.sample_posterior_predictive(test_df[self.X_columns], random_seed=SEED)

            # Define the evaluation metrics to calculate
            metrics_to_calculate = ["rmse", "mae", "mape", "r_squared", "nrmse", "nmae"]
            
            # 1️) Compute full metric distributions
            metric_distributions = calculate_metric_distributions(
                y_true=test_df[self.outcome_column].values,
                y_pred=posterior_preds.y.to_numpy(),
                metrics_to_calculate=metrics_to_calculate,
            )

            # 2) Summarize metric distributions
            distribution_summaries = summarize_metric_distributions(
                metric_distributions,
                hdi_prob=hdi_prob  # Adjust HDI as needed
            )

            # 3) Compute summary metrics (for quick ranking & selection)
            summary_metrics = compute_summary_metrics(
                y_true=test_df[self.outcome_column].values,
                y_pred=posterior_preds.y.to_numpy(),
                metrics_to_calculate=metrics_to_calculate,
                hdi_prob=hdi_prob,
            )

            # Print summary metrics
            print("\n📊 Summary Metrics for Model Evaluation:")
            for metric, stats in summary_metrics.items():
                print(f"\n{metric.upper()}:")
                for stat, value in stats.items():
                    print(f"  {stat}: {value:.4f}")

            return {
                "summary_metrics": summary_metrics,  # Used for model selection
                "metric_distributions": metric_distributions,  # Full distributions
                "distribution_summaries": distribution_summaries,  # Summarized metrics
            }    


    def select_best_model(self, model_results, metric="nrmse"):
        """
        Selects the best model based on the chosen metric.

        Args:
            model_results (dict): Dictionary of model results from fit_and_evaluate_candidates().
            metric (str): The metric to use for ranking models (default: "nrmse").

        Returns:
            MMM: Best-performing MMM **candidate model (config)** from final_mmm_candidates.
        """
        print(f"\n🚀 Selecting the best model based on {metric}...")

        # Check if there are valid results
        valid_results = {i: res for i, res in model_results.items() if res is not None}
        if not valid_results:
            raise ValueError("No valid models found for selection.")

        # Print all candidate models' NRMSE values
        for i, result in valid_results.items():
            print(f"Model {i+1}: {metric} = {result['summary_metrics'][metric]['mean']:.4f}")

        # Select the model with the lowest mean NRMSE
        best_model_id = min(valid_results, key=lambda i: valid_results[i]["summary_metrics"][metric]["mean"])

        print(f"\n✅ Best model selected: Model {best_model_id+1}")

        # Retrieve the **untrained final model** for refitting
        best_mmm_candidate = valid_results[best_model_id]["final_model"]

        return best_mmm_candidate
    
    def create_mmm_candidates(self, data: pd.DataFrame, channel_columns) -> dict:
        """
        Creates two sets of MMM candidates: one for fitting and another identical set for final selection.

        Args:
            data (pd.DataFrame): The input dataset.
            channel_columns (list): List of channel column names.

        Returns:
            dict: A dictionary with:
                - "test_mmm_candidates": List of MMM models for initial fitting.
                - "final_mmm_candidates": A separate identical set for final selection.
        """

        def compute_ad_spend_priors(X: pd.DataFrame) -> list:
            """
            Compute priors proportional to ad spend.

            Args:
                X (pd.DataFrame): DataFrame containing ad spend data with channels as columns.

            Returns:
                list: Prior sigma values proportional to ad spend.
            """
            # Extract relevant columns from the DataFrame
            total_spend_per_channel = X.sum(axis=0)
            
            # Compute spend share per channel
            spend_share = total_spend_per_channel / total_spend_per_channel.sum()
            
            # Number of channels
            n_channels = X.shape[1]
            
            # Compute prior sigma values
            prior_sigma = (n_channels * spend_share.to_numpy()).tolist()
            
            return prior_sigma

        def cross_join_training_params(prior_sigma_scale, yearly_seasonality, max_weeks):
            """
            Create an object holding the cross join of three lists.

            Args:
                prior_sigma_scale (list): Scaling factors for prior sigma.
                yearly_seasonality (list): Yearly seasonality candidates.
                max_weeks (list): Maximum weeks for adstock.

            Returns:
                pd.DataFrame: DataFrame containing all combinations of parameters.
            """
            cross_joined = list(product(prior_sigma_scale, yearly_seasonality, max_weeks))
            
            # Convert to a DataFrame for better structure
            df = pd.DataFrame(cross_joined, columns=['prior_sigma_scale', 'yearly_seasonality', 'max_weeks'])
            
            return df

        # Get priors based on ad spend
        X = data[channel_columns]
        prior_sigma = compute_ad_spend_priors(X)

        # Generate all candidate configurations
        model_tuning_permutations = cross_join_training_params(
            self.prior_sigma_scale_candidates, 
            self.yearly_seasonality_candidates, 
            self.max_weeks_candidates
        )

        test_mmm_candidates = []
        final_mmm_candidates = []

        # Iterate over each row in the DataFrame properly
        for _, model_params in model_tuning_permutations.iterrows():
            # Extract values properly
            prior_sigma_scale = model_params['prior_sigma_scale']
            yearly_seasonality = model_params['yearly_seasonality']
            max_weeks = model_params['max_weeks']

            # Compute new prior sigma
            adjusted_prior_sigma = [prior_sigma_scale * ps for ps in prior_sigma]

            # Define the model configuration
            model_config = {
                "intercept": Prior("Normal", mu=0.5, sigma=0.2),
                "saturation_beta": Prior("HalfNormal", sigma=adjusted_prior_sigma),
                "gamma_control": Prior("Normal", mu=0, sigma=0.05),
                "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
            }

            # Create the MMM model for fitting
            test_mmm = MMM(
                model_config=model_config,
                sampler_config=self.sampler_config,
                date_column=self.date_column,
                adstock=GeometricAdstock(l_max=max_weeks),
                saturation=LogisticSaturation(),
                channel_columns=self.channel_columns,
                control_columns=self.extra_columns,
                yearly_seasonality=yearly_seasonality,
            )

            # Create an identical MMM model for final selection (ensuring a fresh start)
            final_mmm = MMM(
                model_config=model_config,
                sampler_config=self.sampler_config,
                date_column=self.date_column,
                adstock=GeometricAdstock(l_max=max_weeks),
                saturation=LogisticSaturation(),
                channel_columns=self.channel_columns,
                control_columns=self.extra_columns,
                yearly_seasonality=yearly_seasonality,
            )

            # Store both models separately
            test_mmm_candidates.append(test_mmm)
            final_mmm_candidates.append(final_mmm)

            mmm_candidates = {
                "test_mmm_candidates": test_mmm_candidates,
                "final_mmm_candidates": final_mmm_candidates
            }

        return mmm_candidates
    
    def preprocess_data(self, prepped_data_df: pd.DataFrame, combined_dict: dict, date_column: str, all_used_columns: list, outcome_column: str) -> pd.DataFrame:
        """
        Preprocess data by subsetting relevant columns, renaming them, and filtering early zero rows.

        Args:
            prepped_data_df (pd.DataFrame): The input DataFrame.
            date_column (str): The name of the date column.
            combined_dict (dict): Dictionary for renaming columns.
            all_used_columns (list): List of all columns used in the model.
            outcome_column (str): The name of the outcome column.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """

        def filter_early_zero_rows(prepped_data_df, outcome_column, date_column):

            # Find the minimum date where target_column is not zero
            min_date_row = prepped_data_df.loc[prepped_data_df[outcome_column] != 0, date_column].min()

            # Filter the dataset to include only rows from min_date_row onward
            prepped_data_df = prepped_data_df.loc[prepped_data_df[date_column] >= min_date_row]

            return prepped_data_df

        # Rename columns
        prepped_data_df = prepped_data_df.rename(columns=combined_dict)
        
        # Filter beginning data with 0 for outcome
        prepped_data_df = filter_early_zero_rows(prepped_data_df=prepped_data_df, outcome_column=outcome_column, date_column=date_column)

        # Add trend term
        prepped_data_df["t"] = range(prepped_data_df.shape[0])

        # Subset to relevant columns
        prepped_data_df = prepped_data_df[all_used_columns]        
        
        return prepped_data_df

    def split_data(self, df: pd.DataFrame):
        """Split the data for training and test."""
        training_data, testing_data = train_test_split(df, train_size=self.prop_training, shuffle=False)
        return training_data, testing_data
    
    def fit_mmm(self, mmm, X, y):
            # Fit the model
        print('fitting_mmm')
        mmm.fit(X, y, target_accept=0.85, chains=2, random_seed=SEED)
        return mmm

    def store_model_diagnostics(self, mmm, prepped_data_df):

        model_diagnostics = {}
        # Store the number of divergences in the dictionary
        model_diagnostics['n_divergences'] = mmm.idata["sample_stats"]["diverging"].sum().item()

        # Store model summary
        # Define the base variable names
        var_names = [
            "intercept",
            "y_sigma",
            "saturation_beta",
            "saturation_lam",
            "adstock_alpha",
            "gamma_fourier",
        ]
        
        # Check if 'gamma_control' is present in the fit_result
        if 'gamma_control' in mmm.fit_result.data_vars:
            var_names.append("gamma_control")  # Add gamma_control if present

        model_diagnostics['summary'] = az.summary(
            data=mmm.fit_result,
            var_names=var_names,
        )

        # Create the trace plot
        _ = az.plot_trace(
            data=mmm.fit_result,
            var_names=var_names,
            compact=True,
            backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
        )
        
        plt.gcf().suptitle("Model Trace", fontsize=16)

        # Save the current figure object in the diagnostics
        model_diagnostics['trace_plot'] = plt.gcf()

        mmm.sample_posterior_predictive(prepped_data_df[self.X_columns], extend_idata=True, combined=True)

        # Generate prior predictive samples
        mmm.sample_prior_predictive(mmm.X, mmm.y, samples=2_000)
        model_diagnostics['plot_prior_predictive'] = mmm.plot_prior_predictive();
        
        # Sample Posterior Predictive
        mmm.sample_posterior_predictive(mmm.X, extend_idata=True, combined=True)
        model_diagnostics['plot_posterior_predictive'] = mmm.plot_posterior_predictive(original_scale=True);

        # Errors
        errors = mmm.plot_errors(original_scale=True);
        model_diagnostics['plot_errors'] = errors

        # Fix...

        # fig, ax = plt.subplots(figsize=(8, 6))
        # az.plot_dist(
        #     errors, quantiles=[0.25, 0.5, 0.75], color="C3", fill_kwargs={"alpha": 0.7}, ax=ax
        # )
        # ax.axvline(x=0, color="black", linestyle="--", linewidth=1, label="zero")
        # ax.legend()
        # ax.set(title="Errors Posterior Distribution");
        # model_diagnostics['errors_posterior_distribution'] = ax

        # Components Contributions
        model_diagnostics['components_contributions'] = mmm.plot_components_contributions();
        # mmm.plot_components_contributions(original_scale=True);

        return model_diagnostics

    def get_contribution_plot(self, mmm):
        # Get the channels and controls
        channels = mmm.channel_columns
        controls = mmm.control_columns

        # Create the groups dictionary with decomposed base components
        groups = {
            "Intercept": ["intercept"],
            "Positive Seasonality": ["positive_seasonality"],
            "Negative Seasonality": ["negative_seasonality"],
        }

        # Add each channel name directly to the groups dictionary
        for channel in channels:
            groups[channel] = [channel]

        # Add controls as their own group if they exist
        if controls is not None:
            groups["Controls"] = controls

        # Generate a color map dynamically for the channels and controls using matplotlib colors
        color_map = {channel: f"C{i}" for i, channel in enumerate(channels)}

        # Assign unique colors to controls if they exist
        if controls is not None:
            control_start_index = len(channels)
            for i, control in enumerate(controls):
                color_map[control] = f"C{control_start_index + i}"
            color_map["Controls"] = f"C{control_start_index + len(controls)}"

        # Update color map for base components
        color_map.update({
            "Intercept": "gray",
            "Positive Seasonality": "green",
            "Negative Seasonality": "red",
        })

        # Plot using the dynamically created groups and colors
        fig = mmm.plot_grouped_contribution_breakdown_over_time(
            stack_groups=groups,
            original_scale=True,
            area_kwargs={
                "color": color_map,
                "alpha": 0.7,
            },
        )

        # Add a title
        fig.suptitle("Contribution Breakdown over Time", fontsize=16)

        return fig

    def get_direct_contribution_curves(self, mmm):
        fig = mmm.plot_direct_contribution_curves()
        [ax.set(xlabel="x") for ax in fig.axes];
        return fig
    
    def get_roas_plot(self, mmm):
        # Compute channel contributions and total spend
        channel_contribution_original_scale = mmm.compute_channel_contribution_original_scale()
        spend_sum = mmm.X[mmm.channel_columns].sum().to_numpy()

        # Calculate ROAS samples
        roas_samples = (
            channel_contribution_original_scale.sum(dim="date")
            / spend_sum[np.newaxis, np.newaxis, :]
        )

        # Create subplots for each channel
        fig, axes = plt.subplots(
            nrows=len(mmm.channel_columns), 
            ncols=1, 
            figsize=(12, 7), 
            sharex=True, 
            sharey=False, 
            layout="constrained"
        )

        # Plot posterior distributions for ROAS
        az.plot_posterior(roas_samples, ax=axes)

        # Set titles for each axis dynamically and label the x-axis on the last subplot
        for i, channel in enumerate(mmm.channel_columns):
            axes[i].set(title=f"{channel}")

        # Set the x-axis label only for the last subplot
        axes[-1].set(xlabel="ROAS")

        # Set a super title for the figure
        fig.suptitle("ROAS Posterior Distributions", fontsize=18, fontweight="bold", y=1.06)

        # Calculate the mean ROAS values from roas_samples
        mean_roas_by_chain = roas_samples.mean(dim="draw").values  # Extracting the mean values

        # Calculate the mean for each column
        mean_roas = np.mean(mean_roas_by_chain, axis=0).round(2)

        # Create a dictionary mapping channel names to their mean ROAS values
        mean_roas_dict = {
            channel: mean for channel, mean in zip(mmm.channel_columns, mean_roas)
        }

        return fig, mean_roas_dict


    def model_output(self, mmm):

        model_output = {}
        model_output['contribution_plot'] = self.get_contribution_plot(mmm)
        model_output['waterfall_components_decomposition_plot'] = mmm.plot_waterfall_components_decomposition(original_scale=True);
        model_output['contributions_over_time'] = mmm.compute_mean_contributions_over_time(original_scale=True)
        model_output['channel_contribution_share_hdi_plot'] = mmm.plot_channel_contribution_share_hdi(hdi_prob = .94, figsize=(7, 5))
        model_output['direct_contribution_curves_plot'] = self.get_direct_contribution_curves(mmm)
        model_output['channel_contributions_grid_plot'] = mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12);
        model_output['channel_contributions_grid_absolute_plot'] = mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12, absolute_xrange=True);
        model_output['roas_plot'], model_output['mean_roas_dict'] = self.get_roas_plot(mmm)

        return model_output
