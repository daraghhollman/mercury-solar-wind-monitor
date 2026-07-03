from matplotlib.lines import Line2D
import datetime as dt
from dataclasses import dataclass
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from gpflow.kernels import RationalQuadratic
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag
from mvswm.model import GapManager, SolarWindModel, TimeScaler
from mvswm.utils.colours import *

COMPONENTS = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]
COMPONENT_COLOURS = [BLACK, RED, GREEN, BLUE]


def main() -> None:


    data = get_messenger_solar_wind_data(
        TimeRange("2011-03-23", dt.timedelta(hours=120)),
        bow_shock_buffer=dt.timedelta(minutes=10),
    )

    truths: List[pl.DataFrame] = []
    predictions: List[pl.DataFrame] = []
    baselines: List[pl.DataFrame] = []

    print("")

    # GPRs are computationally expensive, scaling with n^3. As a result, it is
    # important to keep input data small. For these reasons, we choose to split
    # the data based on the number of data-points. A reasonable range is between
    # 1k and 10k.
    split_length: int = 1000
    n_splits: int = round(len(data) / split_length)
    for split_index in range(n_splits):

        split_data = data.slice(split_index * split_length, split_length)

        print(f"{dt.datetime.now()} | Training model on data from {split_data['UTC'][0]} to {split_data['UTC'][-1]}")

        # This data will have some data-gaps inherent to MESSENGER's orbit
        # around Mercury. We also need to add additional artificial data-gaps
        # where we will test the model's performance.
        # With a default gap placement strategy of 'middle', the gaps are
        # places in between two real gaps (when they are able to fit). This
        # means that it is possible to have data splits without an artificial
        # gap. We should just skip these instead of training on them.
        gm = GapManager(split_data, gap_length=dt.timedelta(hours=3))

        training_data = gm.training_data
        evaluation_data = gm.evaluation_data

        if training_data is None or evaluation_data is None:
            # If, based on the gaps, we don't have both a training and an
            # evaluation dataset in this window, skip.
            continue

        component_predictions: pl.DataFrame | None = None
        evaluation_predictions: pl.DataFrame | None = None
        baseline_predictions: pl.DataFrame | None = None
        evalutation_times: NDArray = evaluation_data["UTC"].to_numpy()

        for component in COMPONENTS:

            # Reshape data for model
            X: NDArray = training_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
            Y: NDArray = (
                training_data.drop_nulls()[component]
                .to_numpy()
                .reshape(-1, 1)
                .astype("float64")
            )

            time_scaler = TimeScaler(X)

            # Create model
            model = SolarWindModel.build(
                input=X,
                output=Y,
                time_scaler=time_scaler,
                kernel=RationalQuadratic(),
            )

            model.train_model()

            # Make predictions over the whole interval for plotting
            #######################################################
            x_range = np.linspace(0, 1, 10000)[:, None]

            y_mean: tf.Tensor
            y_var: tf.Tensor
            y_mean, y_var = model.model.predict_y(x_range)
            x_out = model.time_scaler.numeric_to_time(x_range).reshape(-1)

            component_df = pl.DataFrame(
                {
                    "UTC": x_out,
                    f"{component} Mean": y_mean.numpy().reshape(-1),
                    f"{component} Variance": y_var.numpy().reshape(-1),
                }
            )

            if component_predictions is None:
                component_predictions = component_df

            else:
                component_predictions = component_predictions.join(
                    component_df, on="UTC"
                )

            # Make predictions over the evaluation data only
            ################################################
            evaluation_x = model.time_scaler.time_to_numeric(
                evalutation_times.reshape(-1, 1)
            )
            evaluation_prediction_mean: tf.Tensor
            evaluation_prediction_var: tf.Tensor
            evaluation_prediction_mean, evaluation_prediction_var = (
                model.model.predict_y(evaluation_x)
            )

            evaluation_df = pl.DataFrame(
                {
                    "UTC": evalutation_times,
                    f"{component}": evaluation_prediction_mean.numpy().reshape(-1),
                }
            )

            if evaluation_predictions is None:
                evaluation_predictions = evaluation_df

            else:
                evaluation_predictions = evaluation_predictions.join(
                    evaluation_df, on="UTC"
                )

            # Baseline prediction: Linear Interpolation
            ###########################################
            # Perform a linear interpolation from the data surrounding the
            # evalation data.
            training_x_numeric = time_scaler.time_to_numeric(X).reshape(-1)
            training_y_flat = Y.reshape(-1)

            linear_interpolation = np.interp(
                evaluation_x.reshape(-1),
                training_x_numeric,
                training_y_flat,
            )

            linear_interpolation_df = pl.DataFrame(
                {
                    "UTC": evalutation_times,
                    f"{component}": linear_interpolation,
                }
            )

            if baseline_predictions is None:
                baseline_predictions = linear_interpolation_df

            else:
                baseline_predictions = baseline_predictions.join(
                    linear_interpolation_df, on="UTC"
                )

        assert component_predictions is not None
        assert evaluation_predictions is not None
        assert baseline_predictions is not None

        truths.append(evaluation_data)
        predictions.append(evaluation_predictions)
        baselines.append(baseline_predictions)

        """
        # Get metrics
        print("\nMODEL PREDICTIONS")
        metrics = get_metrics(evaluation_data, evaluation_predictions)
        for m in metrics:
            print(m)

        print("\nBASELINE PREDICTIONS")
        metrics = get_metrics(evaluation_data, baseline_predictions)
        for m in metrics:
            print(m)

        fig, ax = plt.subplots()

        for component, colour in zip(COMPONENTS, component_colours):
            ax.scatter(
                training_data["UTC"], training_data[component], marker=".", color=colour
            )
            ax.scatter(
                evaluation_data["UTC"],
                evaluation_data[component],
                marker=".",
                color=colour,
                alpha=0.5,
            )

        # Plot predictions
        for component, colour in zip(COMPONENTS, component_colours):
            ax.plot(
                component_predictions["UTC"],
                component_predictions[component + " Mean"],
                color=colour,
            )

            y_upper = component_predictions[component + " Mean"] + 1.96 * np.sqrt(
                component_predictions[component + " Variance"]
            )
            y_lower = component_predictions[component + " Mean"] - 1.96 * np.sqrt(
                component_predictions[component + " Variance"]
            )

            ax.fill_between(
                component_predictions["UTC"], y_lower, y_upper, alpha=0.3, color=colour
            )

        # Plot baseline
        for component, colour in zip(COMPONENTS, component_colours):
            ax.plot(
                baseline_predictions["UTC"],
                baseline_predictions[component],
                color=colour,
                ls="dashed",
            )

        plt.show()
        """

    r = PerformanceReport(truths, predictions, baselines)
    r.make_taylor_diagram()

    plt.show()


class PerformanceReport:

    def __init__(
        self,
        truths: List[pl.DataFrame],
        predictions: List[pl.DataFrame],
        baselines: List[pl.DataFrame],
    ) -> None:
        self._created = dt.datetime.now()

        self.truths = truths
        self.predictions = predictions
        self.baselines = baselines

    def make_taylor_diagram(self, ax: Axes | None = None) -> Axes:

        # Create a matplotlib axis if it doesn't exist.
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # Loop through inputs and add points accordingly
        for truth, prediction, baseline in zip(
            self.truths, self.predictions, self.baselines
        ):
            for component, colour in zip(COMPONENTS, COMPONENT_COLOURS):
                prediction_std = prediction[component].std() / truth[component].std()
                baseline_std = baseline[component].std() / truth[component].std()
                prediction_corr = pearson_r_wrapper(truth[component], prediction[component])
                baseline_corr = pearson_r_wrapper(truth[component], baseline[component])

                ax.scatter(
                    np.arccos(prediction_corr),
                    prediction_std,
                    color=colour,
                    marker="o",
                )
                ax.scatter(
                    np.arccos(baseline_corr),
                    baseline_std,
                    color=colour,
                    marker="x",
                )

        correlation_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1])
        theta_positions = np.arccos(correlation_ticks)

        ax.set(
            thetamin=0,
            thetamax=90,  # Stop after an angle of 90
            xticks=theta_positions,
            xticklabels=correlation_ticks,
            yticks=np.arange(0, 1.2 + 0.2, 0.2),
        )

        ax.text(0.5, -0.1, "$\sigma / \sigma_d$", transform=ax.transAxes)
        ax.text(0.75, 0.75, "$r_p$", rotation=-45, transform=ax.transAxes)

        # Add dashed line at y=1
        ax.axhline(y=1, lw=3, ls="dashed", color="black")

        # RMSE contours (concentric circles centered on the reference point)
        # Reference point is at (theta=0, r=1), i.e. correlation=1, sigma/sigma_d=1
        theta_grid = np.linspace(0, np.pi / 2, 200)
        r_grid = np.linspace(0, 1.2, 200)
        T, R = np.meshgrid(theta_grid, r_grid)

        # Law of cosines: RMSE^2 = sigma_ref^2 + sigma^2 - 2*sigma_ref*sigma*cos(theta)
        # sigma_ref = 1 (normalized), sigma = R, theta = T
        RMSE = np.sqrt(1 + R**2 - 2 * R * np.cos(T))

        rmse_levels = np.linspace(0.2, 1, 3)
        ax.contour(
            T, R, RMSE,
            levels=rmse_levels,
            colors="gray",
            linestyles="dotted",
            linewidths=3,
        )

        # A manual 'table-style' legend
        n_rows = len(COMPONENTS)

        # Position: to the right of the main axes, in axes-fraction coordinates.
        # Adjust the [x0, y0, width, height] values to taste.
        legend_ax = ax.inset_axes((0.9, 0.75, 0.32, 0.05 * n_rows + 0.08), transform=ax.transAxes)
        legend_ax.set_xlim(0, 3)
        legend_ax.set_ylim(0, n_rows + 1)
        legend_ax.axis("off")

        # Column headers
        legend_ax.text(1, n_rows + 0.5, "GPR", ha="center", va="center", fontweight="bold")
        legend_ax.text(2, n_rows + 0.5, "LI", ha="center", va="center", fontweight="bold")

        # Header underline
        legend_ax.plot([0, 3], [n_rows + 0.05, n_rows + 0.05], color="black", linewidth=0.8)

        # One row per component
        for i, component in enumerate(COMPONENTS):
            row_y = n_rows - i - 0.5
            legend_ax.text(
                0, row_y, component[:-4], ha="left", va="center", color=COMPONENT_COLOURS[i]
            )
            legend_ax.scatter([1], [row_y], color=COMPONENT_COLOURS[i], marker="o")
            legend_ax.scatter([2], [row_y], color=COMPONENT_COLOURS[i], marker="x")

            # faint row separator
            legend_ax.plot([0, 3], [row_y - 0.5, row_y - 0.5], color="grey", linewidth=0.4, alpha=0.4)

        return ax


@dataclass
class MetricSummary:
    name: str
    values: List[float]

    @property
    def mean(self):
        return np.mean(self.values)

    @property
    def median(self):
        return np.median(self.values)

    @property
    def sd(self):
        return np.std(self.values)

    def __repr__(self):
        return f"{self.name}: {self.mean:.3f} ({self.median:.3f}) +/- {self.sd:.3f}"


def pearson_r_wrapper(x, y):
    """
    Pulls out the statistic value from scipy.stats.pearsonr
    """
    return pearsonr(x, y).statistic


def get_metrics(
    true_data: pl.DataFrame,
    model_predictions: pl.DataFrame,
    y_variables: List[str] = [
        "|B| [nT]",
        "Br [nT]",
        "Bt [nT]",
        "Bn [nT]",
    ],
    metrics: List[Callable] = [
        # A list of functions which all take input: (y_true, y_pred)
        pearson_r_wrapper,
        r2_score,
        mean_absolute_error,
        root_mean_squared_error,
    ],
) -> List[MetricSummary]:

    # First check that all `y_variables` exist in the data
    for parameter in y_variables:
        if (
            parameter not in true_data.columns
            or parameter not in model_predictions.columns
        ):
            raise ValueError(
                f"Parameter {parameter} does not exist in input data. Cannot determine metrics."
            )

    metric_summaries: List[MetricSummary] = []
    for metric in metrics:
        metric_scores: List[float] = []
        for parameter in y_variables:

            result = metric(true_data[parameter], model_predictions[parameter])
            metric_scores.append(result)

        this_metric = MetricSummary(metric.__name__, metric_scores)
        metric_summaries.append(this_metric)

    return metric_summaries


def plot_kde_region(
    ax, corr, std, color, coverage=0.5, alpha=0.1, grid_size=200, bw_method=None
):
    """
    Overlay a shaded 'coverage'-probability KDE region on a Taylor diagram (polar axes).

    corr : array-like of correlation values
    std  : array-like of standard deviation values
    coverage : float, fraction of probability mass to enclose (0.5 = 50% HDR)
    """
    corr = np.asarray(corr, dtype=float)
    std = np.asarray(std, dtype=float)

    # Clip correlation to valid domain for arccos (guards against floating point drift)
    corr = np.clip(corr, -1.0, 1.0)

    theta = np.arccos(corr)
    r = std

    # Drop any remaining NaN/inf pairs
    mask = np.isfinite(theta) & np.isfinite(r)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"plot_kde_region: dropping {n_dropped} non-finite point(s)")
    theta, r = theta[mask], r[mask]

    if len(theta) < 3:
        print("plot_kde_region: not enough points for a KDE, skipping")
        return None

    # Work in Cartesian space so the KDE bandwidth is isotropic/meaningful
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy = np.vstack([x, y])

    if np.allclose(xy, xy[:, [0]]):
        print("plot_kde_region: all points identical, skipping KDE")
        return None

    kde = gaussian_kde(xy, bw_method=bw_method)

    # Build an evaluation grid directly in polar space (matches your axis limits)
    theta_grid = np.linspace(0, np.pi / 2, grid_size)
    r_max = ax.get_ylim()[1]
    r_grid = np.linspace(0, r_max, grid_size)
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)

    # Convert grid to Cartesian to evaluate the KDE
    x_mesh = r_mesh * np.cos(theta_mesh)
    y_mesh = r_mesh * np.sin(theta_mesh)
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    density = kde(positions).reshape(theta_mesh.shape)

    # --- Find the density threshold enclosing `coverage` fraction of the mass ---
    # Sort densities descending, accumulate mass (weighted by grid cell area in
    # Cartesian terms, approximated here via r since polar cells scale with r)
    cell_area = r_mesh * (theta_grid[1] - theta_grid[0]) * (r_grid[1] - r_grid[0])
    density_flat = density.ravel()
    area_flat = cell_area.ravel()

    order = np.argsort(density_flat)[::-1]
    sorted_density = density_flat[order]
    sorted_mass = sorted_density * area_flat[order]
    cumulative_mass = np.cumsum(sorted_mass)
    cumulative_mass /= cumulative_mass[-1]  # normalise to 1

    idx = np.searchsorted(cumulative_mass, coverage)
    threshold = sorted_density[min(idx, len(sorted_density) - 1)]

    # Plot a single filled region above the threshold
    cs = ax.contourf(
        theta_mesh,
        r_mesh,
        density,
        levels=[threshold, density.max()],
        colors=[color],
        alpha=alpha,
    )
    return cs


def get_messenger_solar_wind_data(
    time_range: TimeRange, bow_shock_buffer: dt.timedelta
) -> pl.DataFrame:
    """
    Loads MESSENGER MAG data and filters to only times in the solar wind based
    on bow shock crossings by Hollman et al. (2026). These crossings are
    buffered to ensure the data is solar wind.

    Params
    ------
    time_range: TimeRange
        The range of data to load

    bow_shock_buffer: datetime.timedelta
        How much time to buffer all bow shock crossings by

    Returns
    -------
    data: polars.DataFrame
    """

    messenger = Spacecraft("MESSENGER")
    data = messenger.get_data_in_range(time_range)

    # Remove time within bow shock. This is buffered
    data = filter_messenger_mag(data, buffer=bow_shock_buffer)

    return data


if __name__ == "__main__":
    main()
