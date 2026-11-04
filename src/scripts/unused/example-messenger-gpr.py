"""
An example GPR and LI proof of concept application for MESSENGER.

4x2 panels, first column is of a small timeseries of MESSENGER data, with two
example artificial gaps. Each row represents a component of the magnetic field.
The second column will have Taylor diagrams for 10 randomly chosen gaps witin
the same timeseries.
"""

import datetime as dt
import random
from typing import Any, Dict, List, Tuple

import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from gpflow.kernels import RationalQuadratic
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, HourLocator
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, linregress
from sunpy.time import TimeRange
from tqdm import tqdm

from mvswm.data import get_messenger_solar_wind_data
from mvswm.model import SolarWindModel, TimeScaler
from mvswm.utils.colours import *

TIME_RANGE = TimeRange("2011-03-28 06:00", "2011-03-29 09:00")

GAP_LENGTH = dt.timedelta(hours=5)

COMPONENTS = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]

SEED = 0


def main() -> None:

    fig = plt.figure(figsize=(8, 9))
    _gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])
    fig.subplots_adjust(wspace=0.25)

    data_axes: Dict[str, Axes] = {}
    taylor_axes: Dict[str, Axes] = {}

    ts_labels = "abcd"
    ty_labels = "efgh"

    for i in range(len(COMPONENTS)):
        data_axes[COMPONENTS[i]] = fig.add_subplot(_gs[i, 0])
        taylor_axes[COMPONENTS[i]] = fig.add_subplot(_gs[i, 1], projection="polar")

        format_taylor(taylor_axes[COMPONENTS[i]])

        taylor_axes[COMPONENTS[i]].tick_params(
            labelleft=False, labelright=True, labeltop=False, labelbottom=True
        )

        _t, _, _ = taylor_axes[COMPONENTS[i]].get_yaxis_text1_transform(-30)
        taylor_axes[COMPONENTS[i]].text(
            np.arccos(0),
            0.5,
            "$\sigma_p / \sigma_d$",
            ha="right",
            va="center",
            rotation=90,
            transform=_t,
        )

        taylor_axes[COMPONENTS[i]].text(
            0.7,
            2,
            "$r_p$",
            clip_on=False,
            rotation=-45,
            ha="center",
            va="center",
        )

        # Panel labels
        data_axes[COMPONENTS[i]].text(
            0.01,
            0.9,
            f"({ts_labels[i]})",
            transform=data_axes[COMPONENTS[i]].transAxes,
        )
        taylor_axes[COMPONENTS[i]].text(
            -0.45,
            0.95,
            f"({ty_labels[i]})",
            transform=taylor_axes[COMPONENTS[i]].transAxes,
        )

    data = get_messenger_solar_wind_data(
        TIME_RANGE, bow_shock_buffer=dt.timedelta(minutes=10)
    )

    # TAYLOR DIAGRAM
    # ==============
    # Now do all the modelling and plotting for the Taylor diagram.
    ###############################################################

    taylor_points = {}
    for c in COMPONENTS:
        taylor_points[c] = {
            "GPR Correlation": [],
            "GPR Deviation": [],
            "LI Correlation": [],
            "LI Deviation": [],
        }

    n_taylor_points = 10
    for taylor_index in tqdm(range(n_taylor_points), desc="Training models"):

        # To keep all remaining data for training, we only create one random
        # gap at a time.
        taylor_data, gap_data_list = add_random_gaps(data, 1, seed=SEED + taylor_index)
        truth_data = gap_data_list[0]

        # Train model and LI
        model_predictions = get_model_predictions(truth_data, taylor_data)
        li_predictions = get_linear_interpolation(truth_data, taylor_data)

        # Get correlation per component
        for c in COMPONENTS:

            ax = taylor_axes[c]

            model_correlation = linregress(model_predictions[c], truth_data[c]).rvalue

            # It is not exactly clear how to determine GPR deviation. Is it the
            # mean sqrt of the model variance, is it the std of the mean
            # prediction?

            # M. Rutala (personal commmunication) suggests taking the
            # root-sum-square of the SD of the GPR mean and the point-wise SD
            # of the GPR.
            # CHECK WITH MATT THAT THIS IS CORRECT!
            model_deviation = np.sqrt(
                model_predictions[c].to_numpy().std() ** 2
                + model_predictions[c + " Variance"].to_numpy().mean()
            ) / np.std(truth_data[c].to_numpy())

            li_correlation = linregress(li_predictions[c], truth_data[c]).rvalue
            li_deviation = np.std(li_predictions[c].to_list()) / np.std(
                truth_data[c].to_list()
            )
            taylor_points[c]["GPR Correlation"].append(model_correlation)
            taylor_points[c]["GPR Deviation"].append(model_deviation)

            taylor_points[c]["LI Correlation"].append(li_correlation)
            taylor_points[c]["LI Deviation"].append(li_deviation)

            ax.scatter(
                np.arccos(model_correlation), model_deviation, marker="o", color=GREEN
            )
            ax.scatter(
                np.arccos(li_correlation), li_deviation, marker="x", color=ORANGE
            )

    for c in COMPONENTS:

        ax = taylor_axes[c]

        plot_kde(
            ax,
            taylor_points[c]["GPR Correlation"],
            taylor_points[c]["GPR Deviation"],
            color=GREEN,
        )
        plot_kde(
            ax,
            taylor_points[c]["LI Correlation"],
            taylor_points[c]["LI Deviation"],
            color=ORANGE,
        )

    # Print a summary of metrics
    for c in COMPONENTS:
        print("")
        print(f"{c} Metrics")
        print("GPR")
        print(
            f"    Correlation: {np.mean(taylor_points[c]['GPR Correlation']):.2f} +/- {np.std(taylor_points[c]['GPR Correlation']):.2f}"
        )
        print(
            f"    Deviation: {np.mean(taylor_points[c]['GPR Deviation']):.2f} +/- {np.std(taylor_points[c]['GPR Deviation']):.2f}"
        )
        print("LI")
        print(
            f"    Correlation: {np.mean(taylor_points[c]['LI Correlation']):.2f} +/- {np.std(taylor_points[c]['LI Correlation']):.2f}"
        )
        print(
            f"    Deviation: {np.mean(taylor_points[c]['LI Deviation']):.2f} +/- {np.std(taylor_points[c]['LI Deviation']):.2f}"
        )

    # TIME SERIES
    # ===========
    # First do all the data handelling and plotting for the time series column.
    ###########################################################################
    for i in range(len(COMPONENTS)):

        ax = data_axes[COMPONENTS[i]]

        # Plot real gaps. We have to do this in its own loop before we create
        # the artificial gaps due to the way I created the `get_real_gaps`
        # function.
        _real_gaps_labelled = False
        for gap_start, gap_end in get_real_gaps(data):
            ax.axvspan(
                gap_start,
                gap_end,
                color=PINK,
                alpha=0.5,
                label="Real Gaps" if i == 0 and not _real_gaps_labelled else "",
            )
            _real_gaps_labelled = True

    # Remove the gap and mark the region
    plotting_data, gap_data_list = add_random_gaps(data, 1, seed=SEED)

    # We want to run our model and a linear interpolation over this plot.
    # Though our metrics will come from other randomly created gaps. This is
    # just for the sake of plotting.
    model_predictions_list = [
        get_model_predictions(gd, plotting_data) for gd in gap_data_list
    ]
    linear_interpolations_list = [
        get_linear_interpolation(gd, plotting_data) for gd in gap_data_list
    ]

    for i in range(len(COMPONENTS)):

        ax = data_axes[COMPONENTS[i]]

        # Plot artificial gaps
        _artificial_gap_labelled = False
        for gap_data in gap_data_list:
            ax.axvspan(
                gap_data["UTC"][0],
                gap_data["UTC"][-1],
                color=BLACK,
                alpha=0.05,
                label=(
                    "Artificial Gap" if i == 0 and not _artificial_gap_labelled else ""
                ),
            )
            _artificial_gap_labelled = True

            ax.plot(
                gap_data["UTC"],
                gap_data[COMPONENTS[i]],
                color=BLACK,
                alpha=0.2,
            )

        # Plot remaining data
        # Plot the gap data as faded dots
        ax.plot(
            plotting_data["UTC"],
            plotting_data[COMPONENTS[i]],
            color=BLACK,
        )

        # Plot model predictions
        _predictions_labelled = False
        for model_predictions, linear_interpolations in zip(
            model_predictions_list, linear_interpolations_list
        ):

            ax.plot(
                linear_interpolations["UTC"],
                linear_interpolations[COMPONENTS[i]],
                color=ORANGE,
                solid_capstyle="round",
                label=(
                    "Linear Interpolation"
                    if i == 0 and not _predictions_labelled
                    else ""
                ),
                lw=3,
                path_effects=[  # Add a black outline to the line
                    matplotlib.patheffects.Stroke(linewidth=3.1, foreground="black"),
                    matplotlib.patheffects.Normal(),
                ],
            )

            ax.plot(
                model_predictions["UTC"],
                model_predictions[COMPONENTS[i]],
                color=GREEN,
                solid_capstyle="round",
                label="GPR + 95% CI" if i == 0 and not _predictions_labelled else "",
                lw=3,
                path_effects=[  # Add a black outline to the line
                    matplotlib.patheffects.Stroke(linewidth=3.1, foreground="black"),
                    matplotlib.patheffects.Normal(),
                ],
            )
            _predictions_labelled = True

            upper = (
                model_predictions[COMPONENTS[i]]
                + 1.96 * model_predictions[COMPONENTS[i] + " Variance"].sqrt()
            )
            lower = (
                model_predictions[COMPONENTS[i]]
                - 1.96 * model_predictions[COMPONENTS[i] + " Variance"].sqrt()
            )

            ax.fill_between(
                model_predictions["UTC"],
                lower,
                upper,
                color=GREEN,
                alpha=0.2,
            )

        data_max = np.nanmax(plotting_data["|B| [nT]"].to_numpy())
        if "|B|" not in COMPONENTS[i]:
            ax.axhline(0, color="grey", ls="dashed", zorder=-1)
            ax.set_ylim(-data_max, data_max)

        else:
            ax.set_ylim(0, 2 * data_max)

        ax.xaxis.set_major_locator(HourLocator(byhour=[0, 6, 12, 18]))

        # FIRST ROW
        if i == 0:
            ax.legend(ncol=2)

        # OTHERS
        else:
            pass

        # LAST ROW
        if i != len(COMPONENTS) - 1:
            ax.set_xticklabels([])

        # OTHERS
        else:
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))

        ax.margins(x=0)
        ax.set_ylabel(COMPONENTS[i])

    plt.savefig(
        "./figures/example-messenger-gpr.pdf", format="pdf", bbox_inches="tight"
    )


def prediction_to_taylor(
    ax: Axes, prediction_data: pl.DataFrame, truth_data: pl.DataFrame, **kwargs
) -> None:

    correlations: Dict[str, float] = {}
    deviations: Dict[str, float] = {}

    # Get correlation per component
    for c in COMPONENTS:
        correlations[c] = linregress(prediction_data[c], truth_data[c]).rvalue
        deviations[c] = np.std(prediction_data[c].to_list()) / np.std(
            truth_data[c].to_list()
        )

        if correlations[c] < 0:
            print(f"WARNING: Negative correlation for parameter {c}")

        ax.scatter(np.arccos(correlations[c]), deviations[c], color=BLACK, **kwargs)


def get_model_predictions(
    gap_data: pl.DataFrame, remaining_data: pl.DataFrame
) -> pl.DataFrame:

    # print(f"Began training at {dt.datetime.now()}")

    model_predictions = pl.DataFrame({"UTC": gap_data["UTC"]})

    for c in COMPONENTS:
        # Reshape data for model
        valid = remaining_data.filter(pl.col(c).is_not_nan() & pl.col(c).is_not_null())
        X: NDArray = valid["UTC"].to_numpy().reshape(-1, 1)
        Y: NDArray = valid[c].to_numpy().reshape(-1, 1).astype("float64")

        time_scaler = TimeScaler(X)

        # Create model
        model = SolarWindModel.build(X, Y, time_scaler, RationalQuadratic())
        model.train_model()

        # Make predictions over the gaps
        x_range = model.time_scaler.time_to_numeric(
            gap_data["UTC"].to_numpy().reshape(-1, 1)
        )

        y_mean: tf.Tensor
        y_var: tf.Tensor
        y_mean, y_var = model.model.predict_y(x_range)

        model_predictions = model_predictions.with_columns(
            pl.Series(c, y_mean.numpy().reshape(-1)),
            pl.Series(c + " Variance", y_var.numpy().reshape(-1)),
        )

    return model_predictions


def get_linear_interpolation(
    gap_data: pl.DataFrame, full_data: pl.DataFrame
) -> pl.DataFrame:

    linear_interpolation = pl.DataFrame({"UTC": gap_data["UTC"]})

    # Numpy interpolation needs numerical time
    gap_time = gap_data["UTC"].cast(pl.Int64).to_numpy()

    for c in COMPONENTS:
        # Our data has nans over the gaps for easy plotting. These must be
        # removed to interpolate.
        valid = full_data.filter(pl.col(c).is_not_nan() & pl.col(c).is_not_null())
        data_time = valid["UTC"].cast(pl.Int64).to_numpy()

        linear_interpolation = linear_interpolation.with_columns(
            pl.Series(np.interp(gap_time, data_time, valid[c].to_numpy())).alias(c)
        )

    return linear_interpolation


def add_random_gaps(
    data: pl.DataFrame,
    n_gaps: int,
    min_gap_duration: dt.timedelta = GAP_LENGTH,
    max_gap_duration: dt.timedelta = GAP_LENGTH,
    seed: int | None = None,
) -> Tuple[pl.DataFrame, List[pl.DataFrame]]:
    """
    Add n_gaps randomly-placed, non-overlapping gaps to `data`, avoiding
    any region that already contains null values in any data column.

    Returns the fully gapped DataFrame plus a list of the extracted
    gap DataFrames (in chronological order), one per call to add_gap().
    """
    rng = random.Random(seed)

    value_cols = [c for c in data.columns if c != "UTC"]

    start_time = data["UTC"][0]
    end_time = data["UTC"][-1]
    total_seconds = (end_time - start_time).total_seconds()

    min_s = min_gap_duration.total_seconds()
    max_s = max_gap_duration.total_seconds()
    if total_seconds <= max_s:
        raise ValueError("Data time range is too short for the requested gap duration")

    # Precompute a per-row flag: True if any value column is already null.
    # Used to reject candidate gaps that would overlap pre-existing nulls.
    null_flagged = data.with_columns(
        pl.any_horizontal([pl.col(c).is_null() for c in value_cols]).alias("_has_null")
    ).select(["UTC", "_has_null"])

    def overlaps_existing_null(gap_start, gap_end) -> bool:
        subset = null_flagged.filter(pl.col("UTC").is_between(gap_start, gap_end))
        if subset.height == 0:
            # No data points at all in this window — treat as invalid rather
            # than silently accepting an empty gap.
            return True
        return subset["_has_null"].any()

    # Randomly sample non-overlapping (start, end) intervals within the data
    # range that also avoid pre-existing null regions.
    intervals: List[Tuple] = []
    max_attempts = n_gaps * 100
    attempts = 0
    while len(intervals) < n_gaps and attempts < max_attempts:
        attempts += 1
        duration_s = rng.uniform(min_s, max_s)
        offset_s = rng.uniform(0, total_seconds - duration_s)
        gap_start = start_time + dt.timedelta(seconds=offset_s)
        gap_end = gap_start + dt.timedelta(seconds=duration_s)

        overlaps_gap = any(not (gap_end < s or gap_start > e) for s, e in intervals)
        if overlaps_gap:
            continue

        if overlaps_existing_null(gap_start, gap_end):
            continue

        intervals.append((gap_start, gap_end))

    if len(intervals) < n_gaps:
        raise RuntimeError(
            f"Could only place {len(intervals)} non-overlapping, null-free gap(s) "
            f"out of {n_gaps} requested — try fewer gaps, shorter durations, "
            f"or a longer/cleaner data range."
        )

    # Chronological order just for readability of the output list
    intervals.sort(key=lambda iv: iv[0])

    remaining_data = data
    gap_dataframes: List[pl.DataFrame] = []
    for gap_start, gap_end in intervals:
        gap_time_range = TimeRange(gap_start, gap_end)
        remaining_data, gap_data = add_gap(remaining_data, gap_time_range)
        gap_dataframes.append(gap_data)

    return remaining_data, gap_dataframes


def add_gap(
    data: pl.DataFrame, gap_time_range: TimeRange
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    gap_start = gap_time_range.start.to_datetime()
    gap_end = gap_time_range.end.to_datetime()
    # Guard against gap being outside range
    if gap_end < data["UTC"][0] or gap_start > data["UTC"][-1]:
        raise ValueError("Can't add gap outside range of data")
    remaining_data = data.with_columns(
        [
            # Loop through each column and set to nan if within gap time range
            pl.when(pl.col("UTC").is_between(gap_start, gap_end))
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in data.columns
            if c != "UTC"
        ]
    )
    gap_data = data.filter(pl.col("UTC").is_between(gap_start, gap_end))
    return remaining_data, gap_data


def get_real_gaps(
    data: pl.DataFrame, time_column="UTC", data_column="|B| [nT]"
) -> List[Tuple[Any, Any]]:

    # Treat both polars null and float NaN as "missing"
    is_missing_expr = pl.col(data_column).is_null()
    if data.schema[data_column] in (pl.Float32, pl.Float64):
        is_missing_expr = is_missing_expr | pl.col(data_column).is_nan()

    tagged = data.select(
        pl.col(time_column),
        is_missing_expr.alias("_is_missing"),
    ).with_row_index("_idx")

    # group id increments every time _is_missing changes value
    tagged = tagged.with_columns(
        (pl.col("_is_missing") != pl.col("_is_missing").shift(1).fill_null(False))
        .cum_sum()
        .alias("_group")
    )

    spans = (
        tagged.filter(pl.col("_is_missing"))
        .group_by("_group", maintain_order=True)
        .agg(
            pl.col(time_column).first().alias("start"),
            pl.col(time_column).last().alias("end"),
        )
        .sort("start")
    )

    return list(zip(spans["start"].to_list(), spans["end"].to_list()))


def format_taylor(ax: Axes) -> None:
    correlation_ticks = np.array(
        # [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
        [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
    )
    theta_positions = np.arccos(correlation_ticks)

    # Configure the look of the Taylor plot
    ax.set(
        thetamin=0,
        thetamax=90,
        xticks=theta_positions,
        xticklabels=correlation_ticks,
        axisbelow=True,
    )

    # Add thicker line at std = 1
    ax.axhline(1, color="black")

    # RMSE contours (concentric circles centered on the reference point)
    # Reference point is at (theta=0, r=1), i.e. correlation=1, sigma/sigma_d=1
    theta_grid = np.linspace(0, np.pi, 200)
    r_grid = np.linspace(0, 2, 200)
    T, R = np.meshgrid(theta_grid, r_grid)
    RMSE = np.sqrt(1 + R**2 - 2 * R * np.cos(T))

    rmse_levels = np.linspace(0.2, 2, 6)
    ax.contour(
        T,
        R,
        RMSE,
        levels=rmse_levels,
        colors="gray",
        linestyles="dashed",
        linewidths=0.5,
        zorder=-2,
    )

    ax.set_ylim(0, 1.5)


def plot_kde(
    ax: Axes,
    correlations: List[float],
    deviations: List[float],
    color: str,
    coverage=0.5,
    alpha=0.3,
    grid_size=200,
):
    """
    Overlay a shaded 'coverage'-probability KDE region on a Taylor diagram (polar axes).

    corr : array-like of correlation values
    std  : array-like of standard deviation values
    coverage : float, fraction of probability mass to enclose (0.5 = 50% HDR)
    """
    corr = np.asarray(correlations, dtype=float)
    std = np.asarray(deviations, dtype=float)

    # Convert to taylor coordinates
    theta = np.arccos(corr)
    r = std

    # Drop any remaining NaN/inf pairs. These can occur if the GPR fails to
    # optimise.
    mask = np.isfinite(theta) & np.isfinite(r)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"plot_kde_region: dropping {n_dropped} non-finite point(s)")
    theta, r = theta[mask], r[mask]

    if len(theta) < 3:
        print("plot_kde_region: not enough points for a KDE, skipping")
        return None

    # For the KDE we need to work in a cartesian space
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy = np.vstack([x, y])

    kde = gaussian_kde(xy)

    # Build an evaluation grid directly in polar space
    theta_grid = np.linspace(0, np.pi / 2, grid_size)
    r_max = ax.get_ylim()[1]
    r_grid = np.linspace(0, r_max, grid_size)
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)

    # Convert grid to Cartesian
    x_mesh = r_mesh * np.cos(theta_mesh)
    y_mesh = r_mesh * np.sin(theta_mesh)
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    density = kde(positions).reshape(theta_mesh.shape)

    # Find the density threshold based on a fraction of the mass Sort densities
    # descending, accumulate mass (weighted by grid cell area in Cartesian
    # terms, approximated here via r since polar cells scale with r)
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
    ax.contourf(
        theta_mesh,
        r_mesh,
        density,
        levels=[threshold, density.max()],
        colors=[color],
        alpha=alpha,
    )

    # Black outline
    cs = ax.contour(
        theta_mesh,
        r_mesh,
        density,
        levels=[threshold],
        colors="black",
        linewidths=0.5,
    )

    return cs


if __name__ == "__main__":
    main()
