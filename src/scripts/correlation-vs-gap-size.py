"""
A script to compare the effect of varied gap sizes on model correlation.
"""

import datetime as dt
import random
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from gpflow.kernels import RationalQuadratic
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import linregress
from sunpy.time import TimeRange
from tqdm import tqdm

from mvswm.data import get_messenger_solar_wind_data
from mvswm.model import SolarWindModel, TimeScaler
from mvswm.utils.colours import *
from mvswm.utils.colours import GREEN, ORANGE

TIME_RANGE: TimeRange = TimeRange("2011-03-28 06:00", "2011-03-29 09:00")

RUNS_PER_GAP = 10

# Define the lengths of queried gap sizes in minutes
minutes = [15, 60, 120, 180, 240, 300]
GAP_LENGTHS: List[dt.timedelta] = [dt.timedelta(minutes=m) for m in minutes]

COMPONENTS = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]

SEED = 0


def main() -> None:

    data = get_messenger_solar_wind_data(
        TIME_RANGE, bow_shock_buffer=dt.timedelta(minutes=10)
    )

    # An array of N_GAP_LENGTHS x N_COMPONENTS x N_RUNS_PER_GAP
    model_correlations: NDArray[np.float64] = np.zeros(
        (len(GAP_LENGTHS), len(COMPONENTS), RUNS_PER_GAP)
    )
    li_correlations: NDArray[np.float64] = np.zeros(
        (len(GAP_LENGTHS), len(COMPONENTS), RUNS_PER_GAP)
    )

    for length_index, gap_length in tqdm(
        enumerate(GAP_LENGTHS), total=len(GAP_LENGTHS)
    ):

        for run_index in tqdm(range(RUNS_PER_GAP), total=RUNS_PER_GAP, leave=False):

            # To keep all remaining data for training, we only create one random
            # gap at a time.
            data_subset, gap_data_list = add_random_gaps(
                data,
                1,
                min_gap_duration=gap_length,
                max_gap_duration=gap_length,
                seed=SEED + run_index,
            )
            truth_data = gap_data_list[0]

            # Train model and LI
            model_predictions = get_model_predictions(truth_data, data_subset)
            li_predictions = get_linear_interpolation(truth_data, data_subset)

            # Get correlation per component
            for component_index, c in enumerate(COMPONENTS):
                model_correlation = linregress(
                    model_predictions[c], truth_data[c]
                ).rvalue
                li_correlation = linregress(li_predictions[c], truth_data[c]).rvalue

                model_correlations[length_index, component_index, run_index] = (
                    model_correlation
                )
                li_correlations[length_index, component_index, run_index] = (
                    li_correlation
                )

    # Make figure
    fig, axes = plt.subplots(
        len(COMPONENTS), 1, figsize=(4, 6), sharex=True, sharey=True
    )

    # Define a width for the boxplots
    width = 10

    for component_index, (c, ax) in enumerate(zip(COMPONENTS, axes)):
        ax: Axes

        model_data = model_correlations[
            :, component_index, :
        ]  # (N_GAP_LENGTHS, N_RUNS)
        li_data = li_correlations[:, component_index, :]

        model_bp_positions = [m - width / 2 for m in minutes]
        li_bp_positions = [m + width / 2 for m in minutes]

        median_props = {
            "color": "black",
        }

        model_bp = ax.boxplot(
            model_data.T,
            positions=model_bp_positions,
            widths=width,
            patch_artist=True,
            medianprops=median_props,
            label="GPR",
        )
        li_bp = ax.boxplot(
            li_data.T,
            positions=li_bp_positions,
            widths=width,
            patch_artist=True,
            medianprops=median_props,
            label="LI",
        )

        box_alpha = 1
        for box in model_bp["boxes"]:
            box.set_facecolor(GREEN)
            box.set_alpha(box_alpha)

        for box in li_bp["boxes"]:
            box.set_facecolor(ORANGE)
            box.set_alpha(box_alpha)

        ax.set_xticks(minutes)
        ax.set_xticklabels([str(m) for m in minutes])

        ax.set_ylabel(f"$r_p$( {c} )")

        if ax == axes[0]:
            ax.legend(ncols=2)

        if ax == axes[-1]:
            ax.set_xlabel("Gap Length [minutes]")

    fig.savefig(
        "./figures/correlation-vs-gap-size.pdf", format="pdf", bbox_inches="tight"
    )


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
    min_gap_duration: dt.timedelta,
    max_gap_duration: dt.timedelta,
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


if __name__ == "__main__":
    main()
