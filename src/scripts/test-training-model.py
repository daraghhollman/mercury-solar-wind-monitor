"""
Take a selection of data from the evalutation data set, and test applying a model.
"""

import datetime as dt
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpflow.kernels import RationalQuadratic
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from numpy.typing import NDArray
from tqdm import tqdm

from mvswm.model import GapManager, SolarWindModel, TimeScaler
from mvswm.utils.colours import BLACK, BLUE, GREEN, RED

SAVE_DIR = Path(__file__).parents[2] / "data/"

# We need chunks longer than the synodic solar rotation if we want to capture
# any periodicities. Converted to minutes, the data cadence
# CHUNK_LENGTH = round(dt.timedelta(days=1.5 * 30).total_seconds() / 60)
CHUNK_LENGTH = round(dt.timedelta(days=0.5).total_seconds() / 60)
MAX_CHUNK_POINTS = 300  # Arbitrarilly defined, used to tweak memory limitations.

PREDICTION_CADENCE = dt.timedelta(minutes=1)

Y_VARS = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]

Y_COLOURS = {
    "|B| [nT]": BLACK,
    "Br [nT]": RED,
    "Bt [nT]": GREEN,
    "Bn [nT]": BLUE,
}


def main() -> None:

    # Load data
    data = pl.read_csv(SAVE_DIR / "evaluation-data.csv", try_parse_dates=True)

    # For this test, we will just use the first Mercury year. 88 days.
    data = data.filter(
        pl.col("UTC").is_between(pl.datetime(2008, 8, 1), pl.datetime(2008, 10, 28))
    )

    # Add gaps
    gap_manager = GapManager.from_constants(
        gap_size_minutes=3 * 60,
        gap_interval_minutes=5 * 60,
    )
    training_data, testing_data = gap_manager.split_data(data)

    X_train: NDArray = training_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
    R_train: NDArray = (
        training_data.drop_nulls()["Radius [au]"]
        .to_numpy()
        .astype(float)
        .reshape(-1, 1)
    )
    Y_train: dict[str, NDArray] = {
        y_var: training_data.drop_nulls()[y_var].to_numpy().reshape(-1, 1)
        for y_var in Y_VARS
    }

    X_test: NDArray = testing_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
    R_test: NDArray = (
        testing_data.drop_nulls()["Radius [au]"].to_numpy().astype(float).reshape(-1, 1)
    )
    Y_test: dict[str, NDArray] = {
        y_var: testing_data.drop_nulls()[y_var].to_numpy().reshape(-1, 1)
        for y_var in Y_VARS
    }

    chunk_slices = get_chunk_slices(len(X_train), CHUNK_LENGTH)
    chunk_predictions: list[ModelOutput] = []

    # We can use product to flatten a for looop in chunks and a second in MAG
    # component into a single loop. This gives more frequent updates to tqdm.
    runs = list(product(enumerate(chunk_slices), Y_VARS))
    for (i, this_chunk), y_var in tqdm(
        runs,
        desc="Fitting models",
        unit="model",
    ):
        if i == 3:
            break

        X_train_chunk = X_train[this_chunk]
        R_train_chunk = R_train[this_chunk]
        Y_train_chunk = Y_train[y_var][this_chunk]

        # Time bounds of this chunk, defined by the training data
        chunk_times = X_train_chunk[:, 0]
        t_start = chunk_times[0]
        t_end = chunk_times[-1]

        # Only keep test points that fall inside the chunk's time window
        test_mask = (X_test[:, 0] >= t_start) & (X_test[:, 0] <= t_end)
        X_test_chunk = X_test[test_mask]
        R_test_chunk = R_test[test_mask]
        Y_test_chunk = Y_test[y_var][test_mask]

        print("")
        print(f"Chunk size: {len(X_train_chunk)}")

        n_inducing_points = min(MAX_CHUNK_POINTS, len(X_train_chunk))
        print(f"Inducing points: {n_inducing_points}")

        time_scaler = TimeScaler(X_train_chunk)

        # Create model
        model = SolarWindModel.build(
            times=X_train_chunk,
            distances=R_train_chunk,
            output=Y_train_chunk,
            time_scaler=time_scaler,
            kernel=RationalQuadratic(),
            n_inducing_points=n_inducing_points,
            seed=0,
        )

        # Fit to data
        model.train_model()

        # Get mean and variance for full range.
        cadence = np.timedelta64(PREDICTION_CADENCE)
        t_range = np.arange(t_start, t_end + cadence, cadence)

        # We need the heliocentric distances for the whole range to make
        # predictions.
        src_t = (
            data.drop_nulls()["UTC"].to_numpy().astype("datetime64[ns]").astype("int64")
        )
        src_r = data.drop_nulls()["Radius [au]"].to_numpy().astype(float)
        grid_t = t_range.astype("datetime64[ns]").astype("int64")

        r_range = np.interp(grid_t, src_t, src_r).reshape(-1, 1)

        # Stack [time, radius] to match the (N, 2) training inputs
        t_numeric = model.time_scaler.time_to_numeric(t_range[:, None])
        x_range = np.hstack([t_numeric, r_range])

        y_mean, y_variance = model.model.predict_y(x_range)

        inputs_dict = {
            "X Train": X_train_chunk.ravel(),
            "R Train": R_train_chunk.ravel(),
            "Y Train": Y_train_chunk.ravel(),
            "X Test": X_test_chunk.ravel(),
            "R Test": R_test_chunk.ravel(),
            "Y Test": Y_test_chunk.ravel(),
        }

        outputs_dict = {
            "UTC": t_range,
            "Radius": r_range.ravel(),
            f"{y_var}": y_mean.numpy().ravel(),
            f"Var({y_var})": y_variance.numpy().ravel(),
        }

        results = ModelOutput(outputs_dict, inputs_dict, model)
        chunk_predictions.append(results)

        # TODO: Add heliocentric distance as parameter
        # TODO: Add metrics calculation
        # TODO: Add linear interpolation to outputs_dict
        # TODO: Save data to file
        # TODO: Add loader function to concat

    combined_chunk_plot(chunk_predictions)
    plt.show()


@dataclass
class ModelOutput:
    outputs: dict[str, Any]
    inputs: dict[str, Any]
    model: SolarWindModel


def get_chunk_slices(
    total_length: int, chunk_length: int, chunk_overlap: int = 0
) -> list[slice]:
    return [
        slice(i, min(i + chunk_length, total_length))
        for i in range(0, total_length, chunk_length - chunk_overlap)
    ]


def single_chunk_plot(chunk_results: ModelOutput, ax: Axes | None = None) -> None:

    if ax is None:
        fig, ax = plt.subplots()

    inputs = chunk_results.inputs
    outputs = chunk_results.outputs

    # We don't know which variable is part of this individual model output, so
    # we try them all.
    for i, y in enumerate(Y_VARS):
        try:
            ax.plot(
                outputs["UTC"],
                outputs[y],
                color=Y_COLOURS[y],
                label=y[:-4],
                zorder=5 - i,
            )

            upper = outputs[y] + 1.95 * np.sqrt(outputs[f"Var({y})"])
            lower = outputs[y] - 1.95 * np.sqrt(outputs[f"Var({y})"])

            ax.fill_between(
                outputs["UTC"],
                lower,
                upper,
                color=Y_COLOURS[y],
                alpha=0.3,
                zorder=6 - i,
            )

            ax.scatter(
                inputs["X Train"],
                inputs["Y Train"],
                marker=".",
                color=Y_COLOURS[y],
                zorder=5 - i,
            )
            ax.scatter(
                inputs["X Test"],
                inputs["Y Test"],
                marker=".",
                color=Y_COLOURS[y],
                alpha=0.2,
                zorder=5 - i,
            )

        except KeyError:
            continue


def combined_chunk_plot(multiple_chunk_results: list[ModelOutput]) -> None:
    """
    Input is a list of ModelOutput where every 4 models correspond to the same
    period of time, but for |B|, Br, Bt, Bn individually.
    """

    fig, ax = plt.subplots(layout="constrained")

    for results in multiple_chunk_results:
        single_chunk_plot(results, ax)

    ax.set_ylabel("[nT]")

    locator = AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        ncol=4,
    )


if __name__ == "__main__":
    main()
