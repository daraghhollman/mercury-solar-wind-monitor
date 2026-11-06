"""
Take a selection of data from the evalutation data set, and test applying a model.
"""

from mvswm.utils.colours import RED
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpflow.kernels import RationalQuadratic
from matplotlib.axes import Axes
from numpy.typing import NDArray
from tqdm import tqdm

from mvswm.model import GapManager, SolarWindModel, TimeScaler
from mvswm.utils.colours import GREEN

SAVE_DIR = Path(__file__).parents[2] / "data/"

# We need chunks longer than the synodic solar rotation if we want to capture
# any periodicities. Converted to minutes, the data cadence
# CHUNK_LENGTH = round(dt.timedelta(days=1.5 * 30).total_seconds() / 60)
CHUNK_LENGTH = round(dt.timedelta(days=0.5).total_seconds() / 60)
MAX_CHUNK_POINTS = 100  # Arbitrarilly defined, used to tweak memory limitations.

PREDICTION_CADENCE = dt.timedelta(minutes=1)


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
    Y_train: NDArray = training_data.drop_nulls()["|B| [nT]"].to_numpy().reshape(-1, 1)

    X_test: NDArray = testing_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
    Y_test: NDArray = testing_data.drop_nulls()["|B| [nT]"].to_numpy().reshape(-1, 1)

    chunk_slices = get_chunk_slices(len(X_train), CHUNK_LENGTH)

    chunk_predictions: list[ModelOutput] = []
    for i, this_chunk in tqdm(enumerate(chunk_slices), desc="Processing data chunk"):

        X_train_chunk = X_train[this_chunk]
        Y_train_chunk = Y_train[this_chunk]

        # Time bounds of this chunk, defined by the training data
        chunk_times = X_train_chunk[:, 0]
        t_start = chunk_times[0]
        t_end = chunk_times[-1]

        # Only keep test points that fall inside the chunk's time window
        test_mask = (X_test[:, 0] >= t_start) & (X_test[:, 0] <= t_end)
        X_test_chunk = X_test[test_mask]
        Y_test_chunk = Y_test[test_mask]

        print("")
        print(f"Chunk size: {len(X_train_chunk)}")

        n_inducing_points = min(MAX_CHUNK_POINTS, len(X_train_chunk))
        print(f"Inducing points: {n_inducing_points}")

        time_scaler = TimeScaler(X_train_chunk)

        # Create model
        model = SolarWindModel.build(
            input=X_train_chunk,
            output=Y_train_chunk,
            time_scaler=time_scaler,
            kernel=RationalQuadratic(),
            n_inducing_points=n_inducing_points,
        )

        # Fit to data
        model.train_model()

        # Get mean and variance for full range.
        cadence = np.timedelta64(PREDICTION_CADENCE)
        t_range = np.arange(t_start, t_end + cadence, cadence)

        x_range = model.time_scaler.time_to_numeric(t_range[:, None])

        y_mean, y_var = model.model.predict_y(x_range)

        inputs_dict = {
            "X Train": X_train_chunk.ravel(),
            "Y Train": Y_train_chunk.ravel(),
            "X Test": X_test_chunk.ravel(),
            "Y Test": Y_test_chunk.ravel(),
        }

        outputs_dict = {
            "UTC": t_range,
            "|B| [nT]": y_mean.numpy().ravel(),
            "Var(|B|)": y_var.numpy().ravel(),
        }

        results = ModelOutput(outputs_dict, inputs_dict, model)
        chunk_predictions.append(results)

        # single_chunk_plot(results)

        if i == 1:
            break

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

    ax.scatter(
        inputs["X Train"], inputs["Y Train"], marker=".", color="black", zorder=5
    )
    ax.scatter(
        inputs["X Test"],
        inputs["Y Test"],
        marker=".",
        color=RED,
        alpha=0.5,
        zorder=4,
    )

    ax.plot(outputs["UTC"], outputs["|B| [nT]"], color=GREEN)

    upper = outputs["|B| [nT]"] + 1.95 * np.sqrt(outputs["Var(|B|)"])
    lower = outputs["|B| [nT]"] - 1.95 * np.sqrt(outputs["Var(|B|)"])

    ax.fill_between(outputs["UTC"], lower, upper, color=GREEN, alpha=0.3)


def combined_chunk_plot(multiple_chunk_results: list[ModelOutput]) -> None:

    fig, ax = plt.subplots()

    for results in multiple_chunk_results:
        single_chunk_plot(results, ax)


if __name__ == "__main__":
    main()
