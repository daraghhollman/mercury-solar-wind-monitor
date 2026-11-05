"""
Take a selection of data from the evalutation data set, and test applying a model.
"""

import datetime as dt
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from gpflow.kernels import RationalQuadratic
from numpy.typing import NDArray
from tqdm import tqdm

from mvswm.model import GapManager, SolarWindModel, TimeScaler

SAVE_DIR = Path(__file__).parents[2] / "data/"

# We need chunks longer than the synodic solar rotation if we want to capture
# any periodicities. Converted to minutes, the data cadence
# CHUNK_LENGTH = round(dt.timedelta(days=1.5 * 30).total_seconds() / 60)
CHUNK_LENGTH = round(dt.timedelta(days=1).total_seconds() / 60)
MAX_CHUNK_POINTS = 1000  # Arbitrarilly defined, used to tweak memory limitations.

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

    # TODO: needs to be split into smaller sections as it is too much data for
    # memory (though maybe not for server? - can we do some calculations?).

    X: NDArray = training_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
    Y: NDArray = training_data.drop_nulls()["|B| [nT]"].to_numpy().reshape(-1, 1)

    chunk_slices = get_chunk_slices(len(X), CHUNK_LENGTH)
    print(chunk_slices)
    print(len(X))
    for this_chunk in tqdm(chunk_slices, desc="Processing data chunk"):

        time_scaler = TimeScaler(X[this_chunk])

        # Create model
        model = SolarWindModel.build(
            input=X[this_chunk],
            output=Y[this_chunk],
            time_scaler=time_scaler,
            kernel=RationalQuadratic(),
            n_inducing_points=MAX_CHUNK_POINTS,
        )

        # Fit to data
        model.train_model()

        # Get mean and variance for full range.
        t_start = np.datetime64(X[0])
        t_end = np.datetime64(X[-1])

        t_range = np.arange(
            t_start, t_end + np.timedelta64(PREDICTION_CADENCE), np.timedelta64(PREDICTION_CADENCE)
        )

        x_range = model.time_scaler.time_to_numeric(t_range[:, None])

        y_mean, y_var = model.model.predict_y(x_range)


def get_chunk_slices(total_length: int, chunk_length: int, chunk_overlap: int = 0) -> list[slice]:
    return [
        slice(i, min(i + chunk_length, total_length))
        for i in range(0, total_length, chunk_length - chunk_overlap)
    ]


if __name__ == "__main__":
    main()
