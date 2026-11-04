"""
Take a selection of data from the evalutation data set, and test applying a model.
"""

from gpflow.kernels import RationalQuadratic
from mvswm.model import TimeScaler
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from numpy.typing import NDArray

from mvswm.data import make_gaps_null
from mvswm.model import GapManager, SolarWindModel

SAVE_DIR = Path(__file__).parents[2] / "data/"


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

    time_scaler = TimeScaler(X)

    # Create model
    model = SolarWindModel.build(
        input=X,
        output=Y,
        time_scaler=time_scaler,
        kernel=RationalQuadratic(),
    )

    model.train_model()


if __name__ == "__main__":
    main()
