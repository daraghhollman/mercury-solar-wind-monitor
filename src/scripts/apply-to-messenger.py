import datetime as dt
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpflow.kernels import RationalQuadratic
from numpy.typing import NDArray
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag
from mvswm.model import GapManager, SolarWindModel, TimeScaler
from mvswm.utils.colours import *


def main() -> None:

    components = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]
    component_colours = [BLACK, RED, GREEN, BLUE]

    data = get_messenger_solar_wind_data(
        TimeRange("2011-03-23", dt.timedelta(hours=120)),
        bow_shock_buffer=dt.timedelta(minutes=10),
    )

    # GPRs are computationally expensive, scaling with n^3. As a result, it is
    # important to keep input data small. For these reasons, we choose to split
    # the data based on the number of data-points. A reasonable range is between
    # 1k and 10k.
    split_length: int = 1000
    n_splits: int = len(data) // split_length
    for split_index in range(n_splits):

        split_data = data.slice(split_index * split_length, split_length)

        # This data will have some data-gaps inherent to MESSENGER's orbit
        # around Mercury. We also need to add additional artificial data-gaps
        # where we will test the model's performance.

        # GapManager.get_real_gaps
        # GapManager.get_artificial_gaps
        gm = GapManager(split_data, gap_length=dt.timedelta(hours=3))

        # Reshape data for model
        component_predictions: Dict[str, Tuple] = {}
        for component in components:
            X: NDArray = split_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
            Y: NDArray = (
                split_data.drop_nulls()[component]
                .to_numpy()
                .reshape(-1, 1)
                .astype("float64")
            )

            time_scaler = TimeScaler(X)

            print(f"Began training model on {component} at {dt.datetime.now()}")

            # Create model
            model = SolarWindModel.build(
                input=X,
                output=Y,
                time_scaler=time_scaler,
                kernel=RationalQuadratic(),
            )

            model.train_model()

            x_range = np.linspace(0, 1, 10000)[:, None]

            y_mean, y_var = model.model.predict_y(x_range)

            x_out = model.time_scaler.numeric_to_time(x_range)

            component_predictions[component] = (x_out, y_mean, y_var)

        fig, ax = plt.subplots()

        for component, colour in zip(components, component_colours):
            ax.scatter(
                split_data["UTC"], split_data[component], marker=".", color=colour
            )

        # Plot predictions
        for component, colour in zip(components, component_colours):
            this_prediction = component_predictions[component]
            ax.plot(this_prediction[0], this_prediction[1], color=colour)

            y_upper = (this_prediction[1] + 1.96 * np.sqrt(this_prediction[2]))[:, 0]
            y_lower = (this_prediction[1] - 1.96 * np.sqrt(this_prediction[2]))[:, 0]

            ax.fill_between(
                this_prediction[0][:, 0], y_lower, y_upper, alpha=0.3, color=colour
            )

        plt.show()


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
