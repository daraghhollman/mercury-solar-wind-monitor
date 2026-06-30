from typing import Dict
import datetime as dt
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from gpflow.kernels import RationalQuadratic
from numpy.typing import NDArray
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag
from mvswm.model import SolarWindModel, TimeScaler
from mvswm.utils.colours import *


def main() -> None:

    components = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]
    component_colours = [BLACK, RED, GREEN, BLUE]

    # Load some data
    # time_range = TimeRange("2011")
    messenger = Spacecraft("MESSENGER")

    # As an example, lets just work with 10 orbits of MESSENGER data.
    # 10 orbits at a ~12 hour period is roughly 120 hours.
    time_range = TimeRange("2011-03-23 20:00", dt.timedelta(hours=120))
    data = messenger.get_data_in_range(time_range)

    # Remove time within bow shock
    data = filter_messenger_mag(data, buffer=dt.timedelta(minutes=10))

    # Split into 1000-point sections
    split_length: int = 1000
    n_splits: int = len(data) // split_length
    for split_index in range(n_splits):

        split_data = data.slice(split_index * split_length, split_length)

        # Reshape data for model
        component_predictions: Dict[str, Tuple] = {}
        for component in components:
            X: NDArray = split_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
            Y: NDArray = (
                split_data.drop_nulls()[component].to_numpy().reshape(-1, 1).astype("float64")
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
            ax.scatter(split_data["UTC"], split_data[component], marker=".", color=colour)

        # Plot predictions
        for component, colour in zip(components, component_colours):
            this_prediction = component_predictions[component]
            ax.plot(this_prediction[0], this_prediction[1], color=colour)

            y_upper = ( this_prediction[1] + 1.96 * np.sqrt(this_prediction[2]) )[:, 0]
            y_lower = ( this_prediction[1] - 1.96 * np.sqrt(this_prediction[2]) )[:, 0]

            ax.fill_between(this_prediction[0][:,0], y_lower, y_upper, alpha=0.3, color=colour)


        plt.show()


if __name__ == "__main__":
    main()
