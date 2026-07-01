import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpflow.kernels import RationalQuadratic
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
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
    n_splits: int = round(len(data) / split_length)
    for split_index in range(n_splits):

        split_data = data.slice(split_index * split_length, split_length)

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

        component_predictions: Dict[str, Tuple] = {}
        for component in components:

            # Reshape data for model
            X: NDArray = training_data.drop_nulls()["UTC"].to_numpy().reshape(-1, 1)
            Y: NDArray = (
                training_data.drop_nulls()[component]
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
        for component, colour in zip(components, component_colours):
            this_prediction = component_predictions[component]
            ax.plot(this_prediction[0], this_prediction[1], color=colour)

            y_upper = (this_prediction[1] + 1.96 * np.sqrt(this_prediction[2]))[:, 0]
            y_lower = (this_prediction[1] - 1.96 * np.sqrt(this_prediction[2]))[:, 0]

            ax.fill_between(
                this_prediction[0][:, 0], y_lower, y_upper, alpha=0.3, color=colour
            )

        plt.show()


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


def get_metrics(
    true_data: pl.DataFrame,
    model_predictions: pl.DataFrame,
    y_variables: List[str] = [
        "|B| [nT]",
        "Bx [nT]",
        "By [nT]",
        "Bz [nT]",
    ],
    metrics: List[Callable] = [
        # A list of functions which all take input: (y_true, y_pred)
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
