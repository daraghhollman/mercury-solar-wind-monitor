from __future__ import annotations

from dataclasses import dataclass

import gpflow
import numpy as np
from gpflow.kernels import Kernel
from gpflow.models import GPR
from gpflow.optimizers import Scipy
from keras.optimizers import Optimizer
from numpy.typing import NDArray
from typing import Any, Tuple

from mvswm.model import TimeScaler

__all__ = [
    "SolarWindModel",
]


# Define a class to hold the model and associated functions
@dataclass
class SolarWindModel:
    model: GPR
    data: Tuple[NDArray[np.datetime64], NDArray[Any]]
    kernel: Kernel
    optimiser: Optimizer
    time_scaler: TimeScaler  # Store scaler on model for later inverse transforms
    seed: int

    @classmethod
    def build(
        cls,
        input: NDArray[np.datetime64],
        output: NDArray[Any],
        time_scaler: TimeScaler,
        kernel: Kernel,
        seed: int = 0,
    ) -> SolarWindModel:
        """
        We choose to define how our model is constructed here so that the
        inital choices are fixed. Any user can override any attributes or
        methods as they see fit. i.e. changing an optimiser, or how the
        training call works.
        """

        # X is a datetime object which we must first convert to being numerical.
        # Additionally, as GP models are based on distance between data, we want to use
        # small numerical values to aid in computation times. For this, we scale the
        # time between 0 and 1.
        X = time_scaler.time_to_numeric(input)
        Y = output

        gpmodel = GPR(
            (X, Y),
            kernel=kernel,
            mean_function=gpflow.mean_functions.Constant(np.mean(output)),
        )

        opt = Scipy()

        return cls(
            model=gpmodel,
            data=(X, Y),
            kernel=kernel,
            optimiser=opt,
            time_scaler=time_scaler,
            seed=seed,
        )

    def train_model(self) -> None:
        """
        Perform iterations of training.
        """

        self.optimiser.minimize(
            self.model.training_loss,
            self.model.trainable_variables,
        )
