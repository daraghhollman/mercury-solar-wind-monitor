import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


class TimeScaler:

    def __init__(self, fit_data: NDArray) -> None:

        self._dtype = fit_data.dtype
        self._fit_data: NDArray = self._datetime_to_float(fit_data)
        self._scaler: MinMaxScaler = self.init_transform()

    def _datetime_to_float(self, data: NDArray) -> NDArray:
        return data.astype("datetime64[ns]").astype(np.float64)

    def _float_to_datetime(self, data: NDArray) -> NDArray:
        return data.astype("datetime64[ns]").astype(self._dtype)

    def _timedelta_to_float(self, data: NDArray) -> NDArray:
        return data.astype("timedelta64[ns]").astype(np.float64)

    def _float_to_timedelta(self, data: NDArray) -> NDArray:
        return np.rint(data).astype("timedelta64[ns]")

    def init_transform(self) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(self._fit_data)

        return scaler

    def time_to_numeric(self, data: NDArray) -> NDArray:
        data = self._datetime_to_float(data)

        return self._scaler.transform(data)

    def numeric_to_time(self, data: NDArray) -> NDArray:

        data = self._scaler.inverse_transform(data)

        return self._float_to_datetime(data)

    def timedelta_to_numeric(self, data: NDArray) -> NDArray:
        data = self._timedelta_to_float(data)

        return data * self._scaler.scale_

    def numeric_to_timedelta(self, data: NDArray) -> NDArray:
        data = np.asarray(data, dtype=np.float64) / self._scaler.scale_

        return self._float_to_timedelta(data)
