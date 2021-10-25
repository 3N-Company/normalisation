import pandas as pd
from typing import Tuple
import numpy as np


class ImageMetadata:
    """
    This class represents the submitted data, that will be further normalised
    """

    def __init__(self, data):
        self.id = data["user"]["id"]
        self.latitude = float(
            data["metadata"]["position"]["latitude"]
        )  # test on empty string
        self.longitude = float(
            data["metadata"]["position"]["longitude"]
        )  # test on empty string
        self.score = False


class Normalisation:
    """
    This class represents the calculation of the IQR method and z- score
    """

    def __init__(self):
        self.user_array = []
        self.latitude_array = []
        self.longitude_array = []
        self.score_array = []

    @staticmethod
    def iqr_lower_uperbound_calculation(
        frame: pd.DataFrame, column: str
    ) -> Tuple[float, float]:
        """
            since we do the quantile over many columns we want only for a single column
            column having back -> therefore [0]
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html
        :param frame:
        :return: upper and lower bounds
        """
        iqr = frame[[column]].quantile(0.75)[0] - frame[[column]].quantile(0.25)[0]
        upper_bound = frame[[column]].quantile(0.75)[0] + 1.5 * iqr
        lower_bound = frame[[column]].quantile(0.25)[0] - 1.5 * iqr
        return upper_bound, lower_bound

    @staticmethod
    def zscore(value: float, mean_frame: float, stand_deviation_frame: float) -> float:
        """
        performs the zscore calculation
        :param value: single value
        :return: calculated value -> if it is
        """
        z = (value - mean_frame) / stand_deviation_frame
        if z < -3 or z > 3:
            print("Z score Reset " + str(z))
            return np.NaN
        return value

    @staticmethod
    def lqr_help_func(x: float, upper_bound: float, lower_bound: float) -> float:
        # performs the evaluation of the lqr
        if x < lower_bound or x > upper_bound:
            print("found NaN IQR")
            x_new = np.NaN
            return x_new
        else:
            return x

    def process_zscore(self, frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Calculate z-score basde on a column value
        Args:
            frame:
            column_name:

        Returns:

        """
        stand_deviation_frame = frame[column_name].std(skipna=True)
        mean_frame = frame[column_name].mean()
        frame[column_name] = frame[column_name].map(
            lambda x: self.zscore(x, mean_frame, stand_deviation_frame)
        )
        return frame
