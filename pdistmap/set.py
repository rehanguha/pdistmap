import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import Tuple


class KDEIntersection:
    """
    A class used to compute the intersection area between two Kernel Density Estimations (KDEs).
    
    Attributes
    ----------
    A : np.ndarray
        A numeric numpy array representing the first dataset.
    B : np.ndarray
        A numeric numpy array representing the second dataset.
    executed : bool
        A flag to check if the KDE intersection calculation has been executed.
    
    Methods
    -------
    resetVectors(A: np.ndarray, B: np.ndarray) -> None:
        Resets the vectors A and B and sets the execution flag to False.
    
    _validate_numeric_array(arr: np.ndarray) -> np.ndarray:
        Validates that the input is a numpy array and contains only numeric values (integers or floats).
    
    _check_limit(value: float, lower: float, upper: float) -> float:
        Validates whether a value is between the specified limits.
    
    _calculate_kde(data: np.ndarray, bw_method: str = 'scott') -> gaussian_kde:
        Calculates the Kernel Density Estimation (KDE) of the input data.
    
    _min_max_finder(data1: np.ndarray, data2: np.ndarray, adjustment_factor: float = 0) -> Tuple[float, float]:
        Finds the minimum and maximum values from two datasets and adjusts them with a factor.
    
    _build_linespace(xmin: float, xmax: float, linespace_num: int = 10000) -> np.ndarray:
        Builds a linearly spaced array between two values.
    
    intersection_area(adjustment_factor: float = 0.2, bw_method: str = 'scott', linespace_num: int = 10000, plot: bool = False) -> float:
        Calculates the intersection area between the KDEs of A and B.
    
    plot() -> None:
        Plots the KDEs of A and B and their intersection.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """
        Initializes the kde_intersection class with two datasets.

        Parameters
        ----------
        A : np.ndarray
            The first numeric array for KDE computation.
        B : np.ndarray
            The second numeric array for KDE computation.
        """
        self.A = A
        self.B = B
        self.executed = False

    def resetVectors(self, A: np.ndarray, B: np.ndarray) -> None:
        """
        Resets the vectors A and B and sets the execution flag to False.

        Parameters
        ----------
        A : np.ndarray
            The first numeric array for KDE computation.
        B : np.ndarray
            The second numeric array for KDE computation.
        """
        self.A = A
        self.B = B
        self.executed = False

    def _validate_numeric_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Validates that the input is a numpy array and contains only numeric values (integers or floats).
        Raises an error if the array contains non-numeric values or NaN.

        Parameters
        ----------
        arr : np.ndarray
            The array to validate.

        Returns
        -------
        np.ndarray
            The validated array.

        Raises
        ------
        TypeError
            If the input is not a numpy array.
        ValueError
            If the array contains non-numeric data or NaN values.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input is not a numpy array.")

        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                "Array contains non-numeric data. Only integers and floats are allowed."
            )

        if np.isnan(arr).any():
            raise ValueError("Array contains NaN values, which are not allowed.")

        return arr

    def _check_limit(self, value: float, lower: float, upper: float) -> float:
        """
        Validates whether a value is between the specified limits.

        Parameters
        ----------
        value : float
            The value to check.
        lower : float
            The lower bound of the valid range.
        upper : float
            The upper bound of the valid range.

        Returns
        -------
        float
            The validated value.

        Raises
        ------
        ValueError
            If the value is not within the specified range.
        """
        if lower <= value <= upper:
            return value
        else:
            raise ValueError(f"Value should be between [{lower}, {upper}].")

    def _calculate_kde(
        self, data: np.ndarray, bw_method: str = "scott"
    ) -> gaussian_kde:
        """
        Calculates the Kernel Density Estimation (KDE) of the input data.

        Parameters
        ----------
        data : np.ndarray
            The data for which to compute the KDE.
        bw_method : str, optional
            The bandwidth method to use for KDE estimation. Default is 'scott'.

        Returns
        -------
        gaussian_kde
            The KDE function for the input data.
        """
        return gaussian_kde(data, bw_method=bw_method)

    def _min_max_finder(
        self, data1: np.ndarray, data2: np.ndarray, adjustment_factor: float = 0
    ) -> Tuple[float, float]:
        """
        Finds the minimum and maximum values from two datasets, adjusting them with a factor.

        Parameters
        ----------
        data1 : np.ndarray
            The first dataset.
        data2 : np.ndarray
            The second dataset.
        adjustment_factor : float, optional
            Factor to adjust the minimum and maximum range, should be between 0 and 1. Default is 0.

        Returns
        -------
        Tuple[float, float]
            The adjusted minimum and maximum values.
        """
        xmin = min(data1.min(), data2.min())
        xmax = max(data1.max(), data2.max())

        adjustment_factor = self._check_limit(adjustment_factor, lower=0, upper=1)

        dx = adjustment_factor * (xmax - xmin)
        adjusted_xmin = xmin - dx
        adjusted_xmax = xmax + dx

        return adjusted_xmin, adjusted_xmax

    def _build_linespace(
        self, xmin: float, xmax: float, linespace_num: int = 10000
    ) -> np.ndarray:
        """
        Builds a linearly spaced array between two values.

        Parameters
        ----------
        xmin : float
            The minimum value of the line space.
        xmax : float
            The maximum value of the line space.
        linespace_num : int, optional
            The number of points in the line space. Default is 10,000.

        Returns
        -------
        np.ndarray
            The linearly spaced array.
        """
        return np.linspace(xmin, xmax, linespace_num)

    def intersection_area(
        self,
        adjustment_factor: float = 0.2,
        bw_method: str = "scott",
        linespace_num: int = 10000,
        plot: bool = False,
    ) -> float:
        """
        Calculates the intersection area between the KDEs of A and B.

        Parameters
        ----------
        adjustment_factor : float, optional
            Factor to adjust the min and max range of the data. Default is 0.2.
        bw_method : str, optional
            The bandwidth method to use for KDE estimation. Default is 'scott'.
        linespace_num : int, optional
            Number of points in the line space. Default is 10,000.
        plot : bool, optional
            If True, plots the KDEs and their intersection. Default is False.

        Returns
        -------
        float
            The area of intersection between the KDEs of A and B.
        """
        A = self._validate_numeric_array(self.A)
        B = self._validate_numeric_array(self.B)

        kdeA = self._calculate_kde(A, bw_method=bw_method)
        kdeB = self._calculate_kde(B, bw_method=bw_method)

        data_min, data_max = self._min_max_finder(
            A, B, adjustment_factor=adjustment_factor
        )

        self.data_linespace = self._build_linespace(
            data_min, data_max, linespace_num=linespace_num
        )

        self.kdeA_data = kdeA(self.data_linespace)
        self.kdeB_data = kdeB(self.data_linespace)

        self.inters = np.minimum(self.kdeA_data, self.kdeB_data)
        self.area_inters = np.trapezoid(self.inters, self.data_linespace)
        self.executed = True

        if plot:
            self.plot()

        return self.area_inters

    def plot(self) -> None:
        """
        Plots the KDEs of A and B and their intersection. 

        Raises
        ------
        RuntimeError
            If `intersection_area()` has not been called before plotting.
        """
        if not self.executed:
            raise Exception(
                "'intersection_area()' needs to be executed first. Or use 'intersection_area(plot=True)'"
            )

        plt.plot(self.data_linespace, self.kdeA_data, color="b", label="A")
        plt.fill_between(self.data_linespace, self.kdeA_data, 0, color="b", alpha=0.2)
        plt.plot(self.data_linespace, self.kdeB_data, color="orange", label="B")
        plt.fill_between(
            self.data_linespace, self.kdeB_data, 0, color="orange", alpha=0.2
        )
        plt.plot(self.data_linespace, self.inters, color="r")
        plt.fill_between(
            self.data_linespace,
            self.inters,
            0,
            facecolor="none",
            edgecolor="r",
            hatch="x",
            label="Intersection",
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        labels[2] += f": {self.area_inters * 100:.1f} %"
        plt.legend(handles, labels, title="")
        plt.title("KDE Intersection")
        plt.tight_layout()
        plt.show()
