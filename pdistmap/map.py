import numpy as np
from pdistmap.set import KDEIntersection

class ClusterMapper:
    """
    A class to map and find the most similar lists between two dictionaries
    using a similarity measure based on Kernel Density Estimation (KDE).

    This class provides a method to find the top `n` closest matches for each key in 
    the dictionary `Adict` by comparing its values with the values of `Bdict`. The 
    similarity is measured using the intersection area between the KDEs of the respective lists.

    Attributes
    ----------
    Adict : dict
        A dictionary where each key corresponds to a list of values for comparison.
    Bdict : dict
        A dictionary where each key corresponds to another list of values for comparison.

    Methods
    -------
    _check_valid_n(n: int) -> None
        Validates the value of `n` to ensure it is within the allowable range.
    _list_similarity(list1: list, list2: list, adjustment_factor: float,
                     bw_method: str, linespace_num: int, scale: bool, plot: bool) -> float
        Computes the similarity score between two lists using KDE intersection.
    find_top_n_matches(n: int = 2, adjustment_factor: float = 0.2,
                       bw_method: str = "scott", linespace_num: int = 10000,
                       scale: bool = False, plot: bool = False) -> dict
        Finds the top `n` closest matches between the lists of two dictionaries.
    """

    def __init__(self, Adict: dict, Bdict: dict):
        """
        Initialize the ClusterMapper with two dictionaries containing lists.

        Parameters
        ----------
        Adict : dict
            A dictionary where each key corresponds to a list of values for comparison.
        Bdict : dict
            A dictionary where each key corresponds to another list of values for comparison.
        """
        self.Adict = Adict
        self.Bdict = Bdict

    def _check_valid_n(self, n: int) -> None:
        """
        Validates the value of `n` to ensure it is within the allowable range.

        Parameters
        ----------
        n : int
            The number of top matches to validate.

        Raises
        ------
        ValueError
            If `n` is less than 1 or greater than the number of keys in `Bdict`.
        """
        if n < 1:
            raise ValueError("`n` must be greater than 0.")
        if n > len(self.Bdict):
            raise ValueError("`n` must not exceed the number of keys in `Bdict`.")

    def _list_similarity(self, list1: list, list2: list, adjustment_factor: float,
                          bw_method: str, linespace_num: int, scale: bool, plot: bool) -> float:
        """
        Calculates the similarity score between two lists using KDEIntersection.

        Parameters
        ----------
        list1 : list
            The first list of numeric values.
        list2 : list
            The second list of numeric values.
        adjustment_factor : float
            A scaling factor to adjust the KDE computation range.
        bw_method : str
            The method used to compute the bandwidth for KDE ('scott' or 'silverman').
        linespace_num : int
            The number of points used to compute the KDE intersection.
        scale : bool
            Whether to normalize the data using min-max scaling before comparison.
        plot : bool
            Whether to generate a plot of the KDE intersection.

        Returns
        -------
        float
            The similarity score, calculated as the area of the intersection between the KDEs of the two lists.
        """
        return KDEIntersection(np.array(list1), np.array(list2)).intersection_area(
            adjustment_factor=adjustment_factor,
            bw_method=bw_method,
            linespace_num=linespace_num,
            scale=scale,
            plot=plot
        )

    def find_top_n_matches(self, n: int = 2, adjustment_factor: float = 0.2,
                           bw_method: str = "scott", linespace_num: int = 10000,
                           scale: bool = False, plot: bool = False) -> dict:
        """
        Finds the top `n` closest matches between the lists of two dictionaries.

        Parameters
        ----------
        n : int, optional
            The number of top matches to return for each key in `Adict` (default is 2).
        adjustment_factor : float, optional
            A scaling factor to adjust the KDE computation range (default is 0.2).
        bw_method : str, optional
            The method used to compute the bandwidth for KDE ('scott' or 'silverman') (default is 'scott').
        linespace_num : int, optional
            The number of points used to compute the KDE intersection (default is 10,000).
        scale : bool, optional
            Whether to normalize the data using min-max scaling before comparison (default is False).
        plot : bool, optional
            Whether to generate plots of the KDE intersections (default is False).

        Returns
        -------
        dict
            A dictionary mapping each key from `Adict` to its top `n` matches from `Bdict`.
            Each match is represented as a list of tuples containing the matching key and its similarity score.

        Raises
        ------
        ValueError
            If `n` is not valid as per `_check_valid_n`.
        """
        self._check_valid_n(n)
        matches = {}

        for key_A, value_A in self.Adict.items():
            similarities = []

            for key_B, value_B in self.Bdict.items():
                similarity = self._list_similarity(value_A, value_B, adjustment_factor,
                                                   bw_method, linespace_num, scale, plot)
                similarities.append((key_B, similarity))

            # Sort by similarity (descending) and then alphabetically by key
            top_n_matches = sorted(similarities, key=lambda x: (-x[1], x[0]))[:n]
            matches[key_A] = top_n_matches

        return matches
