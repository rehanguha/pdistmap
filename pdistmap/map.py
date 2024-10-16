import numpy as np
from pdistmap.set import KDEIntersection

class ClusterMapper:
    def __init__(self, Adict: dict, Bdict: dict):
        """
        Initialize the ClusterMapper with two dictionaries containing lists.

        Parameters:
        - Adict (dict): A dictionary where each key corresponds to a list of values for comparison.
        - Bdict (dict): A dictionary where each key corresponds to another list of values for comparison.
        """
        self.Adict = Adict
        self.Bdict = Bdict
    
    def _check_valid_n(self, n: int) -> None:
        """
        Checks if the value of n is valid.

        Parameters:
        - n: Number of top matches to validate.

        Raises:
        - ValueError: If n is not greater than 1 or not less than the size of the lists in Adict or Bdict.
        """
        # Ensure n is valid
        if n < 1:
            raise ValueError("n must be greater than 1.")
        
        # Check the size of the lists in Adict
        for key in self.Adict:
            if n >= len(self.Adict[key]):
                raise ValueError(f"n must be less than the size of the list for key '{key}' in Adict.")
        
        # Check the size of the lists in Bdict
        for key in self.Bdict:
            if n >= len(self.Bdict[key]):
                raise ValueError(f"n must be less than the size of the list for key '{key}' in Bdict.")


    def list_similarity(self, list1: list, list2: list) -> float:
        """
        Calculate the similarity between two lists using the KDEIntersection method.

        Parameters:
        - list1 (list): The first list of values.
        - list2 (list): The second list of values.

        Returns:
        - float: A similarity score calculated as the intersection area between the two lists.
        """
        return KDEIntersection(np.array(list1), np.array(list2)).intersection_area()

    def find_top_n_matches(self, n: int = 2) -> dict:
        """
        Find the top 'n' closest matches between the lists of the two dictionaries.

        Parameters:
        - n (int): The number of top matches to return for each key in Adict. Default is 2.

        Returns:
        - dict: A dictionary mapping each key from Adict to its top 'n' matches from Bdict, 
                represented as a list of tuples containing the matching key and its similarity score.
        """

        self._check_valid_n(n)

        matches = {}

        # Iterate over each item in Adict
        for key_A, value_A in self.Adict.items():
            similarities = []

            # Iterate over each item in Bdict to calculate similarity
            for key_B, value_B in self.Bdict.items():
                similarity = self.list_similarity(value_A, value_B)
                similarities.append((key_B, similarity))

            # Sort the similarities and store the top 'n' matches
            top_n_matches = sorted(similarities, key=lambda x: (-x[1], x[0]))[:n]
            matches[key_A] = top_n_matches

        return matches
