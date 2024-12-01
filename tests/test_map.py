import pytest
import numpy as np
from pdistmap.set import KDEIntersection
from pdistmap.map import ClusterMapper

# Mocking the KDEIntersection class to avoid actual computation during tests
class MockKDEIntersection:
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def intersection_area(self):
        # A mock similarity score for testing purposes
        return np.random.random()

# Replace the actual KDEIntersection with the mock
KDEIntersection = MockKDEIntersection


@pytest.fixture
def sample_data():
    """Fixture providing sample input data for testing."""
    Adict = {
        "A_o": [1, 2, 3],
        "B_o": [4, 5, 6],
    }
    Bdict = {
        "A": [1, 2, 3],
        "B": [7, 8, 9],
    }
    return Adict, Bdict


def test_initialization(sample_data):
    """Test the initialization of the ClusterMapper class."""
    Adict, Bdict = sample_data
    cluster_mapper = ClusterMapper(Adict, Bdict)
    
    assert cluster_mapper.Adict == Adict
    assert cluster_mapper.Bdict == Bdict


def test_find_top_n_matches(sample_data):
    """Test the find_top_n_matches method."""
    Adict, Bdict = sample_data
    cluster_mapper = ClusterMapper(Adict, Bdict)

    matches = cluster_mapper.find_top_n_matches(n=1)

    assert len(matches) == len(Adict)  # Ensure we have as many matches as keys in Adict
    for key_A in matches.keys():
        assert len(matches[key_A]) == 1  # Each key should return one match


def test_find_top_n_matches_multiple(sample_data):
    """Test find_top_n_matches with more matches."""
    Adict, Bdict = sample_data
    cluster_mapper = ClusterMapper(Adict, Bdict)

    matches = cluster_mapper.find_top_n_matches(n=2)

    assert len(matches) == len(Adict)  # Ensure we have as many matches as keys in Adict
    for key_A in matches.keys():
        assert len(matches[key_A]) == 2  # Each key should return two matches


def test_invalid_n_value(sample_data):
    """Test handling of invalid 'n' values in find_top_n_matches."""
    Adict, Bdict = sample_data
    cluster_mapper = ClusterMapper(Adict, Bdict)

    with pytest.raises(ValueError):
        cluster_mapper.find_top_n_matches(n=-1)  # Invalid number of matches


if __name__ == "__main__":
    pytest.main()
