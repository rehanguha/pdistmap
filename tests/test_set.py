import pytest
import numpy as np
from pdistmap.set import KDEIntersection

# Test cases for valid inputs
def test_intersection_area_valid():
    # Create two valid datasets
    A = np.random.normal(0, 1, 100)
    B = np.random.normal(1, 1, 100)

    # Initialize the kde_intersection object
    kde_obj = KDEIntersection(A, B)

    # Test intersection_area without plotting
    area = kde_obj.intersection_area()
    
    # Check that the intersection area is a float and between 0 and 1
    assert isinstance(area, float), "Intersection area is not a float"
    assert 0 <= area <= 1, "Intersection area is out of bounds"

# Test case for reset functionality
def test_reset_vectors():
    A = np.random.normal(0, 1, 100)
    B = np.random.normal(1, 1, 100)
    kde_obj = KDEIntersection(A, B)
    
    # Calculate the intersection area first
    initial_area = kde_obj.intersection_area()
    assert kde_obj.executed is True
    
    # Reset vectors
    new_A = np.random.normal(2, 1, 100)
    new_B = np.random.normal(3, 1, 100)
    kde_obj.resetVectors(new_A, new_B)
    
    # Ensure reset worked and new area can be calculated
    new_area = kde_obj.intersection_area()
    
    assert kde_obj.executed is True
    assert new_area != initial_area, "Intersection areas before and after reset should not be equal"

# Test case for invalid input types (non-numeric arrays)
def test_invalid_array_type():
    A = np.array([1, 2, 3, 'invalid'])
    B = np.random.normal(1, 1, 100)

    # Initialize kde_intersection with invalid A
    kde_obj = KDEIntersection(A, B)

    with pytest.raises(ValueError):
        kde_obj.intersection_area()

# Test case for handling NaN values
def test_nan_values_in_array():
    A = np.random.normal(0, 1, 100)
    A[10] = np.nan  # Introduce NaN value
    B = np.random.normal(1, 1, 100)
    
    kde_obj = KDEIntersection(A, B)
    
    with pytest.raises(ValueError, match="Array contains NaN values"):
        kde_obj.intersection_area()

# Test case for adjustment_factor limit
def test_adjustment_factor_out_of_bounds():
    A = np.random.normal(0, 1, 100)
    B = np.random.normal(1, 1, 100)
    kde_obj = KDEIntersection(A, B)
    
    with pytest.raises(ValueError, match="Value should be between"):
        kde_obj.intersection_area(adjustment_factor=1.5)

# Test case for plotting before execution
def test_plot_without_execution():
    A = np.random.normal(0, 1, 100)
    B = np.random.normal(1, 1, 100)
    kde_obj = KDEIntersection(A, B)
    
    with pytest.raises(Exception):
        kde_obj.plot()

# Test case for edge cases (empty arrays)
def test_empty_arrays():
    A = np.array([])
    B = np.array([])

    kde_obj = KDEIntersection(A, B)

    with pytest.raises(ValueError, match="`dataset` input should have multiple elements."):
        kde_obj.intersection_area()

# Test case for very small adjustment_factor
def test_very_small_adjustment_factor():
    A = np.random.normal(0, 1, 100)
    B = np.random.normal(1, 1, 100)
    
    kde_obj = KDEIntersection(A, B)
    
    area = kde_obj.intersection_area(adjustment_factor=0.001)
    
    assert 0 <= area <= 1, "Intersection area should be within [0, 1] even with small adjustment factor"
