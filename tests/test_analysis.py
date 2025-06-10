import numpy as np
import astropy.units as u
import pytest
 
from mu.analysis import subtract_background

# ----------------------------------------------------------------
# Subtract Background Tests
# ----------------------------------------------------------------

# If no units are used, the function should work with plain numpy arrays
def test_subtraction_plain_array():
    data = np.array([[1.0, 2.0], 
                     [3.0, 4.0]])
    result = subtract_background(data, 1.0)
    expected = np.array([[0.0, 1.0], 
                         [2.0, 3.0]])
    assert np.array_equal(result, expected)


# If the same units are used, the function should work with astropy quantities
def test_subtraction_same_unit():
    data = np.array([[1.0, 2.0], 
                     [3.0, 4.0]]
                     ) * u.Jy
    bg = 0.5 * u.Jy
    result = subtract_background(data, bg)
    expected = np.array([[0.5, 1.5], 
                         [2.5, 3.5]]
                         ) * u.Jy
    assert isinstance(result, u.Quantity)
    assert result.unit == expected.unit
    assert np.allclose(result.value, expected.value)


# If different units are used, the function should convert them
def test_subtraction_quantity_convertible_unit():
    data = np.array([[1.0, 2.0], [3.0, 4.0]]) * u.Jy
    bg = 500 * u.mJy  # 0.5 Jy
    result = subtract_background(data, bg)
    expected = np.array([[0.5, 1.5], [2.5, 3.5]]) * u.Jy
    assert isinstance(result, u.Quantity)
    assert result.unit == expected.unit
    assert np.allclose(result.value, expected.value)


# If one has units and the other does not, the function should raise a TypeError
@pytest.mark.parametrize(
    "data,bg",
    [
        (np.ones((2, 2)) * u.Jy, 1.0),
        (np.ones((2, 2)), 1.0 * u.Jy),
    ],
)
def test_error_when_units_mismatch(data, bg):
    with pytest.raises(TypeError):
        subtract_background(data, bg)


# If the units are incompatible, the function should raise a ValueError
def test_error_when_units_incompatible():
    data = np.ones((2, 2)) * u.Jy
    bg = 1.0 * u.s
    with pytest.raises(ValueError):
        subtract_background(data, bg)
##########################################################


